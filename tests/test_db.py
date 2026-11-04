"""The local database and the committed sample. Everything here runs offline."""
import numpy as np
import pandas as pd
import pytest

from pitchscout import data, db, demo, pipeline, video
from pitchscout.workspace import Config, get_clips, get_pitches, save_run


@pytest.fixture()
def con():
    c = db.fixture_db()
    yield c
    c.close()


@pytest.fixture()
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tried to reach the network")
    monkeypatch.setattr(video, "download_clip", boom)
    monkeypatch.setattr(video, "game_play_ids", boom)
    monkeypatch.setattr(data, "lookup_pitcher", boom)
    import pybaseball
    monkeypatch.setattr(pybaseball, "statcast_pitcher", boom)


def test_fixture_loads(con):
    counts = db.summary(con).set_index("table")["rows"]
    assert counts["pitches"] > 800 and counts["pose_tracks"] >= 100
    df = db.read_pitches(con, demo.SYNTH_ID)
    assert not df.duplicated(db.PITCH_KEY).any()
    assert df["play_id"].notna().all()
    assert set(df["pitch_type"]) == {"FF", "SL", "CH"}
    assert con.execute("SELECT synthetic FROM pitchers").fetchall()[0][0] is True
    # every stored track points at a pitch
    assert con.execute("SELECT count(*) FROM pose_tracks t ANTI JOIN pitches p USING (play_id)").fetchall()[0][0] == 0


def test_cleaned_fixture_works_with_stats_model(con):
    from pitchscout import baseline
    df = data.clean(db.read_pitches(con))
    res = baseline.fit_baseline(df)
    assert res.metrics["stats_model"]["log_loss"] < res.metrics["naive_mix"]["log_loss"] + 0.02


def test_upsert_is_idempotent_and_keeps_new_columns(con):
    df = db.read_pitches(con).head(20).copy()
    n = con.execute("SELECT count(*) FROM pitches").fetchall()[0][0]
    df["spin_axis"] = 200.0                    # a Statcast column the schema doesn't list
    df["release_speed"] = 50.0
    db.upsert_pitches(con, df)
    db.upsert_pitches(con, df)
    assert con.execute("SELECT count(*) FROM pitches").fetchall()[0][0] == n
    back = db.read_pitches(con).set_index(db.PITCH_KEY).loc[df.set_index(db.PITCH_KEY).index]
    assert (back["spin_axis"] == 200).all() and (back["release_speed"] == 50).all()


def test_track_round_trip(con):
    tr = demo.fake_track("CH", seed=3)
    db.save_track(con, "abc", tr, 1.5, 0.1)
    db.save_track(con, "abc", tr, 1.5, 0.1)    # overwrite, not duplicate
    back = db.load_track(con, "abc", 1.5, 0.1)
    assert np.allclose(back.frames, tr.frames, atol=1e-6)
    assert (back.release, back.start, back.end, back.fps) == (tr.release, tr.start, tr.end, tr.fps)
    assert db.load_track(con, "abc", 1.5, 0.2) is None
    n = con.execute("SELECT count(*) FROM pose_frames WHERE play_id = 'abc'").fetchall()[0][0]
    assert n == tr.frames.shape[0] * tr.frames.shape[1]


def test_features_round_trip(con):
    db.save_features(con, "p1", {"a": 1.0, "b": 2.0}, "pose", 1.5, 0.1)
    db.save_features(con, "p2", {"a": 3.0, "b": 4.0}, "pose", 1.5, 0.1)
    db.save_features(con, "p1", {"a": 9.0, "b": 2.0}, "pose", 1.5, 0.1)
    w = db.load_features(con, "pose", 1.5, 0.1)
    assert w.loc["p1", "a"] == 9.0 and w.loc["p2", "b"] == 4.0 and len(w) == 2


def test_load_pitches_serves_stored_pull_offline(con, no_network):
    db.record_pull(con, demo.SYNTH_ID, "2025-01-01", "2025-12-31", 0)
    df = data.load_pitches(demo.SYNTH_ID, "2025-04-01", "2025-06-30", con=con)
    assert len(df) and df["game_date"].between("2025-04-01", "2025-06-30").all()


def test_load_pitches_imports_legacy_parquet(tmp_path, monkeypatch, no_network):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    raw = demo.miller_shaped_statcast(n_games=3).drop(columns=["pitcher"])
    raw.to_parquet(tmp_path / f"statcast_42_2025-01-01_2025-12-31.parquet")
    c = db.connect(":memory:")
    df = data.load_pitches(42, "2025-01-01", "2025-12-31", con=c)
    assert len(df) == len(raw) and (df["pitcher"] == 42).all()
    assert db.has_pull(c, 42, "2025-01-01", "2025-12-31")


def test_attach_play_ids_keeps_row_labels(monkeypatch, tmp_path):
    df = demo.miller_shaped_statcast(n_games=2)
    ids = df[db.PITCH_KEY + ["play_id"]].assign(pitcher=1)
    sample = df.drop(columns=["play_id"]).sample(15, random_state=0)
    monkeypatch.setattr(video, "game_play_ids", lambda pk: ids[ids.game_pk == pk])
    monkeypatch.setattr(video, "CLIPS", tmp_path / "clips")
    out = video.attach_play_ids(sample, pause=0)
    assert list(out.index) == list(sample.index)
    assert (out["play_id"] == df.loc[sample.index, "play_id"]).all()


def test_runs_and_results(con):
    rid = db.start_run(con, "test", demo.SYNTH_ID, "x", n_pitches=10, synthetic=True, params={"a": 1})
    db.save_results(con, rid, "stats", [{"model": "m1", "family": "baseline", "accuracy": 0.5, "log_loss": 1.0},
                                        {"model": "m2", "family": "supervised", "accuracy": 0.6, "note": "skip"}])
    db.finish_run(con, rid, {"verdict": "ok", "arr": np.float64(1.5)})
    r = db.results(con).set_index("model")
    assert r.loc["m1", "log_loss"] == 1.0 and r.loc["m2", "accuracy"] == 0.6
    assert "note" not in r


def test_process_clips_reads_stored_tracks_offline(con, no_network):
    df = data.clean(db.read_pitches(con))
    have = set(con.execute("SELECT play_id FROM pose_tracks").df()["play_id"])
    sub = df[df["play_id"].isin(have)]
    gx = {}
    feats, tracks = pipeline.process_clips(sub, "R", 60, 1.5, 0.1, log=lambda *_: None, glove_extra=gx, con=con)
    assert len(feats) == 60
    # rows line up with the right pitch
    for i, (tr, pid) in tracks.items():
        assert df.loc[i, "play_id"] == pid
    assert len(db.load_features(con, "pose", 1.5, 0.1)) == 60
    assert len(db.load_features(con, "glove_pose", 1.5, 0.1)) == 60


def test_full_report_run_from_database(tmp_path, monkeypatch, no_network):
    path = tmp_path / "t.duckdb"
    with db.connect(path) as c:
        db.load_fixture(c)
        db.record_pull(c, demo.SYNTH_ID, "2025-01-01", "2025-12-31", 0)
        have = set(c.execute("SELECT play_id FROM pose_tracks").df()["play_id"])
    # Only sample pitches that have a stored track, so nothing needs downloading.
    orig = video.sample_for_video
    monkeypatch.setattr(video, "sample_for_video", lambda df, n, seed=0: orig(df[df.play_id.isin(have)], n, seed))
    monkeypatch.setattr(pipeline, "render_examples", lambda *a, **k: {})   # needs real video
    out = pipeline.run("Synthetic Pitcher", "2025-01-01", "2025-12-31", n_clips=150, out_dir=tmp_path / "r",
                       pitcher_id=demo.SYNTH_ID, tell=True, positive={"FF"}, n_perm=20, db_path=path,
                       log=lambda *_: None)
    assert out.exists()
    with db.connect(path, read_only=True) as c:
        run = c.execute("SELECT kind, n_clips, summary FROM model_runs").fetchall()
        assert len(run) == 1 and run[0][0] == "report" and run[0][1] >= 100
        r = db.results(c)
    assert {"stats", "clips", "glove_tell"} <= set(r["analysis"])
    # same numbers as the CSV a paper would cite
    csv = pd.read_csv(tmp_path / "r" / "model_comparison.csv").set_index("model")
    stats = r[r.analysis == "stats"].set_index("model")
    assert np.isclose(stats.loc["LightGBM situation model", "log_loss"], csv.loc["LightGBM situation model", "log_loss"])


def test_workspace_demo_uses_sample(no_network):
    cfg = Config(demo=True, n_clips=80)
    df, throws = get_pitches(cfg)
    clips = get_clips(cfg, df, throws)
    assert len(clips.pose) == 80 and clips.glove_pix is not None
    rid = save_run(cfg, "notebook:test", df, {"stats": [{"model": "m", "accuracy": 0.5}]})
    from pitchscout.workspace import database
    with database(cfg) as c:
        assert c.execute("SELECT synthetic FROM model_runs WHERE run_id = ?", [rid]).fetchall()[0][0] is True
