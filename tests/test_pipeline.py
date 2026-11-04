import numpy as np
import pandas as pd
import pytest

from pitchscout import baseline, data, features, fusion, pose, unsupervised, viz
from pitchscout.report import build
from pitchscout.demo import TYPES, fake_statcast, fake_track


@pytest.fixture(scope="module")
def df():
    return data.clean(fake_statcast())


def test_prev_pitch_stays_within_at_bat(df):
    X = features.situation_features(df)
    first = df["pitch_number"] == 1
    assert (X.loc[first, "prev_pitch"] == "none").all()
    second = df.index[df["pitch_number"] == 2]
    assert (X.loc[second, "prev_pitch"].astype(str).to_numpy() == df.loc[second - 1, "pitch_type"].to_numpy()).all()


def test_time_split_holds_out_latest_games(df):
    tr, te = features.time_split(df)
    assert df.loc[tr, "game_date"].max() <= df.loc[te, "game_date"].min()
    assert not set(df.loc[tr, "game_pk"]) & set(df.loc[te, "game_pk"])


def test_baseline_beats_naive_on_planted_signal(df):
    res = baseline.fit_baseline(df)
    m = res.metrics
    assert m["stats_model"]["log_loss"] < m["naive_mix"]["log_loss"]
    assert res.oof_proba.shape == (len(df), len(res.classes))
    assert np.allclose(res.oof_proba.sum(axis=1), 1, atol=1e-6)


def test_tendencies_rows_sum_to_one(df):
    t = baseline.tendencies(df)
    filled = t["by_count"][t["n_by_count"] > 0]
    assert np.allclose(filled.sum(axis=1), 1)


def test_normalize_mirrors_lefties():
    r = pose.normalize(fake_track("FF", throws="R").frames, "R")
    l_tr = fake_track("FF", throws="R")
    l_tr.frames[..., 0] = 1 - l_tr.frames[..., 0]   # mirror image of a righty
    swap = list(range(33))
    for a, b in [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]:
        swap[a], swap[b] = b, a
    l_tr.frames = l_tr.frames[:, swap]
    l = pose.normalize(l_tr.frames, "L")
    assert np.allclose(r, l, atol=0.05)


def test_release_found_near_true_release():
    tr = fake_track("FF")
    assert abs(pose.find_release(tr.frames, tr.fps, "R") - tr.release) <= 3


def test_first_shot_end_detects_cut():
    a = [np.full((60, 80, 3), (0, 120, 0), np.uint8)] * 10
    b = [np.full((60, 80, 3), (200, 30, 30), np.uint8)] * 5
    assert pose.first_shot_end(a + b) == 10


def test_pose_features_exclude_post_cutoff_frames():
    tr = fake_track("FF")
    f1 = fusion.pose_features(tr, "R")
    tr.frames[tr.end:, :, :2] += 0.5          # change only frames after the cutoff
    f2 = fusion.pose_features(tr, "R")
    keys = [k for k in f1 if not k.startswith("tempo")]
    assert all(np.isclose(f1[k], f2[k]) for k in keys)


@pytest.fixture(scope="module")
def clip_models(df):
    base = baseline.fit_baseline(df)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), 240, replace=False)
    y = df.loc[idx, "pitch_type"].to_numpy()
    feats = pd.DataFrame([fusion.pose_features(fake_track(t, seed=i), "R") for i, t in zip(idx, y)], index=idx)
    games = df.loc[idx, "game_pk"].to_numpy()
    fus = fusion.fit_fusion(feats, base.oof_proba[idx], y, games, base.classes)
    uns = unsupervised.fit_unsupervised(feats, y, games, base.classes, prior=base.oof_proba[idx])
    return base, fus, uns


def test_fusion_uses_planted_tell(clip_models):
    _, fus, _ = clip_models
    assert fus.contrib.shape[0] == len(fus.y)
    imp = fusion.part_importance(fus)
    # The planted changeup tell is glove-hand height.
    top_body = [p for p in imp.index if p not in ("Stats prior",)][:3]
    assert "Glove hand" in top_body or "Hand separation" in top_body
    assert fus.metrics["stats_plus_vision"]["log_loss"] < fus.metrics["stats_only"]["log_loss"]


def test_unsupervised_outputs(clip_models):
    _, fus, uns = clip_models
    assert uns.k >= 2
    assert np.allclose(uns.proba.sum(axis=1), 1)
    assert 0 <= uns.alignment["purity"] <= 1
    assert set(uns.metrics) == {"clusters_only", "stats_plus_clusters"}
    assert uns.embedding.shape == (len(fus.y), 2)


def test_overlay_draws(tmp_path):
    import cv2
    tr = fake_track("CH")
    clip = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(len(tr.frames)):
        vw.write(np.full((240, 320, 3), 90, np.uint8))
    vw.release()
    w = viz.joint_weights({"Glove hand": np.ones(fusion.N_BINS)}, "R")
    viz.render_overlay(clip, tr, w, "R", tmp_path / "o.mp4", tmp_path / "o.png")
    assert (tmp_path / "o.png").exists()


def test_report_builds(tmp_path, df, clip_models):
    base, fus, uns = clip_models
    out = build(tmp_path / "r.html", "Test Pitcher", "R", "2024-04-01", "2024-09-01", df,
                baseline.tendencies(df), base, fus, uns, synthetic=True)
    html = out.read_text()
    assert "Test Pitcher" in html and "__DATA__" not in html


def test_glove_tell_detects_planted_glove_signal(df, clip_models, tmp_path):
    from pitchscout import glove
    base, fus, uns = clip_models
    idx = fus.index
    y = fus.y
    tracks = {i: fake_track(t, seed=int(i)) for i, t in zip(idx, y)}
    gp = pd.DataFrame({i: glove.glove_pose_features(tr, "R") for i, tr in tracks.items()}).T.loc[idx]
    body = pd.DataFrame([fusion.pose_features(tracks[i], "R") for i in idx], index=idx)
    games = df.loc[idx, "game_pk"].to_numpy()
    # Fake crops: changeups get a brighter glove patch.
    crops = {i: np.full((6, glove.CROP, glove.CROP, 3), 160 if t == "CH" else 90, np.uint8) for i, t in zip(idx, y)}
    pix = pd.DataFrame({i: glove.crop_features(c) for i, c in crops.items()}).T
    res = glove.run_tell_test(gp, pix, body, y, games, positive={"CH"}, negative={"FF"},
                              crops=crops, n_perm=100)
    g = res.sets["Glove (pose)"]
    assert g["auc"] > 0.8 and g["perm_p"] < 0.01
    assert res.sets["Glove (pose)"]["auc"] > res.sets["Body without glove (control)"]["auc"]
    assert res.verdict.startswith("Supported")
    assert glove.crop_panel(res, tmp_path / "p.png").exists()
    out = build(tmp_path / "r.html", "Test Pitcher", "R", "2024-04-01", "2024-09-01", df,
                baseline.tendencies(df), base, fus, uns, synthetic=True, tell=res,
                tell_panel=tmp_path / "p.png")
    assert "Glove tell test" in out.read_text()


def test_glove_crops_from_video(tmp_path):
    import cv2
    from pitchscout import glove
    tr = fake_track("FF")
    clip = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(len(tr.frames)):
        vw.write(np.full((240, 320, 3), 90, np.uint8))
    vw.release()
    c = glove.glove_crops(clip, tr, "R")
    assert c.shape == (6, glove.CROP, glove.CROP, 3)
