"""Local DuckDB store for everything the project produces.

One file (default ``data/pitchscout.duckdb``, override with ``PITCHSCOUT_DB``) holds:

  pitchers        one row per pitcher we've pulled
  statcast_pulls  which (pitcher, date range) pulls are already stored
  pitches         Statcast rows, keyed by (game_pk, at_bat_number, pitch_number)
  clips           one row per Savant clip we tried (play_id, status, local path)
  pose_tracks     per-clip pose metadata (fps, release frame, model window)
  pose_frames     raw MediaPipe landmarks, one row per (clip, frame, landmark)
  features        per-clip model features, long format (feature_set, feature, value)
  model_runs      one row per run (report, notebook, test), with config + summary JSON
  model_results   metrics per run, long format (analysis, model, metric, value)

No server: DuckDB is a library. Only one process can write the file at a time,
so open connections briefly (``with db.connect() as con: ...``).

``python -m pitchscout.db`` prints what's stored; ``python -m pitchscout.db make-fixture``
rebuilds the committed test sample (see tests/fixtures/sample/README.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = Path("data/pitchscout.duckdb")
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "sample"
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS pitchers (
    pitcher      BIGINT PRIMARY KEY,
    name         VARCHAR,
    throws       VARCHAR,
    synthetic    BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS statcast_pulls (
    pitcher      BIGINT,
    start_date   DATE,
    end_date     DATE,
    n_rows       INTEGER,
    source       VARCHAR,
    pulled_at    TIMESTAMP,
    PRIMARY KEY (pitcher, start_date, end_date)
);
CREATE TABLE IF NOT EXISTS pitches (
    game_pk             BIGINT,
    at_bat_number       INTEGER,
    pitch_number        INTEGER,
    pitcher             BIGINT,
    player_name         VARCHAR,
    game_date           DATE,
    game_type           VARCHAR,
    pitch_type          VARCHAR,
    pitch_name          VARCHAR,
    balls               INTEGER,
    strikes             INTEGER,
    stand               VARCHAR,
    p_throws            VARCHAR,
    inning              INTEGER,
    inning_topbot       VARCHAR,
    outs_when_up        INTEGER,
    on_1b               BIGINT,
    on_2b               BIGINT,
    on_3b               BIGINT,
    home_score          INTEGER,
    away_score          INTEGER,
    n_thruorder_pitcher INTEGER,
    release_speed       DOUBLE,
    release_spin_rate   DOUBLE,
    pfx_x               DOUBLE,
    pfx_z               DOUBLE,
    plate_x             DOUBLE,
    plate_z             DOUBLE,
    zone                INTEGER,
    description         VARCHAR,
    events              VARCHAR,
    play_id             VARCHAR,
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);
CREATE TABLE IF NOT EXISTS clips (
    play_id        VARCHAR PRIMARY KEY,
    game_pk        BIGINT,
    at_bat_number  INTEGER,
    pitch_number   INTEGER,
    video_path     VARCHAR,
    status         VARCHAR,      -- ok | no_clip | no_pose | error
    error          VARCHAR,
    updated_at     TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pose_tracks (
    play_id      VARCHAR,
    window_s     DOUBLE,
    cutoff_s     DOUBLE,
    fps          DOUBLE,
    n_frames     INTEGER,
    release      INTEGER,
    start_frame  INTEGER,
    end_frame    INTEGER,
    shot_end     INTEGER,
    created_at   TIMESTAMP,
    PRIMARY KEY (play_id, window_s, cutoff_s)
);
CREATE TABLE IF NOT EXISTS pose_frames (
    play_id     VARCHAR,
    window_s    DOUBLE,
    cutoff_s    DOUBLE,
    frame       SMALLINT,
    landmark    SMALLINT,
    x           REAL,
    y           REAL,
    visibility  REAL
);
CREATE TABLE IF NOT EXISTS features (
    play_id      VARCHAR,
    window_s     DOUBLE,
    cutoff_s     DOUBLE,
    feature_set  VARCHAR,       -- pose | glove_pose | glove_pix
    feature      VARCHAR,
    value        DOUBLE,
    PRIMARY KEY (play_id, window_s, cutoff_s, feature_set, feature)
);
CREATE TABLE IF NOT EXISTS model_runs (
    run_id       VARCHAR PRIMARY KEY,
    created_at   TIMESTAMP,
    kind         VARCHAR,       -- report | notebook:<name> | test
    pitcher      BIGINT,
    pitcher_name VARCHAR,
    start_date   DATE,
    end_date     DATE,
    n_pitches    INTEGER,
    n_clips      INTEGER,
    window_s     DOUBLE,
    cutoff_s     DOUBLE,
    synthetic    BOOLEAN,
    git_sha      VARCHAR,
    params       JSON,
    summary      JSON
);
CREATE TABLE IF NOT EXISTS model_results (
    run_id      VARCHAR,
    analysis    VARCHAR,        -- stats | clips | glove_tell | ...
    dataset     VARCHAR,
    model       VARCHAR,
    family      VARCHAR,        -- baseline | supervised | unsupervised
    evaluation  VARCHAR,
    metric      VARCHAR,
    value       DOUBLE
);
"""


def db_path(path: str | Path | None = None) -> str:
    return str(path or os.environ.get("PITCHSCOUT_DB") or DEFAULT_PATH)


def connect(path: str | Path | None = None, read_only: bool = False):
    """Open (and create if needed) the project database. ``":memory:"`` works too."""
    import duckdb

    p = db_path(path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(p, read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _columns(con, table: str) -> dict[str, str]:
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------- pitches

def upsert_pitches(con, df: pd.DataFrame, pitcher: int | None = None) -> int:
    """Insert or replace Statcast rows. Columns the schema doesn't know yet (Statcast has
    ~100) are added on the fly, so nothing from the pull is thrown away."""
    if df.empty:
        return 0
    df = df.copy()
    if "pitcher" not in df and pitcher is not None:
        df["pitcher"] = pitcher
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df.drop_duplicates(PITCH_KEY, keep="last")
    con.register("_incoming", df)
    try:
        have = _columns(con, "pitches")
        incoming = con.execute("DESCRIBE SELECT * FROM _incoming").fetchall()
        for name, typ, *_ in incoming:
            if name not in have:
                typ = "VARCHAR" if typ in ("NULL", '"NULL"') else typ
                con.execute(f'ALTER TABLE pitches ADD COLUMN "{name}" {typ}')
        con.execute("INSERT OR REPLACE INTO pitches BY NAME SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    if "pitcher" in df:
        for pid, g in df.groupby("pitcher"):
            name = g["player_name"].dropna().iloc[0] if "player_name" in g and g["player_name"].notna().any() else None
            throws = g["p_throws"].mode().iloc[0] if "p_throws" in g and g["p_throws"].notna().any() else None
            add_pitcher(con, int(pid), name, throws)
    return len(df)


def add_pitcher(con, pitcher: int, name: str | None = None, throws: str | None = None,
                synthetic: bool | None = None) -> None:
    con.execute("""
        INSERT INTO pitchers VALUES (?, ?, ?, coalesce(?, FALSE))
        ON CONFLICT (pitcher) DO UPDATE SET
            name = coalesce(excluded.name, pitchers.name),
            throws = coalesce(excluded.throws, pitchers.throws),
            synthetic = CASE WHEN ? IS NULL THEN pitchers.synthetic ELSE excluded.synthetic END
    """, [pitcher, name, throws, synthetic, synthetic])


def read_pitches(con, pitcher: int | None = None, start: str | None = None,
                 end: str | None = None) -> pd.DataFrame:
    where, args = [], []
    if pitcher is not None:
        where.append("pitcher = ?"); args.append(int(pitcher))
    if start:
        where.append("game_date >= ?"); args.append(start)
    if end:
        where.append("game_date <= ?"); args.append(end)
    sql = "SELECT * FROM pitches" + (" WHERE " + " AND ".join(where) if where else "")
    df = con.execute(sql + " ORDER BY game_date, game_pk, at_bat_number, pitch_number", args).df()
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def record_pull(con, pitcher: int, start: str, end: str, n_rows: int, source: str = "statcast") -> None:
    con.execute("INSERT OR REPLACE INTO statcast_pulls VALUES (?, ?, ?, ?, ?, ?)",
                [pitcher, start, end, n_rows, source, _now()])


def has_pull(con, pitcher: int, start: str, end: str) -> bool:
    """True if a stored pull already covers [start, end] for this pitcher."""
    return con.execute("""SELECT count(*) FROM statcast_pulls
                          WHERE pitcher = ? AND start_date <= ? AND end_date >= ?""",
                       [pitcher, start, end]).fetchall()[0][0] > 0


def set_play_ids(con, df: pd.DataFrame) -> None:
    """Write play_ids found via the game feed back onto the pitch rows."""
    if "play_id" not in df:
        return
    ids = df.loc[df["play_id"].notna(), PITCH_KEY + ["play_id"]]
    if ids.empty:
        return
    con.register("_ids", ids)
    try:
        con.execute("""UPDATE pitches SET play_id = _ids.play_id FROM _ids
                       WHERE pitches.game_pk = _ids.game_pk AND pitches.at_bat_number = _ids.at_bat_number
                         AND pitches.pitch_number = _ids.pitch_number""")
    finally:
        con.unregister("_ids")


# ---------------------------------------------------------------- clips + pose

def record_clip(con, play_id: str, row: pd.Series | dict | None = None, status: str = "ok",
                video_path: str | Path | None = None, error: str | None = None) -> None:
    row = row if row is not None else {}
    key = [int(row[k]) if k in row and pd.notna(row[k]) else None for k in PITCH_KEY]
    con.execute("INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [play_id, *key, str(video_path) if video_path else None, status, error, _now()])


def save_track(con, play_id: str, track, window_s: float, cutoff_s: float) -> None:
    key = [play_id, float(window_s), float(cutoff_s)]
    con.execute("DELETE FROM pose_frames WHERE play_id = ? AND window_s = ? AND cutoff_s = ?", key)
    f = np.asarray(track.frames, np.float32)
    T, L, _ = f.shape
    frames = pd.DataFrame({
        "play_id": play_id, "window_s": key[1], "cutoff_s": key[2],
        "frame": np.repeat(np.arange(T, dtype=np.int16), L),
        "landmark": np.tile(np.arange(L, dtype=np.int16), T),
        "x": f[..., 0].ravel(), "y": f[..., 1].ravel(), "visibility": f[..., 2].ravel(),
    })
    con.register("_frames", frames)
    try:
        con.execute("INSERT INTO pose_frames BY NAME SELECT * FROM _frames")
    finally:
        con.unregister("_frames")
    con.execute("INSERT OR REPLACE INTO pose_tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [*key, float(track.fps), T, int(track.release), int(track.start), int(track.end),
                 int(track.shot_end), _now()])


def load_track(con, play_id: str, window_s: float, cutoff_s: float):
    """PoseTrack from the database, or None if this clip/window isn't stored."""
    from .pose import PoseTrack

    key = [play_id, float(window_s), float(cutoff_s)]
    meta = con.execute("""SELECT fps, n_frames, release, start_frame, end_frame, shot_end FROM pose_tracks
                          WHERE play_id = ? AND window_s = ? AND cutoff_s = ?""", key).fetchall()
    if not meta:
        return None
    fps, T, release, start, end, shot_end = meta[0]
    pts = con.execute("""SELECT frame, landmark, x, y, visibility FROM pose_frames
                         WHERE play_id = ? AND window_s = ? AND cutoff_s = ?""", key).fetchnumpy()
    L = int(pts["landmark"].max()) + 1 if len(pts["landmark"]) else 33
    frames = np.zeros((T, L, 3), np.float64)
    frames[pts["frame"], pts["landmark"]] = np.stack([pts["x"], pts["y"], pts["visibility"]], -1)
    return PoseTrack(frames, float(fps), int(release), int(start), int(end), int(shot_end))


def stored_tracks(con, window_s: float, cutoff_s: float, pitcher: int | None = None) -> pd.DataFrame:
    """Which pitches already have a pose track for this window (joined to the pitch key)."""
    sql = """SELECT p.game_pk, p.at_bat_number, p.pitch_number, p.pitch_type, t.play_id
             FROM pose_tracks t JOIN pitches p USING (play_id)
             WHERE t.window_s = ? AND t.cutoff_s = ?"""
    args = [float(window_s), float(cutoff_s)]
    if pitcher is not None:
        sql += " AND p.pitcher = ?"; args.append(int(pitcher))
    return con.execute(sql, args).df()


def save_features(con, play_id: str, feats: dict, feature_set: str, window_s: float, cutoff_s: float) -> None:
    if not feats:
        return
    df = pd.DataFrame({"play_id": play_id, "window_s": float(window_s), "cutoff_s": float(cutoff_s),
                       "feature_set": feature_set, "feature": list(feats),
                       "value": np.asarray(list(feats.values()), dtype=float)})
    con.register("_feats", df)
    try:
        con.execute("INSERT OR REPLACE INTO features BY NAME SELECT * FROM _feats")
    finally:
        con.unregister("_feats")


def load_features(con, feature_set: str, window_s: float, cutoff_s: float,
                  play_ids: list[str] | None = None) -> pd.DataFrame:
    """Wide table: one row per play_id, one column per feature."""
    df = con.execute("""SELECT play_id, feature, value FROM features
                        WHERE feature_set = ? AND window_s = ? AND cutoff_s = ?""",
                     [feature_set, float(window_s), float(cutoff_s)]).df()
    if play_ids is not None:
        df = df[df["play_id"].isin(play_ids)]
    return df.pivot(index="play_id", columns="feature", values="value")


# ---------------------------------------------------------------- runs + results

def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _json(v) -> str | None:
    if v is None:
        return None
    return json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def start_run(con, kind: str, pitcher: int | None = None, pitcher_name: str | None = None,
              start: str | None = None, end: str | None = None, n_pitches: int | None = None,
              n_clips: int | None = None, window_s: float | None = None, cutoff_s: float | None = None,
              synthetic: bool = False, params: dict | None = None) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    con.execute("INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, _now(), kind, pitcher, pitcher_name, start, end, n_pitches, n_clips,
                 window_s, cutoff_s, synthetic, git_sha(), _json(params), None])
    return run_id


def finish_run(con, run_id: str, summary: dict | None = None, n_clips: int | None = None) -> None:
    con.execute("UPDATE model_runs SET summary = ?, n_clips = coalesce(?, n_clips) WHERE run_id = ?",
                [_json(summary), n_clips, run_id])


def save_results(con, run_id: str, analysis: str, rows: list[dict]) -> int:
    """rows: dicts with model/dataset/family/evaluation plus any numeric metrics
    (accuracy, log_loss, auc, perm_p, ...). Each numeric metric becomes one row."""
    meta = ("dataset", "model", "family", "evaluation")
    out = []
    for r in rows:
        for k, v in r.items():
            if k in meta or k == "pitcher":
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            out.append({"run_id": run_id, "analysis": analysis, **{m: r.get(m) for m in meta},
                        "metric": k, "value": v})
    if not out:
        return 0
    df = pd.DataFrame(out)
    con.register("_res", df)
    try:
        con.execute("INSERT INTO model_results BY NAME SELECT * FROM _res")
    finally:
        con.unregister("_res")
    return len(out)


def results(con, run_id: str | None = None, analysis: str | None = None) -> pd.DataFrame:
    """Wide results table (one row per model, one column per metric). Defaults to the latest run."""
    if run_id is None:
        row = con.execute("SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1").fetchall()
        if not row:
            return pd.DataFrame()
        run_id = row[0][0]
    sql, args = "SELECT * FROM model_results WHERE run_id = ?", [run_id]
    if analysis:
        sql += " AND analysis = ?"; args.append(analysis)
    df = con.execute(sql, args).df()
    if df.empty:
        return df
    keys = ["analysis", "dataset", "model", "family", "evaluation"]
    df[keys] = df[keys].fillna("")
    order = df.drop_duplicates(keys)[keys]
    wide = df.pivot_table(index=keys, columns="metric", values="value", aggfunc="first")
    wide.columns.name = None
    return order.merge(wide.reset_index(), on=keys, how="left")


def summary(con) -> pd.DataFrame:
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    return pd.DataFrame({"table": tables,
                         "rows": [con.execute(f"SELECT count(*) FROM {t}").fetchall()[0][0] for t in tables]})


# ---------------------------------------------------------------- fixture

FIXTURE_TABLES = ("pitchers", "pitches", "clips", "pose_tracks", "pose_frames")


def load_fixture(con, fixture_dir: Path = FIXTURE_DIR) -> None:
    """Load the committed sample (parquet per table) into an open database."""
    for t in FIXTURE_TABLES:
        p = fixture_dir / f"{t}.parquet"
        if p.exists():
            verb = "INSERT" if t == "pose_frames" else "INSERT OR REPLACE"
            con.execute(f"{verb} INTO {t} BY NAME SELECT * FROM read_parquet(?)", [str(p)])


def fixture_db(fixture_dir: Path = FIXTURE_DIR):
    """In-memory database preloaded with the committed sample. Nothing touches the network."""
    con = connect(":memory:")
    load_fixture(con, fixture_dir)
    return con


def make_fixture(out: Path = FIXTURE_DIR, source: str | None = None, pitcher: int | None = None,
                 n_games: int = 60, n_clips: int = 150, window_s: float = 1.5, cutoff_s: float = 0.10,
                 seed: int = 7) -> Path:
    """Write the test sample.

    source=None: synthetic, Statcast-shaped rows (see demo.miller_shaped_statcast).
    source=<db path>: real rows copied out of your own database (the latest n_games for
    `pitcher`, plus any pose tracks stored for them). Clips themselves are never copied.
    """
    out.mkdir(parents=True, exist_ok=True)
    tmp = connect(":memory:")
    if source is None:
        from . import demo
        df = demo.miller_shaped_statcast(n_games=n_games, seed=seed)
        upsert_pitches(tmp, df)
        add_pitcher(tmp, int(df["pitcher"].iloc[0]), demo.SYNTH_NAME, "R", synthetic=True)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), min(n_clips, len(df)), replace=False)
        for i in sorted(idx):
            r = df.iloc[i]
            tr = demo.fake_track(r["pitch_type"], seed=int(i))
            save_track(tmp, r["play_id"], tr, window_s, cutoff_s)
            record_clip(tmp, r["play_id"], r, status="ok")
    else:
        if pitcher is None:
            raise ValueError("pitcher is required when copying from a real database")
        src = connect(source, read_only=True)
        games = src.execute("""SELECT game_pk FROM pitches WHERE pitcher = ? GROUP BY game_pk
                               ORDER BY max(game_date) DESC LIMIT ?""", [pitcher, n_games]).df()["game_pk"]
        df = src.execute("SELECT * FROM pitches WHERE pitcher = ? AND list_contains(?, game_pk)",
                         [pitcher, list(map(int, games))]).df()
        upsert_pitches(tmp, df)
        for t in ("pitchers",):
            tmp.register("_p", src.execute(f"SELECT * FROM {t} WHERE pitcher = ?", [pitcher]).df())
            tmp.execute(f"INSERT OR REPLACE INTO {t} BY NAME SELECT * FROM _p"); tmp.unregister("_p")
        ids = [p for p in df["play_id"].dropna()]
        for t in ("clips", "pose_tracks", "pose_frames"):
            part = src.execute(f"SELECT * FROM {t} WHERE list_contains(?, play_id)", [ids]).df()
            if t == "clips":
                part["video_path"] = None   # never ship MLB video paths/files
            if t != "clips":
                part = part[(part["window_s"] == window_s) & (part["cutoff_s"] == cutoff_s)]
            verb = "INSERT" if t == "pose_frames" else "INSERT OR REPLACE"
            tmp.register("_p", part)
            tmp.execute(f"{verb} INTO {t} BY NAME SELECT * FROM _p"); tmp.unregister("_p")
        src.close()
    for t in FIXTURE_TABLES:
        tmp.execute(f"COPY (SELECT * FROM {t}) TO '{out / (t + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.close()
    return out


def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m pitchscout.db")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("info", help="row counts and recent runs")
    mf = sub.add_parser("make-fixture", help="rebuild tests/fixtures/sample")
    mf.add_argument("--from-db", help="copy real rows out of this database instead of generating synthetic ones")
    mf.add_argument("--pitcher", type=int, help="MLBAM id (with --from-db)")
    mf.add_argument("--games", type=int, default=60)
    mf.add_argument("--clips", type=int, default=150)
    mf.add_argument("--out", type=Path, default=FIXTURE_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "make-fixture":
        p = make_fixture(a.out, a.from_db, a.pitcher, a.games, a.clips)
        print(f"wrote {p}")
        return
    with connect(read_only=Path(db_path()).exists()) as con:
        print(db_path())
        print(summary(con).to_string(index=False))
        runs = con.execute("""SELECT run_id, kind, pitcher_name, n_pitches, n_clips, synthetic
                              FROM model_runs ORDER BY created_at DESC LIMIT 10""").df()
        if len(runs):
            print("\nrecent runs\n" + runs.to_string(index=False))


if __name__ == "__main__":
    _main()
