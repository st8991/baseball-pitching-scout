"""Helpers the notebooks share: one call to get pitches, one to get clip features.

Real runs read and write the local database (pitchscout.db). With demo=True everything
comes from the committed synthetic sample in tests/fixtures/sample, loaded into an
in-memory database (no network, nothing written to disk). Demo numbers do not describe
any real pitcher.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, db, demo as _demo, fusion, glove, pose, video


@dataclass
class Config:
    pitcher: str = "Mason Miller"
    pitcher_id: int | None = None
    start: str = "2024-03-28"
    end: str = "2025-09-28"
    n_clips: int = 600
    window_s: float = 1.5
    cutoff_s: float = 0.10
    demo: bool = False
    db_path: str | None = None          # default: $PITCHSCOUT_DB or data/pitchscout.duckdb


@dataclass
class Clips:
    pose: pd.DataFrame                  # body pose features, one row per clip (index = pitch row)
    glove_pose: pd.DataFrame
    glove_pix: pd.DataFrame | None
    crops: dict = field(default_factory=dict)
    tracks: dict = field(default_factory=dict)   # row -> (PoseTrack, play_id)


_FIXTURE_CON = None


@contextmanager
def database(cfg: Config):
    """`with database(cfg) as con:` -> the project database (or the in-memory sample in demo mode).
    Real connections are closed afterwards so scout.py can write while a notebook is open."""
    global _FIXTURE_CON
    if cfg.demo:
        if _FIXTURE_CON is None:
            _FIXTURE_CON = db.fixture_db()
        yield _FIXTURE_CON
        return
    con = db.connect(cfg.db_path)
    try:
        yield con
    finally:
        con.close()


def get_pitches(cfg: Config) -> tuple[pd.DataFrame, str]:
    if cfg.demo:
        with database(cfg) as con:
            df = data.clean(db.read_pitches(con, _demo.SYNTH_ID))
        return df, "R"
    with database(cfg) as con:
        pid = cfg.pitcher_id or data.lookup_pitcher(cfg.pitcher)
        df = data.clean(data.load_pitches(pid, cfg.start, cfg.end, con=con))
    return df, df["p_throws"].mode().iloc[0]


def get_clips(cfg: Config, df: pd.DataFrame, throws: str, log=print) -> Clips:
    if cfg.demo:
        # Pose tracks come from the sample database; glove crops are drawn on the fly.
        with database(cfg) as con:
            have = set(con.execute("SELECT play_id FROM pose_tracks WHERE window_s = ? AND cutoff_s = ?",
                                   [cfg.window_s, cfg.cutoff_s]).df()["play_id"])
            rows = df.index[df["play_id"].isin(have)][: cfg.n_clips]
            tracks = {i: (db.load_track(con, df.loc[i, "play_id"], cfg.window_s, cfg.cutoff_s),
                          df.loc[i, "play_id"]) for i in rows}
        idx = np.asarray(list(tracks))
        crops = {i: _demo.fake_crops(df.loc[i, "pitch_type"], seed=int(i)) for i in idx}
    else:
        from .pipeline import process_clips
        extra = {}
        with database(cfg) as con:
            pose_df, tracks = process_clips(df, throws, cfg.n_clips, cfg.window_s, cfg.cutoff_s, log,
                                            extra, con=con)
        crops = extra.get("crops", {})
        idx = pose_df.index.to_numpy()
    body = pd.DataFrame({i: fusion.pose_features(tracks[i][0], throws) for i in idx}).T.loc[idx]
    gp = pd.DataFrame({i: glove.glove_pose_features(tracks[i][0], throws) for i in idx}).T.loc[idx]
    pix = pd.DataFrame({i: glove.crop_features(c) for i, c in crops.items()}).T if crops else None
    return Clips(body.astype(float), gp.astype(float), pix, crops, tracks)


def save_run(cfg: Config, kind: str, df: pd.DataFrame, results: dict[str, list[dict]],
             n_clips: int | None = None, summary: dict | None = None, params: dict | None = None) -> str:
    """Record a notebook's numbers in model_runs/model_results. `results` maps an analysis
    name (e.g. "glove_tell") to rows like {"model": ..., "auc": ..., "perm_p": ...}."""
    with database(cfg) as con:
        pid = int(df["pitcher"].iloc[0]) if "pitcher" in df and df["pitcher"].notna().any() else cfg.pitcher_id
        run_id = db.start_run(con, kind, pid, cfg.pitcher, cfg.start, cfg.end, len(df), n_clips,
                              cfg.window_s, cfg.cutoff_s, synthetic=cfg.demo, params=params)
        for analysis, rows in results.items():
            db.save_results(con, run_id, analysis, rows)
        db.finish_run(con, run_id, summary)
    return run_id
