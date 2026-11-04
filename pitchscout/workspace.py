"""Helpers the notebooks share: one call to get pitches, one to get clip features.

With demo=True everything comes from synthetic fixtures (no network), so the
notebooks run anywhere. Demo numbers do not describe any real pitcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, demo as _demo, fusion, glove, pose, video


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


@dataclass
class Clips:
    pose: pd.DataFrame                  # body pose features, one row per clip (index = pitch row)
    glove_pose: pd.DataFrame
    glove_pix: pd.DataFrame | None
    crops: dict = field(default_factory=dict)
    tracks: dict = field(default_factory=dict)   # row -> (PoseTrack, play_id)


def get_pitches(cfg: Config) -> tuple[pd.DataFrame, str]:
    if cfg.demo:
        df = data.clean(_demo.fake_statcast())
        return df, "R"
    pid = cfg.pitcher_id or data.lookup_pitcher(cfg.pitcher)
    df = data.clean(data.load_pitches(pid, cfg.start, cfg.end))
    return df, df["p_throws"].mode().iloc[0]


def get_clips(cfg: Config, df: pd.DataFrame, throws: str, log=print) -> Clips:
    if cfg.demo:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(df), min(cfg.n_clips, 300, len(df)), replace=False)
        tracks = {i: (_demo.fake_track(df.loc[i, "pitch_type"], seed=int(i)), None) for i in idx}
        crops = {i: _demo.fake_crops(df.loc[i, "pitch_type"], seed=int(i)) for i in idx}
    else:
        from .pipeline import process_clips
        extra = {}
        pose_df, tracks = process_clips(df, throws, cfg.n_clips, cfg.window_s, cfg.cutoff_s, log, extra)
        crops = extra.get("crops", {})
        idx = pose_df.index.to_numpy()
    body = pd.DataFrame({i: fusion.pose_features(tracks[i][0], throws) for i in idx}).T.loc[idx]
    gp = pd.DataFrame({i: glove.glove_pose_features(tracks[i][0], throws) for i in idx}).T.loc[idx]
    pix = pd.DataFrame({i: glove.crop_features(c) for i, c in crops.items()}).T if crops else None
    return Clips(body.astype(float), gp.astype(float), pix, crops, tracks)
