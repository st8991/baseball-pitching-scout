"""Statcast data access for one pitcher, cached locally.

Uses pybaseball, which scrapes baseballsavant.mlb.com. The machine running this
must be able to reach that host.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path("cache")

# Statcast pitch_type codes -> readable names.
PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
    "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle Curve", "CH": "Changeup",
    "FS": "Splitter", "FO": "Forkball", "SC": "Screwball", "KN": "Knuckleball",
}


def lookup_pitcher(name: str) -> int:
    """Resolve 'First Last' to an MLBAM id."""
    from pybaseball import playerid_lookup

    first, last = name.strip().split(" ", 1)
    hits = playerid_lookup(last, first, fuzzy=True)
    hits = hits.dropna(subset=["key_mlbam"])
    if hits.empty:
        raise ValueError(f"No MLBAM id found for {name!r}")
    # Prefer the most recent career.
    hits = hits.sort_values("mlb_played_last", ascending=False)
    return int(hits.iloc[0]["key_mlbam"])


def load_pitches(pitcher_id: int, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """All pitches thrown by `pitcher_id` between start and end (YYYY-MM-DD)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"statcast_{pitcher_id}_{start}_{end}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from pybaseball import statcast_pitcher

    df = statcast_pitcher(start, end, pitcher_id)
    df.to_parquet(path)
    return df


def clean(df: pd.DataFrame, min_share: float = 0.02) -> pd.DataFrame:
    """Regular-season pitches in game order, dropping rare/unknown pitch types."""
    df = df.copy()
    if "game_type" in df:
        df = df[df["game_type"] == "R"]
    df = df.dropna(subset=["pitch_type"])
    df = df[~df["pitch_type"].isin(["PO", "IN", "EP", "AB", "UN", "FA"])]
    share = df["pitch_type"].value_counts(normalize=True)
    df = df[df["pitch_type"].isin(share[share >= min_share].index)]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    return df.reset_index(drop=True)
