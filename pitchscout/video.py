"""Find and download the Baseball Savant video clip for each Statcast pitch.

Each pitch on Savant has a `play_id` (UUID). The game feed endpoint maps
(game_pk, at_bat_number, pitch_number) -> play_id, and the sporty-videos page
for a play_id embeds the mp4 URL.

Clips are MLB property; this is for personal research use. Keep downloads
modest and cached, and don't redistribute them.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

SAVANT = "https://baseballsavant.mlb.com"
CLIPS = Path("cache/clips")
HEADERS = {"User-Agent": "pitchscout/0.1 (research prototype)"}
MP4_RE = re.compile(r'https://[^"\']+\.mp4')


def game_play_ids(game_pk: int) -> pd.DataFrame:
    r = requests.get(f"{SAVANT}/gf", params={"game_pk": game_pk}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for side in ("team_home", "team_away"):
        for p in js.get(side, []):
            rows.append({
                "game_pk": game_pk,
                "at_bat_number": p.get("ab_number"),
                "pitch_number": p.get("pitch_number"),
                "pitcher": p.get("pitcher"),
                "play_id": p.get("play_id"),
            })
    return pd.DataFrame(rows)


def attach_play_ids(df: pd.DataFrame, pause: float = 0.5) -> pd.DataFrame:
    """Add a play_id column (no-op if Statcast already supplied one)."""
    if "play_id" in df and df["play_id"].notna().mean() > 0.9:
        return df
    CLIPS.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for pk in df["game_pk"].unique():
        cache = CLIPS.parent / f"gf_{pk}.parquet"
        if cache.exists():
            parts.append(pd.read_parquet(cache))
            continue
        g = game_play_ids(int(pk))
        g.to_parquet(cache)
        parts.append(g)
        time.sleep(pause)
    ids = pd.concat(parts, ignore_index=True).drop(columns=["pitcher"])
    out = df.drop(columns=["play_id"], errors="ignore").merge(
        ids, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    return out


def clip_url(play_id: str) -> str | None:
    r = requests.get(f"{SAVANT}/sporty-videos", params={"playId": play_id}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = MP4_RE.search(r.text)
    return m.group(0) if m else None


def download_clip(play_id: str, pause: float = 1.0) -> Path | None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    path = CLIPS / f"{play_id}.mp4"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = clip_url(play_id)
    if not url:
        return None
    with requests.get(url, headers=HEADERS, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    time.sleep(pause)
    return path


def sample_for_video(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Pick pitches to download, balanced across pitch types where possible."""
    per = max(1, n // df["pitch_type"].nunique())
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("pitch_type")]
    out = pd.concat(parts)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out
