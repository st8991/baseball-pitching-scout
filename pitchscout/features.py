"""Game-situation features available *before* a pitch is thrown."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["count", "stand", "prev_pitch", "prev2_pitch", "prev_result"]
NUM_COLS = [
    "balls", "strikes", "outs_when_up", "inning", "on_1b", "on_2b", "on_3b",
    "score_diff", "times_thru", "pitch_of_ab", "pitch_of_game", "prev_speed",
]


def _result_bucket(desc: pd.Series) -> pd.Series:
    d = desc.fillna("")
    out = pd.Series("other", index=d.index)
    out[d.str.contains("ball|pitchout|hit_by_pitch", regex=True)] = "ball"
    out[d.str.contains("called_strike")] = "called_strike"
    out[d.str.contains("swinging_strike|missed_bunt", regex=True)] = "whiff"
    out[d.str.contains("foul")] = "foul"
    return out


def situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects cleaned Statcast rows for one pitcher, sorted in game order."""
    X = pd.DataFrame(index=df.index)
    X["balls"] = df["balls"].clip(0, 3)
    X["strikes"] = df["strikes"].clip(0, 2)
    X["count"] = X["balls"].astype(str) + "-" + X["strikes"].astype(str)
    X["stand"] = df["stand"]
    X["outs_when_up"] = df["outs_when_up"]
    X["inning"] = df["inning"]
    for b in ["on_1b", "on_2b", "on_3b"]:
        X[b] = df[b].notna().astype(int)
    # Score from the pitcher's team perspective.
    pitching_home = df["inning_topbot"].eq("Top")
    X["score_diff"] = np.where(pitching_home, df["home_score"] - df["away_score"],
                               df["away_score"] - df["home_score"])
    X["times_thru"] = df.get("n_thruorder_pitcher", pd.Series(np.nan, index=df.index))
    X["pitch_of_ab"] = df["pitch_number"]
    X["pitch_of_game"] = df.groupby("game_pk").cumcount() + 1

    # Previous pitches within the same at-bat (sequencing).
    ab = df.groupby(["game_pk", "at_bat_number"])
    X["prev_pitch"] = ab["pitch_type"].shift(1).fillna("none")
    X["prev2_pitch"] = ab["pitch_type"].shift(2).fillna("none")
    X["prev_result"] = _result_bucket(ab["description"].shift(1)).where(X["prev_pitch"] != "none", "none")
    X["prev_speed"] = ab["release_speed"].shift(1)

    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X[NUM_COLS + CAT_COLS]


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the most recent games so the test mimics scouting a future start."""
    games = df[["game_pk", "game_date"]].drop_duplicates().sort_values("game_date")
    n_test = max(1, int(len(games) * test_frac))
    test_games = set(games["game_pk"].iloc[-n_test:])
    is_test = df["game_pk"].isin(test_games).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]
