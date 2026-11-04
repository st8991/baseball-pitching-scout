"""Synthetic fixtures for tests and the notebooks' DEMO mode. Nothing here describes a real pitcher."""
import numpy as np
import pandas as pd

from pitchscout.pose import J, PoseTrack

TYPES = ["FF", "SL", "CH"]


def fake_statcast(n_games=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=5 * g)
        for ab in range(1, 26):
            balls = strikes = 0
            stand = rng.choice(["L", "R"])
            for pn in range(1, 8):
                # Count-dependent mix so the stats model has signal.
                p = np.array([0.6, 0.25, 0.15]) if balls > strikes else np.array([0.3, 0.45, 0.25])
                if stand == "L":
                    p = p[[0, 2, 1]]
                pt = rng.choice(TYPES, p=p)
                desc = rng.choice(["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"])
                rows.append(dict(game_pk=1000 + g, game_date=date, game_type="R", at_bat_number=ab,
                                 pitch_number=pn, pitch_type=pt, balls=balls, strikes=strikes, stand=stand,
                                 p_throws="R", inning=1 + ab // 4, inning_topbot="Top", outs_when_up=ab % 3,
                                 on_1b=np.nan if rng.random() < .7 else 1, on_2b=np.nan, on_3b=np.nan,
                                 home_score=0, away_score=0, n_thruorder_pitcher=1 + ab // 9,
                                 release_speed=95 - 8 * (pt != "FF") + rng.normal(), description=desc))
                if desc == "hit_into_play" or balls == 4 or strikes == 3:
                    break
                if desc == "ball":
                    balls += 1
                elif strikes < 2 or desc != "foul":
                    strikes += 1
                if balls == 4 or strikes == 3:
                    break
    return pd.DataFrame(rows)


def fake_track(pitch_type: str, fps=30.0, T=60, seed=0, throws="R") -> PoseTrack:
    """A stick figure that winds up and throws; the glove hand sits a bit lower for
    changeups (a planted 'tell' for tests)."""
    rng = np.random.default_rng(seed)
    base = np.zeros((33, 3)); base[:, 2] = 1
    pos = {"nose": (.5, .30), "l_shoulder": (.47, .38), "r_shoulder": (.53, .38), "l_elbow": (.46, .45),
           "r_elbow": (.54, .45), "l_wrist": (.49, .48), "r_wrist": (.51, .48), "l_hip": (.48, .55),
           "r_hip": (.52, .55), "l_knee": (.48, .65), "r_knee": (.52, .65), "l_ankle": (.48, .75),
           "r_ankle": (.52, .75)}
    for k, (x, y) in pos.items():
        base[J[k], :2] = x, y
    frames = np.repeat(base[None], T, 0)
    rel = 45
    t = np.arange(T)
    thr = "r_wrist" if throws == "R" else "l_wrist"
    frames[:, J[thr], 0] += np.where(t < rel, -0.002 * t, -0.09 + 0.02 * (t - rel))
    frames[:, J[thr], 1] -= 0.08 * np.exp(-((t - rel) ** 2) / 20)
    frames[:, J["l_knee"], 1] -= 0.06 * np.exp(-((t - 25) ** 2) / 30)
    if pitch_type == "CH":
        frames[:, J["l_wrist"], 1] += 0.03
    frames[..., :2] += rng.normal(0, 0.003, frames[..., :2].shape)
    return PoseTrack(frames, fps, rel, rel - 40, rel - 3, T)


def fake_crops(pitch_type: str, seed=0, n=6, size=64) -> np.ndarray:
    """Glove-shaped blob on a grass-colored background; the blob is wider on changeups."""
    import cv2
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img = np.zeros((size, size, 3), np.uint8)
        img[:] = (60, 120, 70)
        axes = (18 if pitch_type == "CH" else 13, 15)
        cv2.ellipse(img, (size // 2, size // 2), axes, 0, 0, 360, (40, 70, 130), -1)
        img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
        out.append(img)
    return np.stack(out)
