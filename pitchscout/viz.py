"""Skeleton overlays that show which body parts the model leaned on, frame by frame."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .fusion import N_BINS
from .pose import BONES, J, PoseTrack, read_frames

# Body-part label -> (mirrored) joint names it lights up.
PART_JOINTS = {
    "Head": ["nose"], "Glove shoulder": ["l_shoulder"], "Throwing shoulder": ["r_shoulder"],
    "Glove elbow": ["l_elbow"], "Throwing elbow": ["r_elbow"], "Glove hand": ["l_wrist"],
    "Throwing hand": ["r_wrist"], "Glove hip": ["l_hip"], "Throwing hip": ["r_hip"],
    "Front knee": ["l_knee"], "Back knee": ["r_knee"], "Front foot": ["l_ankle"],
    "Back foot": ["r_ankle"], "Hand separation": ["l_wrist", "r_wrist"],
    "Throwing elbow angle": ["r_elbow"], "Trunk lean": ["l_shoulder", "r_shoulder"],
    "Shoulder tilt": ["l_shoulder", "r_shoulder"], "Stride length": ["l_ankle", "r_ankle"],
    "Throwing arm height": ["r_wrist"],
}
# Sequential blue ramp (light -> dark), BGR for OpenCV.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RAMP_BGR = [tuple(int(h[i:i + 2], 16) for i in (5, 3, 1)) for h in RAMP]
HOT = (52, 104, 235)  # orange (#eb6834) for the single most-used joint, BGR


def _actual(joint: str, throws: str) -> int:
    """Mirrored joint name -> real MediaPipe index (lefties swap sides back)."""
    if throws == "L" and joint[:2] in ("l_", "r_"):
        joint = ("r_" if joint.startswith("l_") else "l_") + joint[2:]
    return J[joint]


def joint_weights(grid_by_part: dict[str, np.ndarray], throws: str) -> np.ndarray:
    """(N_BINS, 33) importance per real landmark per time bin, scaled to [0,1]."""
    w = np.zeros((N_BINS, 33))
    for part, vals in grid_by_part.items():
        for j in PART_JOINTS.get(part, []):
            w[:, _actual(j, throws)] += np.abs(vals)
    m = w.max()
    return w / m if m > 0 else w


def draw(frame: np.ndarray, pts: np.ndarray, weights: np.ndarray, throws: str) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    P = lambda i: (int(pts[i, 0] * w), int(pts[i, 1] * h))
    for a, b in BONES:
        cv2.line(out, P(J[a]), P(J[b]), (255, 255, 255), 2, cv2.LINE_AA)
    top = int(np.argmax(weights)) if weights.max() > 0 else -1
    for name, i in J.items():
        v = weights[i]
        r = int(4 + 14 * v)
        color = HOT if i == top else RAMP_BGR[min(len(RAMP) - 1, int(v * (len(RAMP) - 1)))]
        cv2.circle(out, P(i), r + 2, (255, 255, 255), -1, cv2.LINE_AA)  # surface ring
        cv2.circle(out, P(i), r, color, -1, cv2.LINE_AA)
    return out


def _bin_of(t: int, track: PoseTrack) -> int | None:
    if t < track.start or t >= track.end:
        return None
    return min(N_BINS - 1, int((t - track.start) / max(1, track.end - track.start) * N_BINS))


def render_overlay(clip: Path, track: PoseTrack, weights: np.ndarray, throws: str,
                   out_mp4: Path | None = None, strip_png: Path | None = None, n_strip: int = 6):
    frames, fps = read_frames(clip)
    frames = frames[: track.shot_end]
    drawn = []
    for t, f in enumerate(frames[: len(track.frames)]):
        b = _bin_of(t, track)
        wt = weights[b] if b is not None else np.zeros(33)
        g = draw(f, track.frames[t], wt, throws)
        if t >= track.end:
            cv2.putText(g, "after cutoff (not used)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
        drawn.append(g)
    if out_mp4 and drawn:
        h, w = drawn[0].shape[:2]
        vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for g in drawn:
            vw.write(g)
        vw.release()
    if strip_png and drawn:
        idx = np.linspace(track.start, max(track.start, track.end - 1), n_strip).astype(int)
        tiles = []
        for t in idx:
            g = drawn[t]
            # Crop around the pitcher.
            xs, ys = track.frames[t, :, 0], track.frames[t, :, 1]
            h, w = g.shape[:2]
            cx, cy = np.nanmean(xs) * w, np.nanmean(ys) * h
            half = int(max(np.ptp(ys) * h * 0.75, 80))
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            crop = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
            tiles.append(cv2.resize(crop, (240, 240)) if crop.size else np.zeros((240, 240, 3), np.uint8))
        sep = np.full((240, 4, 3), 255, np.uint8)
        row = [tiles[0]]
        for t in tiles[1:]:
            row += [sep, t]
        cv2.imwrite(str(strip_png), np.hstack(row))
    return drawn
