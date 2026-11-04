"""Pitcher pose extraction from broadcast clips, limited to the pre-release window.

Pipeline per clip:
  1. Keep only the first camera shot (the center-field view of the delivery).
  2. Run MediaPipe Pose (several people per frame) and track the pitcher.
  3. Find release as the peak speed of the throwing wrist.
  4. Keep frames from `window_s` before release up to `cutoff_s` before release,
     so the model only sees what a hitter or dugout could see *before* the ball is out.
  5. Normalize (hip-centered, torso-scaled, lefties mirrored) and resample to fixed bins.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path("cache/pose_landmarker_full.task")

# MediaPipe landmark indices we use.
J = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25,
    "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}
JOINTS = list(J)
# Bones for drawing.
BONES = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
         ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("nose", "l_shoulder"), ("nose", "r_shoulder")]


@dataclass
class PoseTrack:
    frames: np.ndarray       # (T, 33, 3) x, y in [0,1] image coords, visibility
    fps: float
    release: int             # frame index of release
    start: int               # first frame of the model window
    end: int                 # last frame of the model window (exclusive)
    shot_end: int            # last frame of the first camera shot


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def read_frames(path: Path, max_s: float = 8.0) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < int(max_s * fps):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, float(fps)


def first_shot_end(frames: list[np.ndarray], thresh: float = 0.55) -> int:
    """Index where the first hard camera cut happens (or len(frames))."""
    prev = None
    for i, f in enumerate(frames):
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) < thresh:
            return i
        prev = h
    return len(frames)


def detect_people(frames: list[np.ndarray], fps: float, num_poses: int = 4) -> list[list[np.ndarray]]:
    from mediapipe.tasks.python import BaseOptions, vision
    import mediapipe as mp

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO, num_poses=num_poses,
        min_pose_detection_confidence=0.4, min_tracking_confidence=0.4,
    )
    out = []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(i * 1000 / fps))
            out.append([np.array([[p.x, p.y, p.visibility] for p in pose]) for pose in res.pose_landmarks])
    return out


def track_pitcher(people: list[list[np.ndarray]]) -> np.ndarray:
    """Pick the pitcher in each frame. In the center-field view the pitcher is the
    largest figure near the bottom-center at the start; after that, follow the
    nearest hip center frame to frame."""
    T = len(people)
    track = np.full((T, 33, 3), np.nan)
    last = None
    for t, poses in enumerate(people):
        if not poses:
            continue
        if last is None:
            def score(p):
                h = np.nanmax(p[:, 1]) - np.nanmin(p[:, 1])
                return h - 0.5 * abs(np.nanmean(p[:, 0]) - 0.5)
            best = max(poses, key=score)
        else:
            best = min(poses, key=lambda p: np.linalg.norm(p[[23, 24], :2].mean(0) - last))
            if np.linalg.norm(best[[23, 24], :2].mean(0) - last) > 0.15:
                continue  # lost track this frame
        track[t] = best
        last = best[[23, 24], :2].mean(0)
    return track


def interp_nans(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    flat = a.reshape(len(a), -1)
    idx = np.arange(len(a))
    for k in range(flat.shape[1]):
        col = flat[:, k]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            flat[:, k] = np.interp(idx, idx[ok], col[ok])
    return flat.reshape(a.shape)


def find_release(track: np.ndarray, fps: float, throws: str) -> int:
    w = J["r_wrist"] if throws == "R" else J["l_wrist"]
    xy = track[:, w, :2]
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    sp = np.nan_to_num(sp)
    k = max(1, int(round(fps / 30)))
    sp = np.convolve(sp, np.ones(k) / k, mode="same")
    return int(np.argmax(sp)) + 1


def extract(path: Path, throws: str, window_s: float = 1.5, cutoff_s: float = 0.10) -> PoseTrack | None:
    frames, fps = read_frames(path)
    if not frames:
        return None
    shot = first_shot_end(frames)
    frames = frames[:shot]
    track = track_pitcher(detect_people(frames, fps))
    if np.isnan(track[:, 0, 0]).mean() > 0.5:
        return None
    track = interp_nans(track)
    rel = find_release(track, fps, throws)
    start = max(0, rel - int(window_s * fps))
    end = max(start + 2, rel - int(round(cutoff_s * fps)))
    return PoseTrack(track, fps, rel, start, end, shot)


def normalize_all(track: np.ndarray, throws: str) -> np.ndarray:
    """Like normalize() but returns all 33 landmarks, (T, 33, 2)."""
    xy = track[:, :, :2].copy()
    hip = xy[:, [23, 24]].mean(1, keepdims=True)
    sh = xy[:, [11, 12]].mean(1, keepdims=True)
    torso = np.nanmedian(np.linalg.norm((sh - hip)[:, 0], axis=1)) or 1.0
    xy = (xy - hip) / torso
    if throws == "L":
        xy[..., 0] *= -1
        swap = list(range(33))
        for a, b in [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                     (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
            swap[a], swap[b] = b, a
        xy = xy[:, swap]
    return xy


def normalize(track: np.ndarray, throws: str) -> np.ndarray:
    """Hip-centered, torso-scaled coordinates for our joints; lefties mirrored so
    'r_' always means throwing side. Returns (T, len(JOINTS), 2)."""
    return normalize_all(track, throws)[:, [J[j] for j in JOINTS]]


def resample(seq: np.ndarray, n_bins: int) -> np.ndarray:
    """Linear resample along time to n_bins."""
    T = len(seq)
    src = np.linspace(0, T - 1, n_bins)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return seq[lo] * (1 - w) + seq[hi] * w


def save(track: PoseTrack, path: Path) -> None:
    np.savez_compressed(path, frames=track.frames, fps=track.fps, release=track.release,
                        start=track.start, end=track.end, shot_end=track.shot_end)


def load(path: Path) -> PoseTrack:
    z = np.load(path)
    return PoseTrack(z["frames"], float(z["fps"]), int(z["release"]), int(z["start"]),
                     int(z["end"]), int(z["shot_end"]))
