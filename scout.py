#!/usr/bin/env python3
"""Build a scouting report for one pitcher.

  python scout.py "Zack Wheeler" --start 2024-03-28 --end 2025-09-28 --clips 600
"""
import argparse
from pathlib import Path

from pitchscout.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("pitcher", help='e.g. "Zack Wheeler"')
ap.add_argument("--start", default="2024-03-28")
ap.add_argument("--end", default="2025-09-28")
ap.add_argument("--clips", type=int, default=0, help="video clips to process for the vision layer (0 = stats only)")
ap.add_argument("--window", type=float, default=1.5, help="seconds before release the vision window starts")
ap.add_argument("--cutoff", type=float, default=0.10, help="seconds before release the vision window ends")
ap.add_argument("--id", type=int, help="MLBAM id, if the name lookup picks the wrong player")
ap.add_argument("--tell", action="store_true", help="run the glove tell test (needs --clips)")
ap.add_argument("--positive", default=None, help="comma list of pitch codes for the 'tell' class (default: his fastballs)")
ap.add_argument("--negative", default=None, help="comma list for the other class (default: everything else)")
ap.add_argument("--perm", type=int, default=200, help="label permutations for the p-value")
ap.add_argument("--out", type=Path, default=Path("reports"))
a = ap.parse_args()
split = lambda v: set(v.split(",")) if v else None
run(a.pitcher, a.start, a.end, a.clips, a.out, a.window, a.cutoff, pitcher_id=a.id, tell=a.tell,
    positive=split(a.positive), negative=split(a.negative), n_perm=a.perm)
