# 2026-11-19: When in the delivery does the tell appear, and is it early enough to use?

**Question.** The tell report splits the glove pose into two phases: the set (clip start to peak leg lift) and the windup (leg lift to 0.10 s before release). Notebook 05 reports glove-pose AUC 0.59 in the set and 0.56 in the windup for FF vs SL. The wrist-only check puts the remaining wrist differences in the windup. These are two point estimates with no uncertainty, and each phase is resampled to 4 bins with a different time scale per clip. For the paper, *when* the cue appears matters as much as whether it exists. A cue a full second before release is something a hitter or a runner on second could use. A cue in the last 200 ms adds little to what the hitter already sees from the ball. How should we measure the timing?

## Findings

1. **The current phase numbers cannot answer the timing question.**
   - The set-vs-windup gap (0.59 vs 0.56) has no interval. By the Nov 15 power note, a gap of 0.03 between correlated AUCs is far below what 441 clips can detect.
   - The phase boundaries are clip-relative. The set phase starts wherever the broadcast clip starts, and resampling to 4 bins stretches a 0.5 s set and a 2 s set onto the same axis. So "set bin 2" is not a fixed moment in the delivery.
   - Set length can itself depend on game state. With runners on, pitchers vary their hold and use a quicker move, and pitch mix can shift with base state. A set-phase signal could therefore be a base-state effect seen through the glove.

2. **Time-resolved decoding is the standard tool.**
   In MEG/EEG decoding, a classifier is trained and tested separately at each time point, and the result is an AUC curve over time (Grootswagers, Wardle & Carlson, 2017). For us, that means:
   - align every clip to the detected release frame, not to the clip start or leg lift;
   - cut the window from −1.5 s to −0.10 s into fixed bins of about 100 ms;
   - fit the existing game-grouped CV classifier on each bin's glove features;
   - plot AUC against time to release, with the game bootstrap for a band.

   King & Dehaene (2014) extend this to a *temporal generalization* matrix: train at time t, test at t′. A square block of high AUC means one stable cue, such as a glove held in a different spot through the set. A narrow diagonal means a transient, changing cue, which points to motion and supports the optical-flow work in #39.

3. **Use cluster-based permutation for the curve, and don't over-read the onset.**
   Fourteen bins mean fourteen looks. Maris & Oostenveld (2007) control the family-wise error over time with a cluster-based permutation test. Threshold each bin's AUC, sum the AUC excess over adjacent supra-threshold bins into clusters, and compare the largest cluster with the same statistic under label permutation. Here that means the within-game shuffle, with the same permutations across bins, as in the Nov 18 note.
   Sassenhagen & Draschkow (2019) show that a significant cluster says "there is an effect somewhere in this window". It does *not* establish where the effect starts. So for the earliest time the cue is present, report a game-bootstrap interval on the onset (for example, the first bin whose AUC passes a fixed threshold), not the cluster edge.

4. **What "early enough" means here.**
   At about 100 mph and roughly 54 ft from release to the plate, the ball takes about 0.37 s. The pipeline already stops 0.10 s before release, so everything we measure is pre-release by design. A cue from −1.5 s to about −0.5 s is clearly actionable: it could be relayed or read before the hitter commits. A cue only in the last 0.1–0.3 s before release arrives at about the time the hitter starts tracking the ball, so it is of little practical value even if it is statistically real. Release detection uses peak wrist speed and can jitter by a frame (33 ms at 30 fps). That is small next to 100 ms bins, but the alignment step should report how often release falls at the clip edge.

## Concrete suggestions for the repo

- **Release-aligned time course of the tell.** Add `tell_time_course(tracks, y, games, bin_s=0.1)`. It computes the glove-pose features per fixed time bin before release and returns the per-bin AUC with game-bootstrap bands, a cluster-based permutation p-value (within-game, shared permutations), and the earliest bin with AUC ≥ 0.55 with a bootstrap interval. Add the plot to notebook 05 and the report next to the set/windup bars. A temporal generalization matrix is an optional second figure. → #45.
- **Base-state check.** In the same issue, repeat the curve with runners-on and bases-empty pitches separately, or report the AUC of base state alone for pitch type, so a set-phase effect is not just base state.
- **Paper (#8).** Report the tell's timing relative to release and the 0.37 s flight time. "Present but only in the last few hundred milliseconds" is a different finding from "present in the set".

## Sources

- Grootswagers, T., Wardle, S. G., & Carlson, T. A. (2017). *Decoding dynamic brain patterns from evoked responses: A tutorial on multivariate pattern analysis applied to time series neuroimaging data.* Journal of Cognitive Neuroscience 29(4), 677–697. https://arxiv.org/abs/1606.02840
- King, J.-R., & Dehaene, S. (2014). *Characterizing the dynamics of mental representations: the temporal generalization method.* Trends in Cognitive Sciences 18(4), 203–210. https://pmc.ncbi.nlm.nih.gov/articles/PMC5635958
- Maris, E., & Oostenveld, R. (2007). *Nonparametric statistical testing of EEG- and MEG-data.* Journal of Neuroscience Methods 164(1), 177–190. (Cited for the cluster-based permutation test; I did not reread it this run.)
- Sassenhagen, J., & Draschkow, D. (2019). *Cluster-based permutation tests of MEG/EEG data do not establish significance of effect latency or location.* Psychophysiology 56(6), e13335. (I did not reread it this run.)
- Flight time: own arithmetic, 54 ft at 100 mph (146.7 ft/s) ≈ 0.37 s.
- Code and numbers: `pitchscout/glove.py` (`phases`, `glove_pose_features`, `PHASE_BINS = 4`, `run_tell_test` `phase_auc`), `pitchscout/pose.py` (`find_release`, `extract` with `window_s=1.5`, `cutoff_s=0.10`). Notebook 05 results: set 0.59, windup 0.56 (Nov 5 run). Earlier notes: 2026-11-15 (power), 2026-11-18 (multiple looks).
