# 2026-11-04: How to make the glove-tell result defensible with few pitches

**Question.** The glove test in `pitchscout/glove.py` (`test_feature_set`) reports a game-grouped CV AUC, a label-permutation p-value, and k=2 GMM agreement (ARI/NMI/purity). With one reliever's pitches (likely a few hundred clips at most), what should the test do so that issue #4 ("is the reported tell real?") and the supervised vs unsupervised comparison in #5 hold up in a paper?

## Findings

1. **Permutation tests are the right tool, but the permutation has to match the dependence structure.**
   Ojala & Garriga (2010) define the label-permutation test for classifier accuracy that the repo already uses (permute `y`, rerun the whole CV, count how often the null beats the real score). The repo's `(1 + #null ≥ obs) / (1 + n_perm)` form is the standard unbiased estimate.
   The catch: shuffling labels freely across all pitches assumes every pitch is exchangeable. Ours are not. Pitch mix, broadcast camera, lighting and park differ by game, and folds are grouped by game. A free shuffle breaks the link between game and label, so game-level artifacts that happen to line up with pitch mix can look like a "tell". Winkler et al. (2014) formalize this with *exchangeability blocks*: when observations are only exchangeable within groups, permute within those groups. Here the block is `game_pk`.

2. **Small samples give wide error bars on CV scores, so report an interval and not only a point estimate.**
   Varoquaux (2017/2018) shows that CV accuracy estimates are very noisy at the sample sizes we have, roughly "±10% for 100 samples". An AUC of 0.60 on ~200 pitches can be entirely consistent with chance. The paper should report a confidence interval on the AUC (for example a game-level bootstrap of the out-of-fold predictions), alongside the permutation p-value.

3. **The unsupervised side has no null right now.**
   `test_feature_set` reports ARI/NMI/purity for the GMM but no p-value, so the supervised-vs-unsupervised comparison in #5 is asymmetric. Purity is never below the majority rate and ARI has a nonzero spread under chance, so these need a reference too. The same label-permutation idea applies directly. Fit the clustering once (it never sees labels), then permute `yb` (within game) and recompute ARI/NMI/purity. That is cheap because the GMM doesn't need refitting.

4. **Multiple looks inflate false positives.** The report runs several feature sets (glove pose, glove pixels, body control), two phases, and optionally several pitch pairs. If the paper's headline is "the glove tips pitches", the p-values across those looks need a correction (Holm is simple and valid under dependence), or one test needs to be named in advance as primary.

5. **Related work: what to expect from body-only pitch prediction.** Bright, Lu & Zelek (arXiv, March 2026) predict pitch type from broadcast video before release using 3D body kinematics only, on 119,561 pitches. They report 80.4% accuracy over eight types, with upper-body mechanics carrying 64.9% of the signal and wrist position the most informative joint group (14.8%). They also find that grip-defined variants (four-seam vs two-seam) are not separable from pose. This is the closest prior work for issue #6. It also suggests our "body without glove" control is a strong baseline, so a glove effect has to beat a real body signal and not just chance. (The page I read does not mention glove-side cues.)

## Concrete suggestions for the repo

- **Within-game permutation** in `test_feature_set`: replace `rng.permutation(yb)` with a shuffle inside each `game_pk` (a few lines with numpy). Keep the old free shuffle as an option and report both. If the tell survives only the free shuffle, it is probably a game-level artifact. → issue opened.
- **AUC confidence interval**: bootstrap games (not pitches) over the out-of-fold `p` already computed in `test_feature_set`, and report the 2.5–97.5% interval. → same issue.
- **Null for ARI/NMI/purity**: add a permutation p-value for the unsupervised metrics so #5 compares like with like. → issue opened.
- **Fold balance (minor)**: `_folds` assigns games to folds with `rng.integers`, so fold sizes can be very uneven and a fold can be empty. `sklearn.model_selection.GroupKFold` (or `StratifiedGroupKFold`) gives balanced folds. Not opened as an issue; fold it into the first issue if convenient.
- **Paper (#7/#8)**: state the primary test up front (glove pose, fastball vs offspeed, within-game permutation) and apply Holm to the secondary looks.

## Sources

- Ojala, M., & Garriga, G. C. (2010). *Permutation Tests for Studying Classifier Performance.* JMLR 11, 1833–1863. https://www.jmlr.org/papers/v11/ojala10a.html
- Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). *Permutation inference for the general linear model.* NeuroImage 92, 381–397. https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/ (I could not load the full text this run; the exchangeability-block idea is the paper's central contribution.)
- Varoquaux, G. (2018). *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage. Preprint: https://arxiv.org/abs/1706.07581
- Bright, J., Lu, M., & Zelek, J. (2026). *Interpretable Pre-Release Baseball Pitch Type Anticipation from Broadcast 3D Kinematics.* arXiv:2603.04874. https://arxiv.org/abs/2603.04874

Not used: a SPIE paper, "Skeleton-based baseball pitch classification on broadcast videos" (Proc. SPIE 13510), came up in search, but the page was behind a bot check, so I have not verified its contents. Worth a look for #6.
