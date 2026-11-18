# 2026-11-18: The tell now has many looks. Which p-values survive?

**Question.** The Nov 4 note said to name one primary test and apply Holm to the rest. Since then the number of looks has grown: glove pose, glove pixels, wrist-only, two seasons, three pitch pairs, and soon DINOv2 (#38), optical flow (#39) and per-venue folds (#30). `glove.verdict` also picks the *better* of glove pose and glove pixels and reports that set's own p-value. How much does this inflate false positives, and what is the cheapest correct fix with our permutation machinery?

## Findings

1. **Picking the best of several looks makes its p-value optimistic.**
   If k looks are independent and all null, the chance that at least one reaches p < 0.05 is 1 − 0.95^k: 0.10 for k = 2, 0.14 for 3, 0.23 for 5 and 0.34 for 8. Correlated looks inflate less, but glove pose and HOG pixels share almost nothing, so `verdict`'s "best of pose and pixels" is close to the k = 2 case. Gelman & Loken (2014) point out that the problem exists even without deliberate fishing. The wrist-only check, the season split and the choice of pitch pair were all made after seeing data, and each choice is one more path.

2. **Holm on today's numbers: the glove-pose headlines survive, the rest do not.**
   Holm's (1979) step-down method is valid under any dependence. Applied to the eight glove-vs-chance looks reported so far (README and notebook 05, Nov 5 and Nov 8 runs; FF vs CH counted as n.s.):

   | Look | p | Holm threshold | Survives |
   |---|---|---|---|
   | FF vs SL, glove pose | 0.004 | 0.0063 | yes |
   | FF vs rest, glove pose | 0.005 | 0.0071 | yes |
   | FF vs SL, wrist-only | 0.024 | 0.0083 | no (and stop) |
   | 2025 only | 0.04 | — | no |
   | FF vs SL+CH, wrist-only | 0.056 | — | no |
   | 2024 only, pixels, FF vs CH | ≥ 0.38 | — | no |

   Two caveats. These p-values are from the free shuffle. The within-game p-values from #18 are usually larger, so the table is a best case. More importantly, **200 permutations cannot resolve Holm thresholds.** The smallest possible permutation p is 1/(n_perm + 1) ≈ 0.005 (Phipson & Smyth, 2010). That is already at the first threshold, so one null draw more or less decides "survives". With 8 to 12 looks the thresholds fall to 0.004–0.006. Resolving them needs about 1,000–2,000 permutations for the looks that matter.

3. **A max-statistic permutation test corrects for all looks at once, nearly for free.**
   Westfall & Young (1993) and, for permutation tests in practice, Nichols & Holmes (2002) use the same label permutation for every test. For each permutation, record the *maximum* statistic across tests. Then compare each observed statistic with that null distribution of maxima. This controls the family-wise error exactly, uses the correlation between tests instead of assuming the worst case like Bonferroni, and needs no new model fits.
   Our code is already almost there. `test_feature_set` seeds its RNG with the same `seed`, so glove pose and the body control, which share rows and games, already get *identical* within-game permutations. Today each set compares its AUC with its own null array. Taking the element-wise max of those arrays across the glove sets gives the maxT null. The one gap is the pixel set, which drops rows without crops. For it, permute on the full row set and then subset, or restrict the family to the rows every set has. The subset version shifts class counts slightly per draw, so the paper should say which one was used.

4. **What belongs in the family.** The confirmatory family is "does any glove representation beat chance on the primary pitch pair": glove pose, wrist-only, pixels, and later DINOv2 and flow. maxT over that family answers "the glove carries signal" honestly, whatever representation wins. Season, venue and phase splits are diagnostics for *where* the signal lives (#29, #30). Report them with intervals, not as extra wins. The body control is a comparator, not a member of the family.

## Concrete suggestions for the repo

- **Family-wise p-value in the tell report.** In `run_tell_test`, keep each glove set's permutation null and compute a max-AUC null across the glove family on shared permutations. Report `perm_p_fwer` next to each set's `perm_p`, and make `verdict` use the family-wise p for whichever glove set it picks. Raise the default `n_perm` for the notebook run to 1,000 (CI can stay at a small number). → #42.
- **Paper (#7/#8).** Pre-state the glove family and the primary pair in methods. Put the Holm or maxT-adjusted p-values in the results table, and label the season, venue and phase splits as exploratory.

## Sources

- Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics 6(2), 65–70.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment.* Wiley. (Cited for the maxT step-down method; I did not reread it this run.)
- Nichols, T. E., & Holmes, A. P. (2002). *Nonparametric permutation tests for functional neuroimaging: A primer with examples.* Human Brain Mapping 15(1), 1–25. https://pubmed.ncbi.nlm.nih.gov/11747097/ (PDF: https://www.fil.ion.ucl.ac.uk/spm/doc/papers/NicholsHolmes.pdf)
- Phipson, B., & Smyth, G. K. (2010). *Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn.* Statistical Applications in Genetics and Molecular Biology 9(1), Article 39.
- Gelman, A., & Loken, E. (2014). *The statistical crisis in science.* American Scientist 102(6), 460–465. (The "garden of forking paths" argument.)
- Numbers: README "Result: Mason Miller glove tell" and notebook 05 (Nov 5 run, wrist-only Nov 8). Code: `pitchscout/glove.py` (`test_feature_set`, `_perm_null`, `run_tell_test`, `verdict`). Earlier note: 2026-11-04-validating-small-sample-tell.md.
