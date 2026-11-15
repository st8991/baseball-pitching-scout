# 2026-11-15: How many clips do we need, and which claims can one pitcher support?

**Question.** The real-data run used 441 FF/SL clips (474 with changeups) out of 600 sampled. Several open issues add new comparisons: glove vs body (#25), glove beyond body (#26), held-out venue/season (#30), and stats vs stats + vision (#23). Before more runs: how precise are our AUCs at this sample size, how many clips exist, and which of these comparisons can a single reliever's data actually decide?

## How much data exists

Baseball-Reference lists Miller at 65.0 IP, 55 games and 249 batters faced in 2024, and 61.2 IP, 60 games and 234 BF in 2025 (Athletics 151 BF, Padres 83). At a typical 3.5–4 pitches per plate appearance that is roughly **1,700–1,900 pitches over 115 games**, almost all FF or SL. So the pipeline uses about a quarter of the available clips, and a game has about 15 pitches on average.

## Findings

1. **Against chance, 441 clips is already enough. More clips mainly narrow the interval.**
   With the Hanley & McNeil (1982) standard error for an AUC (balanced classes, their exponential approximation for Q1/Q2):

   | FF/SL clips | SE at AUC 0.58 | 95% CI half-width | Power vs 0.5, AUC 0.576 (wrist-only) | Power vs 0.5, AUC 0.598 |
   |---|---|---|---|---|
   | 441 (now) | 0.027 | ±0.053 | 0.79 | 0.95 |
   | 1,000 | 0.018 | ±0.035 | 0.99 | ≈1 |
   | 1,900 (all) | 0.013 | ±0.026 | ≈1 | ≈1 |

   These treat pitches as independent. With about 15 pitches per game and within-game correlation, the effective sample is smaller (the Kish design effect, 1 + (m − 1)·ICC). The game bootstrap from #10 already accounts for this empirically, so read the table as a best case.

2. **The glove-vs-body gap cannot be decided with one pitcher.**
   For two correlated AUCs on the same pitches, the variance of the difference is about 2·SE²·(1 − r), where r is the correlation between the two AUC estimates (Hanley & McNeil 1983 give the paired form; Obuchowski & McClish 1997 treat sample size for this design). For the observed wrist-only gap of 0.012 (0.576 vs 0.564):

   | Correlation r between glove and body AUCs | Power at 441 clips | Power at 1,900 clips | Clips for 80% power |
   |---|---|---|---|
   | 0.3 | 0.06 | 0.12 | ≈ 25,000 |
   | 0.6 | 0.07 | 0.17 | ≈ 14,000 |

   That is 7–13 times all the pitches Miller threw in two seasons, before any design effect. A gap this small can neither be confirmed nor ruled out with his data. #25 will almost surely report an interval that includes zero, and the paper should read that as "not distinguishable", not as "no difference". The full-finger glove gap (0.598 vs 0.564 = 0.034) is about three times larger and needs roughly a ninth as many clips, which is still more than Miller has.

3. **What this means for the open comparisons.**
   - *Glove vs chance (#4):* answerable now. Using all clips would bring the interval to about ±0.03.
   - *Glove vs body (#25) and glove beyond body (#26):* only large effects are detectable. Report intervals and a minimum effect of interest instead of a yes/no.
   - *Held-out venue/season (#30):* each held-out domain has a few hundred clips at most, so per-domain AUC intervals will be about ±0.07–0.10. That is good for spotting a collapse to 0.5, not for fine comparisons.
   - *Stats vs stats + vision (#23):* the same paired-difference logic applies to Δlog-loss. Expect a wide interval.

4. **Two ways to get more power.**
   - *Use every clip.* Scaling the pipeline from 600 to about 1,900 Miller clips is the cheapest gain. About 20 s per clip on one core (README) is roughly 10 CPU-hours.
   - *Pool pitchers for the comparative question.* "Does the glove add beyond the body?" is a general claim. Fitting the same per-pitcher test on several relievers and combining the per-pitcher differences (a random-effects meta-analysis, or a pooled model with pitcher as a group) multiplies the sample without pretending one pitcher has 14,000 pitches. Miller stays the case study, and the pooled result answers the methods question.

## Concrete suggestions for the repo

- **Run on all available clips.** Make `n_clips` default to "all" for the real-data notebook run (keep the current sample for demo/CI), add a resumable batch mode so the ~10 CPU-hour pose extraction can run in pieces, and rerun notebook 05. → #33.
- **State a minimum effect of interest and report precision.** In the tell report, print each AUC interval half-width and, for glove − body, the smallest gap the current n could detect at 80% power (Hanley–McNeil formula with the observed correlation). → #34.
- **Paper (#7/#8).** Present the glove-vs-chance result as the confirmatory claim. Present glove-vs-body and glove-beyond-body as estimates with intervals, note the power limits, and mention a multi-pitcher extension as future work.

## Sources

- Hanley, J. A., & McNeil, B. J. (1982). *The meaning and use of the area under a receiver operating characteristic (ROC) curve.* Radiology 143(1), 29–36. https://math.ucdavis.edu/~saito/data/roc/hanley-mcneil-auc.pdf
- Hanley, J. A., & McNeil, B. J. (1983). *A method of comparing the areas under receiver operating characteristic curves derived from the same cases.* Radiology 148(3), 839–843. (Cited for the paired-AUC variance; I did not reread it this run.)
- Obuchowski, N. A., & McClish, D. K. (1997). *Sample size determination for diagnostic accuracy studies involving binormal ROC curve indices.* Statistics in Medicine 16(13), 1529–1542. (Cited for paired-design sample size; a 2025 methods comparison of the two approaches is at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12924612/, which I did not read in full.)
- Mason Miller season totals: Baseball-Reference. https://www.baseball-reference.com/players/m/millema03.shtml
- Calculations: Hanley–McNeil SE with Q1 = A/(2 − A), Q2 = 2A²/(1 + A), balanced classes, two-sided α = 0.05. Script in the PR description.
