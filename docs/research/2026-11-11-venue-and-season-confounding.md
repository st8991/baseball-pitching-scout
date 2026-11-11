# 2026-11-11: Could the glove tell be a camera or ballpark artifact?

**Question.** Notebook 05 reports that the glove signal is stronger in 2025 (AUC 0.59, p 0.04) than in 2024 (0.51, p 0.38). Our clips span three home parks and two broadcast crews: the A's at the Oakland Coliseum in 2024, the A's at Sutter Health Park in Sacramento in 2025, and the Padres at Petco Park after Miller was traded at the 2025 deadline (to San Diego with JP Sears, for Leo De Vries and three others). Is the season difference a sign of a camera, venue or broadcast artifact, and how should we check?

## Findings

1. **Vision models readily pick up "where the image came from", and that can masquerade as signal.**
   Zech et al. (2018) trained CNNs to detect pneumonia on chest X-rays pooled from two hospitals. A CNN could identify which hospital a radiograph came from with 99.95–99.98% accuracy. Because pneumonia prevalence differed between hospitals (34.2% vs 1.2%), sorting by hospital *alone* reached AUC 0.861. The jointly trained model scored 0.931 internally but 0.815 on a third, unseen hospital. Geirhos et al. (2020) call this *shortcut learning*: the model solves the benchmark through a correlated cue that does not transfer. Our version is "pitch mix differs by venue or period, and pose features carry a venue fingerprint". Center-field camera height, lens and angle differ by park, and the README already lists camera angle as a known limit.

2. **Our setup has the ingredients for exactly that.**
   - *A venue fingerprint in the features.* Normalization (hip-centered, torso-scaled) removes scale and position but not perspective. Glove height relative to the chest, elbow flare and "glove motion" all change with camera elevation.
   - *Pitch mix that can differ by venue or period.* A closer's FF/SL split can shift between seasons, with game state, and between teams after a trade. The clip sample is balanced by pitch type overall, not within venue or period.
   - *Folds grouped by game, not by venue.* Game-grouped CV stops a single game leaking across folds, but every venue appears in both train and test. So a venue shortcut is rewarded, not penalized, exactly as in Zech's pooled internal test.

   This does not mean the 2025 signal *is* an artifact. Miller could have changed his glove set. It means the current validation cannot tell the two apart.

3. **The standard checks are cheap.**
   - *Can venue/period alone predict pitch type?* A "domain-only" baseline (home park × broadcast side × season, plus the stats context) gives the AUC a shortcut could reach. Zech's hospital-only AUC is the template.
   - *Can the glove features identify the venue?* If a classifier predicts the venue or period from glove features well above chance, the fingerprint exists and finding 1 applies.
   - *Leave-one-domain-out validation.* Train on two domains and test on the third (for example, train on Oakland and Sacramento, test on Padres games), and also train on 2024 and test on 2025. A real glove tell should survive in the held-out domain, possibly weaker. A shortcut collapses to about 0.5. This is the external-validation idea from Zech et al. applied to parks instead of hospitals.
   - *Within-domain permutation.* The within-game permutation from #10 already holds game-level artifacts fixed. A within-venue shuffle is the analogous check one level up, and costs nothing extra.

4. **The data we need is one step away.** Statcast's pitch-level export includes `home_team` and `away_team`, but `db.py`'s `pitches` table does not keep them. `inning_topbot` plus Miller's team on that date (A's until 2025-07-31, Padres after) gives home or away and the park. The broadcast feed (home or away crew) may differ per clip and should be recorded from the clip metadata if Savant exposes it. I did not verify that this run.

## Concrete suggestions for the repo

- **Domain columns and shortcut baselines.** Keep `home_team`/`away_team` in `pitches`, derive `venue`, `season` and (if available) broadcast side per clip, and add two checks to the tell report: a domain-only baseline AUC for pitch type, and the AUC of glove features for predicting venue/period. → #29.
- **Leave-one-domain-out tell test.** Add an option to `run_tell_test` that uses venue or season as the CV group instead of `game_pk` (train on the rest, test on the held-out domain), and report AUC per held-out domain alongside the pooled result. → #30.
- **Paper (#7/#8).** Report the season and venue split explicitly. If the tell holds only in pooled CV and not across domains, say it is not shown to generalize beyond the venues it was trained on.

## Sources

- Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study.* PLOS Medicine 15(11), e1002683. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2, 665–673. https://bethgelab.org/publication/2020_11_geirhos/ (cited for the concept; I did not reread the full text this run.)
- Mason Miller trade, 2025 deadline: NESN, "Padres trade for Mason Miller in Athletics blockbuster". https://nesn.com/2025/07/padres-trade-mason-miller-athletics-blockbuster (the article does not give the exact date; other coverage from the same search dates it July 31, 2025.)
- Season split: `notebooks/05_mason_miller_glove_tell.ipynb`, results section (Nov 5, 2026 run). Schema: `pitchscout/db.py` (`pitches` table).
