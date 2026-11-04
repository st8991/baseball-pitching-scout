# Contributing

Two people work on this repo: Sam and Claude. Everything goes through a pull request so `main` always has passing tests.

## Workflow

1. Pick or open an issue on the project board.
2. Branch off the latest `main`: `git switch main && git pull && git switch -c <short-topic>`.
   Claude's branches start with `claude/`.
3. Commit in small steps with messages that say what changed.
4. Open a PR into `main` and fill in the template. Link the issue with `Closes #N`.
5. Wait for the **tests** check to go green, then squash-merge. The branch is deleted on merge.

`main` is protected by a ruleset: no direct pushes, no force-pushes, no deletion, and the `tests` check must pass. No approval is required, because PRs can come from the same account that merges them. Sam still reads every Claude PR before merging.

## Checks to run before pushing

```bash
pip install -r requirements.txt
python -m pytest -q
PITCHSCOUT_DEMO=1 jupyter nbconvert --to notebook --execute notebooks/<changed>.ipynb --output-dir /tmp
```

CI runs the same unit tests plus every notebook in demo mode (`.github/workflows/ci.yml`).

## Data and results

- Demo mode (`PITCHSCOUT_DEMO=1`) is synthetic. Never quote its numbers as findings.
- `cache/` and `reports/` stay out of git. Put result tables you want to keep for the paper in the PR description or under `paper/` in a dedicated PR.
- Clear notebook outputs before committing unless the output is the point of the change.
