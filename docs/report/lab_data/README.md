# The rows behind the report's figures

`make figures` draws from the newest study recorded under `results/`, and falls back on these files when there is none, so
that a fresh clone can redraw every figure without re-running studies that take hours.

| File | Study | Card | Recorded | Code |
|---|---|---|---|---|
| `lambda2.csv` | `scripts/exp_lambda2.py` | T1.9 | 2026-11-05 | `df4d1cb` |
| `er_robustness.csv` | `scripts/exp_er_robustness.py` | T2.6 | 2026-11-05 | `bcf13be` |
| `degradation.csv` | `scripts/exp_degradation.py` | T2.7 | 2026-11-05 | `bcf13be` |
| `tradeoff.csv` | `scripts/exp_tradeoff.py` | T6.3 | 2026-11-09 | `e75d77e` |

Each `<study>.env.json` holds the commit, package versions and seeds, and `<study>.summary.json` the numbers the report
quotes. The first three recordings say `git_dirty: true`, because the study's own script was not committed yet when it
ran. The trade-off study ran from a clean checkout.

The first three studies were checked against the code as it stands at commit `b6ad70b`, on 2026-11-09. Their quick versions re-fly
some of the same cases. All 45 overlapping rows of the robustness study came out identical to the last digit. The 24
overlapping rows of the λ₂ study agreed to within 9 × 10⁻¹⁴ relative, which is round-off in the eigenvalue solver. The
degradation study's quick version uses a different fleet size, so it has no rows in common.

The trade-off study (T6.3) is its 14,400 runs as they came from the server, unchanged: the same SHA-256 there and here.
It ran for 6 h 18 min on 24 workers, and its last line confirms that every run of a seed had the same start, liars,
packet drops and attack noise. Drawn from this file alone, its figure is identical, pixel for pixel, to the one drawn
from the recording. The only code changed since `e75d77e` moves that figure's legend. Its results are in
`docs/report/tradeoff.md`.
