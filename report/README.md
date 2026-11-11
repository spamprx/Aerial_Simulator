# AerialReport: building the report

The report uses the IITH project template (`IITHThesis.cls`). It needs **XeLaTeX** (or LuaLaTeX), **Biber**, and
**shell escape** (for `minted`).

**Fonts.** The Minion Pro fonts are not in this repository (they are commercial). Copy the `Configurations/Fonts/`
folder from the IITH project report template into `report/Configurations/Fonts/` before compiling.

## On Overleaf

1. Upload the whole `report/` folder (with the fonts added) as a new project.
2. Menu → Settings → **Compiler: XeLaTeX**, Main document: `main.tex`.
3. Recompile. Overleaf runs Biber and enables shell escape automatically.
4. Download the PDF and name it **`AerialReport.pdf`** (the course brief asks for `<GroupName>Report.pdf`).

## Before submitting

- Replace the GitHub placeholder: `\repolink` in `main.tex` (search for `TODO`).
- Check the group number in `Metadata/Metadata.tex` (`\ThesisType{GDP: IITH-AI-MAS-2026-\#0001}`).
- Review the individual contributions in `Chapters/07-Contributions.tex`. They are the last page of the report.

## Where the numbers come from

Every number in the report comes from rows in `../results/`:

| Report part | Source |
|---|---|
| Sweep (Sec. 5.1.2.2) | `results/sweep/20261110-044831/` via `scripts/analyze_sweep.py` |
| E1–E3, localisation | `results/{e1_fair,e2_current,e3_current,localisation}/` via `scripts/analyze_report.py` → `results/report_summary.json` |
| Demo mission | `docs/media/viewer_data.json`, `figures/log.txt` |
| 2D theory studies | `docs/report/lab_data/*.summary.json` |

Figures are copies of `../figures/*.png`, the lab figures (`scripts/lab/fig_*.py`) and screenshots of the 3D replay
(`docs/media/viewer*.html`).
