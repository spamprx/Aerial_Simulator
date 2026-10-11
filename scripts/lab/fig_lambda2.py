"""Figure 1: lambda_2 against convergence, from the rows of T1.9's study.  Options: --quick, --workers, --root."""
from __future__ import annotations

import exp_lambda2
from figure_tools import figure_args, study_rows
from plot_lambda2 import plot_all


def main(argv=None):
    args = figure_args(__doc__, argv)
    rows = study_rows("lambda2", exp_lambda2.main, args)
    figures = plot_all(rows, args.figures, exp_lambda2.K_P, exp_lambda2.K_V, exp_lambda2.DT)
    for figure in figures:
        print(figure)
    return figures


if __name__ == "__main__":
    main()
