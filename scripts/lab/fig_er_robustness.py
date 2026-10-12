"""Figure 2: Erdos-Renyi robustness against success, from the rows of T2.6's study.  Options: --quick, --workers, --root."""
from __future__ import annotations

import exp_er_robustness
from figure_tools import figure_args, study_rows
from plot_robustness import plot_success


def main(argv=None):
    args = figure_args(__doc__, argv)
    figure = plot_success(study_rows("er_robustness", exp_er_robustness.main, args), args.figures)
    print(figure)
    return figure


if __name__ == "__main__":
    main()
