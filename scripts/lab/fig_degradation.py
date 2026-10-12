"""Figure 3: degradation against F, with the honest information lost, from the rows of T2.7's study.
Options: --quick, --workers, --root."""
from __future__ import annotations

import exp_degradation
from figure_tools import figure_args, study_rows
from plot_robustness import plot_degradation


def main(argv=None):
    args = figure_args(__doc__, argv)
    figure = plot_degradation(study_rows("degradation", exp_degradation.main, args), args.figures)
    print(figure)
    return figure


if __name__ == "__main__":
    main()
