#!/bin/env python
"""
Plot the convergence of an MLKAPS surrogate models over the number of samples.
The data to plot should have been generated via the `run_mlkaps.py` program
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def plot_errors(errors: pd.DataFrame, title: str | None, extension: str):
    """
    Plot the data as histograms with UVSQ graphical style guide
    """

    color_seq = ["#0092BB", "#77AD1C", "rgb(240, 182, 0)"]
    names = ["Mean Absolute Error", "Mean Absolute Percent Error (%)", "R² Score"]

    for metric, color, name in zip(["mae", "mape", "r2"], color_seq, names):

        fig = go.Figure(
            data=[
                go.Bar(
                    x=errors["nb_samples"],
                    y=errors[metric],
                    marker_color=color,
                )
            ]
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Number of samples",
            yaxis_title=name,
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
            width=900,
            height=400,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"t": 40, "r": 0, "l": 40, "b": 20},
            title=title,
            font_family="Gotham-Light",
            title_font_family="Gotham",
            legend_font_family="Gotham",
        )
        if title is None:
            fig.update_layout(
                margin={"t": 0, "r": 0, "l": 40, "b": 20},
            )

        fig.update_xaxes(title_font_family="Gotham")
        fig.update_yaxes(gridcolor="rgba(111, 111, 110, 0.5)", title_font_family="Gotham")
        fig.show()
        fig.write_image(f"error_evolution_{metric}.{extension}")


def main():

    parser = ArgumentParser(
        prog="plot_convergence.py",
        description="Utility used to plot the convergence of a model regarding the evolution of MEA, MAPE and R2",
    )
    parser.add_argument(
        "data_file",
        help="CSV file containing the data to plot",
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Title to put on the plots (default: None)",
    )
    parser.add_argument(
        "--extension",
        choices=["jpeg", "jpg", "png", "svg"],
        help="Image format symbolized by the extension used (default: png)",
        type=str,
        default="png",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.data_file)

    data["mape"] *= 100

    plot_errors(data, args.title, args.extension)


if __name__ == "__main__":
    main()
