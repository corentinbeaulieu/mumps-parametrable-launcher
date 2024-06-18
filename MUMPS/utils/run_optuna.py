#!/bin/env python3

from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from mlkaps.modeling import OptunaTunerLightgbm
from mlkaps.modeling.optuna_model_tuner import OptunaRecorder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


class OptunaTunerWithRecorder:

    ran_init = False

    def __init__(
        self,
        record_path="optuna_records.csv",
        plot_output_dir=Path("convergence_plots_optuna/"),
    ):
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)

        if not OptunaTunerWithRecorder.ran_init:
            # This is the first time we use the OptunaTuner in this session
            # Delete previous records if they exist
            record_path.unlink(missing_ok=True)
            OptunaTunerWithRecorder.ran_init = True

        self.record_path = record_path
        self.plot_output_dir = plot_output_dir

    def _plot_optuna_convergence(self, records: pd.DataFrame, output_path: str):

        reference = records.iloc[0].to_dict()
        output_path = self.plot_output_dir / Path(output_path.format(**reference))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sliding window with the min score found by optuna
        sliding_min_score = records["score"].expanding().min()

        fig, ax = plt.subplots(figsize=(10, 3), layout="constrained", dpi=100)

        sns.lineplot(y=sliding_min_score, x=records["iteration"], ax=ax, color="purple")

        ax.grid(axis="y", linestyle="--", alpha=0.8)
        ax.set_xticks(np.linspace(0, len(records) - 1, 8, dtype=int))
        ax.set_xlabel("Number of trials", fontsize=10, fontweight="bold")

        ax.set_ylabel(
            "MAE\n (logscale)",
            fontsize=10,
            fontweight="bold",
            rotation=-90,
            labelpad=40,
        )
        ax.set_yscale("log")
        ax.set_yticks(np.geomspace(sliding_min_score.min(), sliding_min_score.max(), 5))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()

        fig.suptitle("Optuna score convergence", fontsize=12, fontweight="bold")
        fig.savefig(output_path)
        plt.close()

    def __call__(
        self,
        samples: pd.DataFrame,
        objective_name="performance",
        time_budget=120,
        n_trials=100,
    ):

        # Prepare the sample
        X = samples.drop([objective_name], axis=1)
        y = samples[objective_name]

        # ============= MLKAPS interface
        # Build the optuna tuner, use 5 folds to reduce cost
        tuner = OptunaTunerLightgbm(X, y, n_folds=5)
        tuner = OptunaRecorder(tuner, self.record_path)
        model, params = tuner.run(time_budget, n_trials)

        # ============= Plotting
        # Fetch and parse the tuning results
        records = pd.read_csv(self.record_path)
        curr_id = records["training_id"].max()
        records = records[records["training_id"] == curr_id]
        scores = records["score"]

        self._plot_optuna_convergence(
            records,
            "optuna_convergence_for_id_{training_id}_nsamples_{number_of_samples}.png",
        )

        res = namedtuple("OptunaResult", ["model", "params", "best_score", "scores", "records"])
        res = res(model, params, scores.min(), scores, records)
        return res


def main():

    parser = ArgumentParser(
        description="Run optuna optimisation on growing subsets of the samples to output the error and understand the "
        "convergence of the model made by MLKAPS",
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to the MLKAPS experiment directory containing the samples and trained model",
        type=Path,
    )
    parser.add_argument(
        "validation_samples",
        help="Path to the csv file containing the validation samples (i.e. random samples...)",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        help="Path to the output file (default: stdout)",
        type=Path,
    )
    parser.add_argument(
        "--no-header",
        dest="no_header",
        action="store_true",
        help="Don't print the csv header",
    )
    parser.add_argument(
        "--start",
        dest="start",
        help="Size of the first chunk of samples to train on (default: 100)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--step",
        dest="step",
        help="Size of a step between two trainings (default: 100)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-t",
        "--time-budget",
        dest="time_budget",
        help="Time budget for optuna optimization in seconds (default: 120)",
        type=int,
        default=120,
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        dest="n_trials",
        help="Maximum number of optuna trials (default: 200)",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    dirpath = Path(args.experiment_dir)
    validation_path = Path(args.validation_samples)
    all_samples = pd.read_csv(dirpath / "kernel_sampling" / "samples.csv").astype({"ordering": "category"})

    validation_samples = pd.read_csv(validation_path).astype({"ordering": "category"})

    if args.no_header == False:
        if args.output_file is None:
            print("nb_samples,best_scores,mae,mape,r2")
        else:
            with open(args.output_file, "a") as fd:
                fd.write("nb_samples,best_scores,mae,mape,r2\n")

    for size in range(args.start, len(all_samples) + 1, args.step):

        # Subset of the samples
        data = all_samples.iloc[:size]

        tuner = OptunaTunerWithRecorder()
        res = tuner(data, objective_name="performance", time_budget=args.time_budget, n_trials=args.n_trials)

        model = res.model  # A pre-built LightGBM model with the best parameters
        # params = res.params  # Best parameters found for the model
        best_score = res.best_score  # The best MAE score found
        # scores = res.scores  # All the MAE scores found

        predicted = model.predict(validation_samples.drop("performance", axis=1))
        mae = mean_absolute_error(validation_samples["performance"], predicted)
        mape = mean_absolute_percentage_error(validation_samples["performance"], predicted)
        r2 = r2_score(validation_samples["performance"], predicted)

        if args.no_header == False:
            if args.output_file is None:
                print(f"{size},{best_score},{mae},{mape},{r2}")
            else:
                with open(args.output_file, "a") as fd:
                    fd.write(f"{size},{best_score},{mae},{mape},{r2}\n")


if __name__ == "__main__":
    main()
