import logging
from config import Config
from pathlib import Path

logging_config = Config(
    data_procesing_params={}, ml_processing_params={}, verbose=False
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.FileHandler(
            Path(logging_config.LOGS_DIR, "ml_experiments.log"), mode="w+"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

import argparse
import os
import json
import numpy as np
import pandas as pd

from typing import List, Dict, Any, Tuple
from dataset import Dataset, WINE_DATA_SCHEMA
from experiment_runner import MLExperimentRunner
from learners.DT import DTClassifier
from experiment_runner import MLExperimentRunner
from config import Config, DATA_PROCESSING_PARAMS, ML_PREPROCESS_PARAMS
from utilities import print_tuples

from pipeline import (
    load_data,
    cast_datatypes,
    create_train_test_split,
    convert_to_binary,
)

# Used for logging purposes
width = os.get_terminal_size().columns

###############################################################################
# ML Experiment Parameter Configurations

# Decision Tree
DT_EVAL_METRIC = "accuracy"
# Outlines what experiments we want to run. These get passed to the underlying estimator
DT_PARAM_GRID = {
    "max_depth": np.arange(1, 21),
    "ccp_alpha": np.linspace(0.001, 0.1, 100),
    "min_samples_leaf": np.arange(1, 101, 10),
}

# GBDT

# KNN

# SVM

# ANN


###############################################################################
def run_experiment_configuration(
    datasets: List[Tuple[pd.DataFrame, pd.Series]],
    estimator: MLExperimentRunner,
    base_params: Dict[str, float],
    eval_metric: str,
    param_grid: Dict[str, float],
    config: Config,
) -> List[dict]:
    """
    Runs the provided MLExperimentRunner against all datasets passed in

    Args:
        datasets (List[Dataset]): Dataset objects to fit the models against
        estimator (MLExperimentRunner): Learning algorithm to run experiment methods against
        base_params (Dict[str, float]): Base estimator parameter settings to use across all experiments (Does not apply if the experiment is running the base_param[param_name])
        eval_metric (str): Evaluation metric
        param_grid (Dict[str, float]): Hyperparameters and associated ranges to run experiments against
        config (Config): Project configuration settings

    Returns:
        List[dict]: Experiment run times for each configuration
    """

    experiment_times = []

    for X, y in datasets:
        experiment = MLExperimentRunner(
            estimator, base_params, X, y, config, eval_metric, param_grid
        )
        if config.VERBOSE:
            print(f"Experiment Name: {experiment.experiment_name}")

        experiment_times_dict = experiment.main(features=X, target=y)
        experiment_times.append(dict(experiment_times_dict))
        print("-" * width)

    return experiment_times


if __name__ == "__main__":
    #############################################################################
    # CLI Args
    parser = argparse.ArgumentParser(
        prog="ML Experiment Runner", description="Perform some ML experiments"
    )

    # Define the valid choices for the experiment type
    experiment_choices = ["dt", "ann", "boosting", "knn", "svm", "all"]
    parser.add_argument(
        "--experiment_type",
        choices=experiment_choices,
        default=False,
        help="Type of experiment to run. Choose from: dt, ann, boosting, knn, svm, all",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If true, provide verbose output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to set for non-determinstic portions of the experiments",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Number of threads (defaults to -1)",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Dry run. Only reports passed in CLI arguments",
    )
    parser.add_argument(
        "-p",
        "--profile-report",
        action="store_true",
        help="Flag to generate a YData Profile report on the dataset. Defaults to False",
    )
    args = parser.parse_args()
    verbose = args.verbose

    if args.profile_report:
        DATA_PROCESSING_PARAMS["profile_report"] = True
    #############################################################################
    # Setup logging and parameters

    config = Config(
        data_procesing_params=DATA_PROCESSING_PARAMS,
        ml_processing_params=ML_PREPROCESS_PARAMS,
        verbose=verbose,
    )

    if verbose:
        print(f"{parser.prog} CLI Arguments")
        print("-" * width)
        print(print_tuples(args._get_kwargs()))
        print("-" * width)

    #############################################################################
    # Data Processing - Collect & Process Datasets

    # Wine Data
    target = "quality"
    wine_df = load_data(config, "wine-quality-white-and-red.csv", delimiter=",")
    wine_df.attrs = {"name": "wine_quality"}
    # fmt: off
    wine_df = (
        wine_df
        .pipe(cast_datatypes, column_type_map=WINE_DATA_SCHEMA)
        .pipe(convert_to_binary, column=target, threshold=6)
    )
    # fmt: on

    X_TRAIN, X_TEST, y_train, y_test, wine_X, wine_y = create_train_test_split(
        wine_df, target, config, verbose=verbose
    )

    # Bank Marketing

    datasets = [(wine_X, wine_y)]

    print("Finished Processing Dataset")
    print("-" * width)
    #############################################################################
    # ML Experiments

    print("Beginning ML Experiments")
    experiment_results = []

    exp_type = args.experiment_type
    if exp_type in ["dt", "all"]:
        # Set across experiment configs to avoid overfitting
        dt_base_params = {"max_depth": 5}
        dt_experiment_results = run_experiment_configuration(
            datasets=datasets,
            estimator=DTClassifier,
            base_params=dt_base_params,
            eval_metric=DT_EVAL_METRIC,
            param_grid=DT_PARAM_GRID,
            config=config,
        )

    if exp_type in ["ann", "all"]:
        raise NotImplementedError
    if exp_type in ["boosting", "all"]:
        raise NotImplementedError
    if exp_type in ["knn", "all"]:
        raise NotImplementedError
    if exp_type in ["svm", "all"]:
        raise NotImplementedError

    print(json.dumps(dt_experiment_results, indent=4))
