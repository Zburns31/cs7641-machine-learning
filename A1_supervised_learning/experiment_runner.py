# Import logging first so other libraries don't take precedence for logging
import logging
import os
from config import Config
from pathlib import Path

from pipeline import (
    load_data,
    cast_datatypes,
    create_train_test_split,
    convert_to_binary,
)

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

# Used for loggin purposes
width = os.get_terminal_size().columns

import numpy as np
import pandas as pd
import json

from sklearn.base import BaseEstimator
from typing import Self, List, Dict, Type, Union
from datetime import datetime
from collections import defaultdict

from learners.base_classifier import BaseClassifier
from learners.DT import DTClassifier
from config import Config
from dataset import Dataset, WINE_DATA_SCHEMA


class MLExperimentRunner:
    def __init__(
        self,
        model_class: Type[Union[BaseClassifier, BaseEstimator]],
        base_params: Dict[str, float],
        X: pd.DataFrame,
        y: pd.DataFrame,
        config: Type[Config],
        eval_metric: str,
        param_grid: Dict[str, float],
        test_size: float = 0.3,
    ):
        self.model_class = model_class
        self.base_params = base_params
        self.model = None
        self.X = X
        self.y = y
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.seed = config.RANDOM_SEED
        self.verbose = config.VERBOSE
        self.test_size = test_size

    @property
    def dataset_name(self) -> str:
        return self.X._attrs["name"]

    @property
    def experiment_name(self) -> str:
        # Check if underlying model has been instantiated
        if not self.model:
            self.model = self.init_estimator()

        return self.model.name.replace(" ", "_") + "_" + self.X._attrs["name"]

    def init_estimator(self) -> Union[BaseClassifier, BaseEstimator]:
        return self.model_class(
            self.config, self.eval_metric, self.param_grid, base_params={}
        )

    def run_experiment(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        param_range: float,
        base_params: Dict[str, float],
    ) -> None:
        experiment_times = {}

        if self.config.VERBOSE:
            logger.info(
                f"Running Experiment: {self.model.name} | Parameter Name: {param_name} = {param_range.tolist()}"
            )

        start = datetime.now()

        # default_params = {}
        if param_name in base_params:
            # For simplicity, if we are running an experiment on a hyperparameter that is
            # being used across other experiments with a specific value, set to default
            base_params = base_params.copy()
            del base_params[param_name]

        # Choose
        # default_params = default_params or base_params
        estimator = self.model_class(
            self.config, self.param_grid, self.eval_metric, base_params
        )

        if self.config.VERBOSE:
            logger.info(json.dumps(estimator.get_params(), indent=4))

        best_param_value = estimator.plot_validation_curve(
            X,
            y,
            dataset_name=self.dataset_name,
            param_name=param_name,
            param_range=param_range,
            save_plot=True,
        )

        estimator.set_params(**{param_name: best_param_value})
        estimator.plot_learning_curve(
            X,
            y,
            param_name=param_name,
            dataset_name=self.dataset_name,
            save_plot=True,
        )
        end = datetime.now()
        run_time = end - start

        experiment_times[f"{estimator.name}_{param_name}"] = run_time.seconds
        return experiment_times

    def main(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        self.model = self.init_estimator()

        logger.info(
            f"Estimator Hyperparameter Grid: \n{json.dumps(self.param_grid, default = str, indent = 4)}"
        )

        logger.info(
            f"Starting Experiments for: {self.model.name} | Dataset Name: {self.dataset_name}"
        )

        logger.info(f"Experiment Name: {self.experiment_name}")

        experiment_details = defaultdict(list)
        for param_name, param_range in self.param_grid.items():

            experiment_details[self.model.name].append(
                self.run_experiment(
                    features, target, param_name, param_range, self.base_params
                )
            )
            logger.info("-" * width)

        return experiment_details


if __name__ == "__main__":
    #############################################################################
    # Setup logging and parameters
    DATA_PROCESSING_PARAMS = {
        "profile_report": False,
    }

    ML_PREPROCESS_PARAMS = {
        "shuffle": True,
        "stratify": True,
        "test_size": 0.3,
        "class_weights": None,
    }

    config = Config(
        data_procesing_params=DATA_PROCESSING_PARAMS,
        ml_processing_params=ML_PREPROCESS_PARAMS,
        verbose=True,
    )

    #############################################################################
    # Start of Data Processing
    VERBOSE_MODE = True
    target = "quality"
    wine_df = load_data(config, "wine-quality-white-and-red.csv", delimiter=",")
    wine_df.attrs = {"name": "wine_quality"}
    wine_df = wine_df.pipe(cast_datatypes, column_type_map=WINE_DATA_SCHEMA).pipe(
        convert_to_binary, column=target, threshold=6
    )
    X_TRAIN, X_TEST, y_train, y_test, wine_X, wine_y = create_train_test_split(
        wine_df, target, config, verbose=VERBOSE_MODE
    )

    logger.info(f"Finished Processing Dataset: {wine_df._attrs['name']}")
    print("-" * width)
    #############################################################################
    # ML Experiments
    logger.info("Beginning ML Experiments")

    eval_metric = "accuracy"
    base_params = {"max_depth": 5}
    # Outlines what experiments we want to run. These get passed to the underlying estimator
    param_grid = {
        "max_depth": np.arange(1, 21),
        "ccp_alpha": np.linspace(0.001, 0.1, 100),
        "min_samples_leaf": np.arange(1, 101, 10),
    }

    dt_experiment = MLExperimentRunner(
        DTClassifier, base_params, wine_X, wine_y, config, eval_metric, param_grid
    )
    logger.info(f"Experiment Name: {dt_experiment.experiment_name}")

    experiment_times_dict = dt_experiment.main(features=wine_X, target=wine_y)
    logger.info(json.dumps(dict(experiment_times_dict), indent=4))
