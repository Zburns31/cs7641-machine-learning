import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

from config import Config
from utilities import get_directory
from learners.base_classifier import BaseClassifier
from typing import Dict, Tuple, List, Self, Type, Union
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve,
    cross_val_score,
)


class DTClassifier(BaseClassifier):
    """TODO: Add default params for DT"""

    def __init__(
        self,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
        base_params: Dict[str, float],
    ):
        super().__init__(
            model=DecisionTreeClassifier(
                random_state=config.RANDOM_SEED, **base_params
            ),
            config=config,
            param_grid=param_grid,
            eval_metric=eval_metric,
        )

    def fit(self, X: np.ndarray, y: np.array):
        """Fits the underlying estimator/model to the data

        Args:
            X np.array: Represents the independentfeatures used to train the model
            y np.array: Represents the target/response variable
            verbose (bool, optional): Flag to log additional information to the console

        Returns:
            DTClassifier: Returns the fitted model
        """
        if self.verbose:
            print("Fitting the model...")
        self.model.fit(X, y)
        if self.verbose:
            print("Model fitting completed.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the mean response given a set of independent features

        Args:
            X (np.ndarray): Represents the independent features

        Returns:
            np.ndarray: Returns the mean response given the predictors
        """
        return self.model.predict(X)

    def plot_training_run_time(self, X, y, cv=5, show_plot=False):
        """
        TODO

        Args:
            X (_type_): _description_
            y (_type_): _description_
            cv (int, optional): _description_. Defaults to 5.
            show_plot (bool, optional): _description_. Defaults to False.
        """
        start_time = time.time()
        scores = cross_val_score(self.model, X, y, cv=cv)
        end_time = time.time()

        runtime = end_time - start_time

        plt.figure()
        plt.title(f"Training Runtime ({self.name})")
        plt.xlabel("Training examples")
        plt.ylabel("Runtime (seconds)")
        plt.bar(["Run Time"], [runtime])

        if show_plot:
            plt.show()

    def prune_tree(self):
        pass

    def export_tree(self):
        pass
