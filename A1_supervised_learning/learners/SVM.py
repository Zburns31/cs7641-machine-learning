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

from sklearn.svm import SVC


class SVMClassifier(BaseClassifier):

    def __init__(
        self,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
        base_params: Dict[str, float] = {},
    ):
        super().__init__(
            model=SVMClassifier(random_state=config.RANDOM_SEED, **base_params),
            config=config,
            param_grid=param_grid,
            eval_metric=eval_metric,
        )
        self.grouping_vals = list(
            zip(["kernel"] * 5, ["linear", "poly", "rbf", "sigmoid", "precomputed"])
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
