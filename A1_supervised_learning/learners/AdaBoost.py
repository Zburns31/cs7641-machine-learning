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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class ABoostingClassifier(BaseClassifier):
    """TODO: Add default params for DT"""

    def __init__(
        self,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
        base_params: Dict[str, float],
    ):
        # Seperate hyperparams for Adaboost and Underlying Estimator
        base_est_params = {
            k.replace("estimator__", ""): v
            for k, v in base_params.items()
            if "estimator__" in k
        }
        base_params = {k: v for k, v in base_params.items() if "estimator__" not in k}

        super().__init__(
            model=AdaBoostClassifier(
                random_state=config.RANDOM_SEED,
                estimator=DecisionTreeClassifier(**base_est_params),
                **base_params
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

    def export_tree(self):
        pass
