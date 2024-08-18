import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import Config
from pathlib import Path
from utilities import get_directory
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Self, Any, Callable, Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import learning_curve, validation_curve, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)


class BaseClassifier(ClassifierMixin, BaseEstimator, ABC):
    """
    TODO: Nest directories for images and results for easier sorting

    Args:
        ClassifierMixin (_type_): _description_
        BaseEstimator (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(
        self,
        model: BaseEstimator,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
    ):
        self.model = model
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.seed = config.RANDOM_SEED
        self.verbose = config.VERBOSE

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_

    @property
    def unique_hyperparameters(self) -> List[str]:
        return list(self.param_grid.keys())

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the current parameters for the learner. This passes the call back to the learner from learner()

        :param deep: If true, fetch deeply
        :return: The parameters
        """
        return self.model.get_params(deep)

    def set_params(self, **params) -> None:
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        print(f"New Parameters Set: {params}")
        self.model.set_params(**params)

    def get_scorer(self, metric_name: str) -> Callable:
        """
        Given a metric name, return the corresponding sklearn scoring function

        :param metric_name: A string representing the metric name (e.g., 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix')
        :return: A callable scoring function from sklearn.metrics
        """
        scorers = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "fbeta": fbeta_score,
            "roc_auc": roc_auc_score,
        }

        if metric_name in scorers:
            return make_scorer(scorers[metric_name])
        else:
            raise ValueError(
                f"Unknown metric name: {metric_name}. Valid options are: {list(scorers.keys())}"
            )

    @abstractmethod
    def fit(self, X, y, verbose=True) -> Self:
        pass

    @abstractmethod
    def predict(self, X, y, proba=False) -> Self:
        pass

    def plot_learning_curve(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        dataset_name: str,
        cv: int = 5,
        save_plot: bool = True,
        show_plot: bool = False,
    ):
        """Generates a learning curve for the underlying model

        TODO: Add parameter for custom training set sizes

        Args:
            X (np.array): Represnets the predictor/independent features
            y (np.array): Represents the target/repsonse variable
            param_name (str): Hyperparameter that we are using in the underlying model
            dataset_name (str): Name of the dataset we are training/predicting against
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            save_plot (bool, optional): Whether to save the generated charts or not. Defaults to True.
            show_plot (bool, optional): Whether to plot the generated charts or not. Defaults to False.
        """
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, cv=cv)

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        annotation = (
            param_name.replace("_", " ").capitalize()
            + " = "
            + str(self.model.get_params().get(param_name))
        )
        plt.title(f"Learning Curve ({self.name}) | {annotation}")
        plt.xlabel("# of Training Observations")
        plt.ylabel("Score")

        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc="best")
        plt.tight_layout()

        if save_plot:
            model_name = self.name.replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_{param_name}_learning_curve.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    dataset_name,
                    model_name,
                    param_name,
                ),
                plot_name,
            )
            plt.savefig(image_path)

            if self.verbose:
                print(f"Saving Learning Curve to: {image_path.relative_to(Path.cwd())}")

        if show_plot:
            plt.show()

    # def compute_best_param_value(
    #     self,
    #     param_range: Union[np.ndarray, List[float]],
    #     mean_test_scores: np.ndarray,
    #     incr_threshold: float = 0.5,
    # ) -> float:
    #     """
    #     Returns the best parameter value based on maximum test score and when test scores
    #     stop increasing by more than a threshold percentage.

    #     Args:
    #         param_range (Union[np.ndarray, List[float]]): The range of parameter values.
    #         test_scores (np.ndarray): Array of test scores.
    #         incr_threshold (float, optional): Threshold percentage for stopping criteria. Defaults to 0.5.

    #     Returns:
    #         float: The best parameter value.
    #     """

    #     best_param_value1 = None
    #     # Loop to find the parameter value when test accuracy stops increasing by more than the threshold
    #     for i in range(1, len(mean_test_scores)):
    #         if mean_test_scores[i] - mean_test_scores[i - 1] <= incr_threshold:
    #             best_param_value1 = param_range[i - 1]
    #             break

    #     # If no such point is found, return the best parameter value based on max test score
    #     best_param_index = np.argmax(mean_test_scores)
    #     best_param_value2 = param_range[best_param_index]

    #     if best_param_value1 == best_param_value2:
    #         return best_param_value1
    #     else:
    #         return best_param_value2

    def compute_best_param_value(
        self, train_scores, test_scores, param_range, threshold=0.01
    ):
        """Returns the best parameter value based on maximum test score and when test scores stop increasing by more than a threshold percentage

        Args:
            train_scores (_type_): Train scores for the CV run
            test_scores (_type_): Test scores for the CV run
            param_range (_type_): Hyperparameter values
            threshold (float, optional): Minimum increase in eval metric for test scores. Defaults to 0.01.

        Returns:
            int or float: Best hyperparameter value
        """

        # Compute the mean scores across the cross-validation folds
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Initialize variables to track the best parameter
        # best_param_idx = 0
        best_test_score = test_scores_mean[0]

        for idx in range(1, len(test_scores_mean)):
            # If test score decreases, break and return the previous parameter value
            if test_scores_mean[idx] < test_scores_mean[idx - 1]:
                best_param_idx = idx
                break

            # Check if the difference between training and test scores is within the acceptable range
            score_difference = train_scores_mean[idx] - test_scores_mean[idx]

            # Update best parameter if the test score is higher and close to the train score
            if (
                test_scores_mean[idx] > best_test_score
                and score_difference <= threshold
            ):
                best_param_idx = idx
                best_test_score = test_scores_mean[idx]
                break

        return param_range[best_param_idx]

    def plot_validation_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        param_name: str,
        param_range: Union[np.ndarray, List[float]],
        cv: int = 5,
        save_plot: bool = True,
        show_plot: bool = False,
        incr_threshold: float = 0.01,
    ) -> int:
        """Plot a validation curve with the range of hyperparameter values on the X-axis and the metric score on the Y-axis. This function
        also returns the value of the specified hyperparameter with the best testing score

        Args:
            X (np.array): Represnets the predictor/independent features
            y (np.array): Represents the target/repsonse variable
            dataset_name (str): Name of the dataset we are training/predicting against
            param_name (str): Hyperparameter that we are using in the underlying model
            param_range ()
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            save_plot (bool, optional): Whether to save the generated charts or not. Defaults to True.
            show_plot (bool, optional): Whether to plot the generated charts or not. Defaults to False.

        Returns:
            int: The value of the specified hyperparameter that returns the best mean test score
        """

        train_scores, test_scores = validation_curve(
            self.model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
        )

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        train_scores_std = np.std(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100
        test_scores_std = np.std(test_scores, axis=1) * 100

        # Find the parameter value when test accuracy stops increasing by more than incr_threshold %
        best_param_value = self.compute_best_param_value(
            train_scores=train_scores,
            test_scores=test_scores,
            param_range=param_range,
            threshold=incr_threshold,
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        cleaned_param_name = param_name.replace("_", " ").capitalize()
        plt.title(f"Validation Curve for {param_name}")
        plt.xlabel(cleaned_param_name)
        plt.ylabel("Score")

        # Ensure X-labels are discrete values if param range is discrete
        # import pdb

        # pdb.set_trace()
        if np.all(np.mod(param_range, 1) == 0):
            plt.xticks(np.arange(min(param_range), max(param_range) + 1, step=1))

        plt.plot(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )

        plt.plot(
            param_range, test_scores_mean, label="Cross-validation score", color="g"
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc="best")
        plt.tight_layout()

        if save_plot:
            model_name = self.name.replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_validation_curve.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    dataset_name,
                    model_name,
                    param_name,
                ),
                plot_name,
            )
            plt.savefig(image_path)

            if self.verbose:
                # Relative path
                print(
                    f"Saving Validation Curve to: {image_path.relative_to(Path.cwd())}"
                )

        if show_plot:
            plt.show()

        if isinstance(best_param_value, float):
            best_param_value = round(best_param_value, 3)

        return best_param_value

    def plot_validation_curve_by_groups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        param_name: str,
        param_range: Union[np.ndarray, List[str]],
        cv: int = 5,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> str:
        """Plot a validation curve with the range of hyperparameter values on the X-axis and the metric score on the Y-axis.
        This function also returns the value of the specified hyperparameter with the best testing score.

        Args:
            X (np.array): Represents the predictor/independent features.
            y (np.array): Represents the target/response variable.
            dataset_name (str): Name of the dataset we are training/predicting against.
            param_name (str): Hyperparameter that we are using in the underlying model.
            param_range (Union[np.ndarray, List[str]]): Range of hyperparameter values.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            save_plot (bool, optional): Whether to save the generated charts or not. Defaults to True.
            show_plot (bool, optional): Whether to plot the generated charts or not. Defaults to False.

        Returns:
            str: The value of the specified hyperparameter that returns the best mean test score.
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        cleaned_param_name = param_name.replace("_", " ").capitalize()
        plt.title(f"Validation Curve for {param_name}")
        plt.xlabel(cleaned_param_name)
        plt.ylabel("Score")

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.grouping_vals)))

        # Plot for each grouping value
        metric_scores = []
        for idx, (group_param_name, grouping_val) in enumerate(self.grouping_vals):
            # Update the model with the current grouping value
            self.model.set_params(**{group_param_name: grouping_val})

            train_scores, test_scores = validation_curve(
                self.model,
                X,
                y,
                param_name=param_name,
                param_range=param_range,
                cv=cv,
            )

            train_scores_mean = np.mean(train_scores, axis=1) * 100
            train_scores_std = np.std(train_scores, axis=1) * 100
            test_scores_mean = np.mean(test_scores, axis=1) * 100
            test_scores_std = np.std(test_scores, axis=1) * 100

            # best_param_value = self.compute_best_param_value(train_scores, test_scores, param_range, threshold = 0.01)
            best_param_value = param_range[np.argmax(test_scores_mean)]
            metric_scores.append(best_param_value)

            # Plotting the curve for the current grouping_val
            plt.plot(
                param_range,
                train_scores_mean,
                label=f"Training score ({grouping_val})",
                linestyle="--",
                color=colors[idx],
            )
            plt.plot(
                param_range,
                test_scores_mean,
                label=f"Cross-validation score ({grouping_val})",
                color=colors[idx],
            )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc="best")
        plt.tight_layout()

        if save_plot:
            model_name = self.name.replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_validation_curve.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    dataset_name,
                    model_name,
                    param_name,
                ),
                plot_name,
            )
            plt.savefig(image_path)

            if self.verbose:
                # Relative path
                print(
                    f"Saving Validation Curve to: {image_path.relative_to(Path.cwd())}"
                )

        if show_plot:
            plt.show()

        # For string-based parameters, simply return the best grouping value based on test scores
        return max(metric_scores)

    def compute_run_times(self, X: pd.DataFrame, y: pd.Series, cv=5) -> pd.DataFrame:
        clf = self.model.__class__()
        scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
        scores["model"] = [self.model.__class__.__name__] * cv

        return pd.DataFrame(scores)

    def plot_confusion_matrix(self):
        pass

    def plot_roc_curve(self, binary_clf=False):
        pass
