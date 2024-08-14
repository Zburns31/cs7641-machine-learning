import logging
from config import Config
from pathlib import Path

logging_config = Config(
    data_procesing_params={}, ml_processing_params={}, verbose=False
)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.FileHandler(
            Path(logging_config.LOGS_DIR, "data_processing.log"), mode="w+"
        ),
        logging.StreamHandler(),
    ],
)

import pandas as pd
import numpy as np
from learners.DT import DTClassifier
from typing import Dict, Tuple, List, Type, Union, Self
from utilities import get_directory

import matplotlib.pyplot as plt

plt.set_loglevel("WARNING")
import seaborn as sns

from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split

pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_columns", 999)

###############################################################################
# Dataset Schemas
WINE_DATA_SCHEMA = {
    "fixed acidity": "float64",
    "volatile acidity": "float64",
    "citric acid": "float64",
    "residual sugar": "float64",
    "chlorides": "float64",
    "free sulfur dioxide": "float64",
    "total sulfur dioxide": "float64",
    "density": "float64",
    "pH": "float64",
    "sulphates": "float64",
    "alcohol": "float64",
    "quality": "int64",
}

BANK_MARKETING_TYPES_MAP = {
    "age": "int64",
    "job": "category",
    "marital": "category",
    "education": "category",
    "default": "category",
    "housing": "category",
    "loan": "category",
    "contact": "category",
    "month": "category",
    "day_of_week": "category",
    "duration": "int64",
    "campaign": "int64",
    "pdays": "int64",
    "previous": "int64",
    "poutcome": "category",
    "emp.var.rate": "float64",
    "cons.price.idx": "float64",
    "cons.conf.idx": "float64",
    "euribor3m": "float64",
    "nr.employed": "float64",
    "accepted": "int64",
}
###############################################################################


class Dataset:
    """
    Wrapper class to help automate common data analysis elements. This is primarily used for already cleaned datasets

    TODO:
        - Add functionality for creating a balanced dataset from an inbalanced one (i.e. straitified sampling)
        - Add standard or min-max scaling
        - Add functionality for creating ML Pipeline for train and test sets

    """

    def __init__(self, data_path: str, data_delimiter: str, config: Config):
        self.data_path: str = (
            data_path  # CHange to dataset_path and make name a property
        )
        self.data = None
        self.delimiter: str = data_delimiter
        self.config: Type[Config] = config
        self.verbose: bool = config.VERBOSE
        self.outliers = None

    @property
    def dataset_name(self) -> str:
        return self.data_path.replace(".csv", "").replace("-", "_")

    def summary_statistics(self, target_col: str, normalize_counts=True) -> None:
        """Provides summary statistics of all columns"""
        if self.data is None or self.data.empty:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method"
            )
        else:
            target_class_dist = (
                self.data[target_col]
                .value_counts(normalize=normalize_counts)
                .sort_index()
                .reset_index()
                .style.hide()  # hide index
                .format({"proportion": "{:,.2%}"})
                .to_string()
            )
            # return target_class_dist
            summary_df = self.data.describe(
                include="all", percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
            ).round(2)

            if self.verbose:
                logger.info(
                    "Target Column: {} | Class Distribution: {}".format(
                        target_col, target_class_dist
                    )
                )
                logger.info("Data Summary: \n{}".format(summary_df))

    def check_missing_values(self):
        """Checks for any missing values in the dataset."""
        has_nulls = self.data.isnull().values.any()
        if has_nulls:
            logger.warning("Warning Missing Values Detected")

    def check_outliers(self):
        """
        Detects outliers in each column of the DataFrame using the IQR method.

        Args:
        data (pd.DataFrame): The input DataFrame.

        Returns:
        dict: A dictionary where keys are column names and values are lists of indices of outliers.
        """
        if self.data is not None or not self.data.empty:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            outliers_dict = {}

            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = self.data.index[
                    (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                ].tolist()

                if outlier_indices:
                    outliers_dict[col] = outlier_indices

            logger.warning(
                f"Outliers Detected in columns: {list(outliers_dict.keys())}"
            )
            return outliers_dict
        else:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method."
            )

    def create_violin_plots(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        target: str,
        save_plot: bool = True,
        show_plot: bool = False,
    ):
        """
        Creates a grid of violin plots using seaborn with the x-axis as the class label and the y-axis as the feature value.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the features and target.
        feature_list (List[str]): List of independent features to plot.
        target (str): The target column name to plot features against.
        """
        # Melt the dataframe to long-form or tidy-form
        melted_df = pd.melt(
            df,
            id_vars=[target],
            value_vars=feature_list,
            var_name="Feature",
            value_name="Value",
        )

        # Create a FacetGrid for the violin plots
        # Uses the implied categorical ordering for the target variable
        g = sns.FacetGrid(melted_df, col="Feature", col_wrap=3, sharey=False, height=4)
        g.map(
            sns.violinplot,
            target,
            "Value",
            palette="muted",
            inner="quartile",
            # order=label_order,
            hue=melted_df[target],
            legend=False,
        )

        # Set titles and labels
        for ax in g.axes.flat:
            ax.set_title(ax.get_title().split("=")[-1])
            ax.set_xlabel(target)
            ax.set_ylabel("Value")

        plt.tight_layout()

        if save_plot:
            plot_name = f"{self.dataset_name}_facet_violin_plot.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    self.dataset_name,
                    self.config.EDA_DIR,
                ),
                plot_name,
            )
            plt.savefig(image_path, dpi=200)

            if self.verbose:
                # Relative path
                print(
                    f"Saving Facet Violin Plot to: {image_path.relative_to(Path.cwd())}"
                )

        if show_plot:
            plt.show()

    def plot_heatmap(
        self,
        heatmap_type: str = "correlation",
        save_plot: bool = True,
        show_plot: bool = False,
    ):
        """
        Creates a heatmap of either correlation or covariance matrix.

        Parameters:
        heatmap_type (str): Type of heatmap to create. Should be 'correlation' or 'covariance'.
        save_plot (bool): Whether to save the plot as an image.
        show_plot (bool): Whether to display the plot.
        """
        if self.data is None or self.data.empty:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method."
            )

        if heatmap_type not in ["correlation", "covariance"]:
            raise ValueError(
                "Invalid heatmap type. Please choose either 'correlation' or 'covariance'."
            )

        if heatmap_type == "correlation":
            matrix = self.data.corr()
            title = "Correlation Heatmap"
        else:
            matrix = self.data.cov()
            title = "Covariance Heatmap"

        mask = np.zeros_like(matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidth=0.5,
        )
        plt.title(title)
        plt.grid(False)
        plt.tight_layout()

        if save_plot:
            plot_name = f"{self.dataset_name}_{heatmap_type}_heatmap.png"
            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    self.dataset_name,
                    self.config.EDA_DIR,
                ),
                plot_name,
            )
            plt.savefig(image_path, dpi=200)

            if self.verbose:
                # Relative path
                print(f"Saving {title} to: {image_path.relative_to(Path.cwd())}")

        if show_plot:
            plt.show()

    def generate_data_profile_report(self) -> None:
        profile = ProfileReport(
            self.data, title=f"{self.dataset_name}: Profiling Report"
        )
        report_path = get_directory(
            self.config.ARTIFACTS_DIR,
            self.config.IMAGE_DIR,
            self.dataset_name,
            self.config.EDA_DIR,
        )
        profile.to_file(report_path / f"{self.dataset_name}_profile_report.html")

    def run(self, target_col: str, column_types: Dict[str, str]) -> Tuple[
        Self,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if self.verbose:
            logger.info(f"Reading, Loading and Processing Dataset: {self.dataset_name}")

        self.load_data(verbose=self.verbose, delimiter=self.delimiter)
        self.cast_datatypes(column_type_map=column_types)

        # Check if target variable needs to be integer encoded
        if self.data[target_col].dtype.name in ["object", "category"]:
            self.convert_target_to_integer(target_col)

        self.summary_statistics(target_col=target_col)
        self.check_missing_values()
        self.outliers = self.check_outliers()

        # Run YData Profile Report
        generate_report = self.config.DATA_PROCESSING_PARAMS.get(
            "profile_report", False
        )
        if generate_report:
            self.generate_data_profile_report()

        # Data Vizualization Processing
        feature_list = self.data.loc[:, self.data.columns != target_col].columns

        self.create_violin_plots(
            self.data,
            feature_list,
            target_col,
            save_plot=True,
        )
        self.plot_heatmap(heatmap_type="correlation")
        self.plot_heatmap(heatmap_type="covariance")

        # ML Processing
        X_TRAIN, X_TEST, y_train, y_test, X, y = self.create_train_test_split(
            target_col
        )
        self.features_list = list(X.columns)
        self.target = y.name

        return self, X_TRAIN, X_TEST, y_train, y_test, X, y


if __name__ == "__main__":
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
    dataset, X_TRAIN, X_TEST, y_train, y_test, X, y = Dataset(
        "winequality-white.csv", data_delimiter=";", config=config
    ).run(target_col="quality", column_types=WINE_DATA_SCHEMA)

    logger.info("Finished Processing Dataset")
    #############################################################################

    # parameter_grid = {"max_depth": np.linspace(0, 10, 1)}
    # eval_metric = "accuracy"
    # wine_dt = DTClassifier(config, parameter_grid, eval_metric)

    # scoring_func = wine_dt.get_scorer("accuracy")

    # best_param_value = wine_dt.plot_validation_curve(
    #     dataset.features,
    #     dataset.target,
    #     dataset_name="winequality-white",
    #     param_name="max_depth",
    #     param_range=np.arange(1, 11),
    #     save_plot=True,
    # )

    # wine_dt.set_params(max_depth=best_param_value)
    # wine_dt.plot_learning_curve(
    #     dataset.features,
    #     dataset.target,
    #     param_name="max_depth",
    #     dataset_name="winequality-white",
    #     save_plot=True,
    # )
