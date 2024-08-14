from config import Config, DATA_PROCESSING_PARAMS, ML_PREPROCESS_PARAMS
from pathlib import Path
from typing import Type, Dict, Tuple

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


def load_data(
    config: Type[Config], dataset_path: str, verbose: bool = False, **kwargs
) -> None:
    """
    Loads data from the specified path.
    """
    try:
        data_path = Path(config.DATA_DIR, dataset_path)
        data = pd.read_csv(data_path, **kwargs)
        if verbose:
            logger.info(f"Loading Dataset: {data_path}")

    except FileNotFoundError:
        print(f"Dataset not found in location: {data_path}")

    logger.info("Data loaded successfully")
    logger.info(
        f"Number of Rows: {data.shape[0]} | Number of Features: {len(data.columns)}"
    )
    return data


def cast_datatypes(df, column_type_map: Dict[str, str], ordered: bool = True) -> None:
    """
    Casts datatypes to the appropriate formats based on the data content and orders categories if applicable. Automatically sets
    order of categories via:
        - If numeric, sort numerically
        - If string, sort lexicographically

    Args:
        column_type_map (Dict[str, str]): _description_
        ordered (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: Raises error if DF is not instantiated yet
    """
    if df is not None and not df.empty:
        for column, dtype in column_type_map.items():
            if column in df.columns:
                df[column] = df[column].astype(dtype)

                if dtype == "category":
                    if df[column].dtype.name == "category":
                        if pd.api.types.is_numeric_dtype(df[column].cat.categories):
                            df[column].cat.reorder_categories(
                                sorted(df[column].cat.categories, key=float),
                                ordered=ordered,
                            )
                        else:
                            df[column].cat.reorder_categories(
                                sorted(df[column].cat.categories), ordered=ordered
                            )
    else:
        raise ValueError(
            "Data not loaded. Please load the data first using the load_data method"
        )

    # Return only columns passed in
    df = df.loc[:, list(column_type_map.keys())]
    return df


def convert_to_binary(df, column: pd.Series, threshold: float) -> pd.Series:
    """
    Convert a numeric DataFrame column into a binary column based on a specified threshold.

    Args:
        df_column (pd.Series): The numeric column to be converted.
        threshold (float): The threshold value for converting to binary.

    Returns:
        pd.Series: A binary column where values are 1 if they are greater than or equal to the threshold, otherwise 0.
    """
    df[column] = df[column].apply(lambda x: 1 if x >= threshold else 0)
    return df


def create_train_test_split(
    df,
    target_col: str,
    config: Type[Config],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = df.loc[:, df.columns != target_col]
    y = df[target_col]

    stratify = y if config.ML_PREPROCESS_PARAMS["stratify"] else None
    X_TRAIN, X_TEST, y_train, y_test = train_test_split(
        X,
        y,
        random_state=config.RANDOM_SEED,
        test_size=config.ML_PREPROCESS_PARAMS["test_size"],
        shuffle=config.ML_PREPROCESS_PARAMS["shuffle"],
        stratify=stratify,
    )
    if verbose:
        logger.info(f"Train Set Size: {len(X_TRAIN)} | Test Set Size: {len(X_TEST)}")

    return X_TRAIN, X_TEST, y_train, y_test, X, y
