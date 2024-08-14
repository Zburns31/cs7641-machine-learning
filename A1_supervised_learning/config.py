from typing import Dict, Union

DATA_PROCESSING_PARAMS = {
    "profile_report": False,
}

ML_PREPROCESS_PARAMS = {
    "shuffle": True,
    "stratify": True,
    "test_size": 0.3,
    "class_weights": None,
}


class Config:
    RANDOM_SEED = 42
    ARTIFACTS_DIR = "artifacts/"
    IMAGE_DIR = "images/"
    RESULTS_DIR = "results/"
    EDA_DIR = "eda/"
    DATA_DIR = "data/"
    LOGS_DIR = "logs/"

    def __init__(
        self,
        data_procesing_params: Dict[str, bool],
        ml_processing_params: Dict[str, Union[bool, float]],
        verbose: bool = False,
    ):
        self.DATA_PROCESSING_PARAMS = data_procesing_params
        self.ML_PREPROCESS_PARAMS = ml_processing_params
        self.VERBOSE = verbose
