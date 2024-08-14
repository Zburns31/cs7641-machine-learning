from pathlib import Path
from typing import Dict, Any


def get_directory(child_dir: str, *args: str) -> Path:
    path = Path.cwd() / child_dir / Path(*args)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    else:
        print(f"Folder was created: {path}")

    return path


def pretty_print(d: Dict[str, Any], indent=0) -> None:
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pretty_print(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


def print_tuples(tuples_list):
    # Determine the maximum length of the keys for alignment
    max_key_length = max(len(str(key)) for key, _ in tuples_list)

    # Print each tuple in a formatted manner
    for key, value in tuples_list:
        print(f"{key.ljust(max_key_length)} : {value}")


# TODO
# def set_categorical_variable_order(
#     self, col_name: str, category_order: List[Union[str, int]], ordered=True
# ) -> None:

#     df = self.data
#     df[col_name].cat.reorder_categories(
#         new_categories=category_order, ordered=ordered
#     )
#     self.data = df

# def convert_target_to_integer(self, target_col: str) -> pd.DataFrame:
#     """If the target column is a string or categorical value, convert to an integer

#     Args:
#         target_col (str): Target column name

#     Returns:
#         pd.DataFrame: New DF with tagret column integer encoded
#     """
#     df = self.data
#     if self.verbose:
#         mapping = dict(enumerate(df[target_col].cat.categories))
#         logger.info(f"Target Variable Encoding: {mapping}")

#     df[target_col] = df[target_col].cat.codes
#     self.data = df
