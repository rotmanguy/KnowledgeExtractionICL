# MIT License
# Copyright (c) [2025] [Anonymized]
# See LICENSE file for full license text

import os
import pandas as pd
from typing import Dict, Tuple, Any


class DatasetLoader:
    """
    A class for loading various text classification datasets.

    This class handles loading, preprocessing, and splitting datasets for text classification tasks,
    with support for different dataset formats and sources.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the dataset loader with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters including:
                - data_dir: Directory containing the dataset files
                - objective: Classification objective
                - label_map: Mapping from numeric labels to text labels
        """
        self.data_dir = config["data_dir"]
        self.objective = config["objective"]
        self.label_map = config["label_map"]

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load a generic classification dataset with standard train/test CSV files.

        Returns:
            Tuple containing (train_df, test_df)
        """
        # Load train and test datasets from separate files
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Add textual labels
        train_df["textual_label"] = train_df["label"].map(self.label_map)
        test_df["textual_label"] = test_df["label"].map(self.label_map)

        return train_df, test_df
