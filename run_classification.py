# MIT License
# Copyright (c) [2025] [Anonymized]
# See LICENSE file for full license text

import argparse
import asyncio
from dotenv import load_dotenv
import json
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
import logging
from matplotlib import pyplot as plt
import os
import pandas as pd
from pathlib import Path
import random
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from time import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import wandb

from classifiers import ClassifierBase, BinaryClassifier
from criteria_creator import BinaryCriteriaCreator
from dataset_loader import DatasetLoader
from description_creator import BinaryDescriptionCreator


logger: Optional[logging.Logger] = None


def convert_keys_to_int(label_map: Dict[Any, str]) -> Dict[Union[int, Any], str]:
    """
    Convert dictionary keys to integers if all keys are numeric strings.

    Args:
        label_map: Dictionary with keys to potentially convert

    Returns:
        Dictionary with numeric keys converted to integers if applicable
    """
    if not label_map:
        return {}
    if all(str(key).isdigit() for key in label_map):
        return {int(key): value for key, value in label_map.items()}
    else:
        return label_map


def process_label_map(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the label map in the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration dictionary
    """
    if isinstance(config.get("label_map"), str):
        config["label_map"] = json.loads(config["label_map"])
    config["label_map"] = convert_keys_to_int(config["label_map"])
    return config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the classification experiment.

    Returns:
        Parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Classification experiments using LLMs with various prompting strategies."
    )
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_id", type=str, default="gpt-4o", help="ID of the model to use."
    )
    model_group.add_argument(
        "--model_source",
        type=str,
        default="azure",
        choices=["azure", "bedrock"],
        help="Source of the model (e.g., azure, bedrock).",
    )
    model_group.add_argument(
        "--objective", type=str, help="Objective of the experiment."
    )
    model_group.add_argument(
        "--data_dir", type=str, help="Directory containing the dataset."
    )
    model_group.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for model inference."
    )
    fewshot_group = parser.add_argument_group("Few-shot Configuration")
    fewshot_group.add_argument(
        "--num_few_shot_examples",
        type=int,
        nargs="+",
        default=[0, 10, 25, 50, 75, 100],
        help="List of few-shot example counts to use.",
    )
    fewshot_group.add_argument(
        "--num_experiments",
        type=int,
        default=5,
        help="Number of experiments to run for statistical significance.",
    )
    fewshot_group.add_argument(
        "--mix_examples",
        action="store_true",
        default=False,
        help="Allow mixing of examples across classes instead of grouping by label.",
    )
    fewshot_group.add_argument(
        "--sampling_method",
        type=str,
        default="label_distribution",
        choices=["random", "label_distribution"],
        help="Sampling method to use for sampling the few-shot examples. \
            'random' for random sampling, \
            'label_distribution' ensures sampling from all labels while maintaining the label distribution.",
    )
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir", type=str, help="Directory to save the outputs."
    )
    output_group.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable logging with Weights & Biases.",
    )
    output_group.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override existing outputs.",
    )
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the dataset.",
    )
    data_group.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )
    data_group.add_argument(
        "--example_column",
        type=str,
        default="text",
        help="Name of the example column in the dataset.",
    )
    data_group.add_argument(
        "--label_map",
        type=convert_keys_to_int,
        default='{"0": "Negative", "1": "Positive"}',
        help='Label map as a JSON string, e.g., \'{"0": "Negative", "1": "Positive"}\'',
    )
    wandb_group = parser.add_argument_group("Weights & Biases Configuration")
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        default="nlp",
        help="Entity name for Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default="criteria-classification-research",
        help="W&B project name.",
    )
    wandb_group.add_argument(
        "--wandb_run", type=str, default="criteria-experiment", help="W&B run name."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--env_file", type=str, default=".env", help="Path to the .env file"
    )
    return parser.parse_args()


def setup_logger(
    log_dir: Union[str, Path], log_filename: str = "experiment.log"
) -> None:
    """
    Set up a custom logger for the experiment.

    Args:
        log_dir: Directory to save log files
        log_filename: Name of the log file
    """
    global logger
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_filename)
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def init_reports_dict(
    methods: List[str], few_shot_examples: List[int]
) -> Dict[str, Dict[int, List]]:
    """
    Initialize a dictionary to store reports for each method and few-shot configuration.

    Args:
        methods: List of methods to include in the reports dictionary
        few_shot_examples: List of few-shot example counts

    Returns:
        Nested dictionary for storing experiment reports
    """
    reports = {}
    for method in methods:
        reports[method] = {}
        for num_few_shot_examples in few_shot_examples:
            reports[method][num_few_shot_examples] = []
    return reports


def get_bedrock_llm(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs: Dict[str, Any] = None,
) -> ChatBedrock:
    """
    Initialize a Bedrock LLM with the specified configuration.
    """
    # Set default model kwargs with more reasonable timeout
    if model_kwargs is None:
        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "timeout": 120,
            "max_retries": 3,
        }

    # More robust provider detection
    if "anthropic" in model_id:
        provider = "anthropic"
    elif "meta" in model_id:
        provider = "meta"
    elif "mistral" in model_id:
        provider = "mistral"
    elif "amazon" in model_id:
        provider = "amazon"
    else:
        raise ValueError(f"Unable to determine provider for model_id: {model_id}")

    # Get AWS configuration with clear defaults
    credentials_profile_name = os.environ.get("AWS_PROFILE")
    region = os.environ.get(
        "AWS_REGION", "us-east-1"
    )  # Default to a region if not specified
    endpoint_url = os.environ.get("AWS_BEDROCK_ENDPOINT")

    # Initialize ChatBedrock with the provided model ID and configuration
    llm = ChatBedrock(
        model_id=model_id,
        credentials_profile_name=credentials_profile_name,
        region=region,
        endpoint_url=endpoint_url,
        provider=provider,
        model_kwargs=model_kwargs,
    )
    return llm


def get_azure_llm(
    model_id: str = "gpt-4o",
    model_kwargs: Dict[str, Any] = None,
) -> AzureChatOpenAI:
    """
    Initialize an Azure OpenAI LLM with the specified configuration.

    Args:
        model_id: ID of the Azure OpenAI model to use
        model_kwargs: Additional model parameters

    Returns:
        Configured AzureChatOpenAI instance
    """
    if model_kwargs is None:
        model_kwargs = {"temperature": 0.0, "top_p": 1}

    model_version = os.environ.get("AZURE_OPENAI_MODEL_VERSION", None)

    # Initialize AzureChatOpenAI with the provided model ID and version
    llm = AzureChatOpenAI(
        deployment_name=model_id,
        model_version=model_version,
        temperature=model_kwargs.get("temperature", 0.0),
        top_p=model_kwargs.get("top_p", 1),
        timeout=600,
        max_retries=3,
    )
    return llm


def get_llm(
    config: Dict[str, Any],
    model_kwargs: Dict[str, Any] = {"temperature": 0.0, "top_p": 1},
    credentials_profile_name: str = "default",
    endpoint_url: str = "https://bedrock-runtime.us-east-1.amazonaws.com",
    region: str = "us-east-1",
) -> Union[ChatBedrock, AzureChatOpenAI]:
    """
    Get an LLM based on the configuration.

    Args:
        config: Configuration dictionary
        model_kwargs: Additional model parameters
        credentials_profile_name: AWS credentials profile (for Bedrock)
        endpoint_url: Bedrock endpoint URL (for Bedrock)
        region: AWS region (for Bedrock)

    Returns:
        Configured LLM instance (either Azure or Bedrock)
    """
    if config["model_source"].lower() == "azure":
        llm = get_azure_llm(model_id=config["model_id"], model_kwargs=model_kwargs)
    elif config["model_source"].lower() == "bedrock":
        llm = get_bedrock_llm(model_id=config["model_id"], model_kwargs=model_kwargs)
    else:
        raise ValueError(f"Unsupported model source: {config['model_source']}")
    return llm


def format_examples(
    few_shot_examples_df: Optional[pd.DataFrame], mix_examples: bool = False
) -> Optional[Dict[str, str]]:
    """
    Format examples for use in prompts.

    Args:
        few_shot_examples_df: DataFrame containing examples
        mix_examples: Whether to mix examples or group them by label

    Returns:
        Dictionary of formatted examples or None if input is None
    """
    if few_shot_examples_df is None:
        return None

    if mix_examples:
        few_shot_examples = {}
        few_shot_examples["mixed"] = "\n".join(
            [
                f"Example {idx + 1}: {row['text']} | Label: {row['label'].title()}"
                for idx, row in few_shot_examples_df.iterrows()
            ]
        )
    else:
        labels = few_shot_examples_df["label"].unique()
        few_shot_examples = {}
        for label in labels:
            relevant_examples = few_shot_examples_df[
                few_shot_examples_df["label"] == label
            ]
            few_shot_examples[label.title()] = "\n".join(
                [
                    f"Example {idx + 1}: {text}"
                    for idx, text in enumerate(relevant_examples["text"])
                ]
            )
    return few_shot_examples


def maybe_load_few_shot_examples(
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    mix_examples: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """
    Load few-shot examples from disk if they already exist.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        mix_examples: Whether to mix the examples or group them by label

    Returns:
        Tuple containing loaded few-shot examples DataFrame and formatted examples dict (or None if not found)
    """
    few_shot_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    few_shot_examples_file_path = os.path.join(
        config["output_dir"], f"{few_shot_prefix}.csv"
    )
    if os.path.exists(few_shot_examples_file_path):
        logger.info("Loading few-shot examples from %s", few_shot_examples_file_path)
        few_shot_examples_df = pd.read_csv(few_shot_examples_file_path)
        few_shot_examples = format_examples(few_shot_examples_df, mix_examples)
        return few_shot_examples_df, few_shot_examples
    else:
        if num_few_shot_examples > 0:
            logger.info(
                "Few-shot examples file not found: %s. Generating new examples.",
                few_shot_examples_file_path,
            )
        return None, None


def maybe_load_artifact(
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    artifact_type: str,
) -> Optional[Any]:
    """
    Load artifact (criteria or description) from disk if it already exists.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        artifact_type: Type of artifact ('criteria' or 'description')

    Returns:
        Loaded artifact or None if not found
    """
    prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    file_path = os.path.join(config["model_output_dir"], f"{prefix}.json")
    if os.path.exists(file_path):
        logger.info("Loading %s from %s", artifact_type, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        return artifact
    else:
        logger.info("%s file not found: %s", artifact_type.capitalize(), file_path)
        return None


def maybe_load_criteria(
    config: Dict[str, Any], method: str, num_few_shot_examples: int, iter_num: int
) -> Optional[Dict[str, str]]:
    """
    Load criteria from disk if they already exist.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number

    Returns:
        Loaded criteria dictionary or None if not found
    """
    return maybe_load_artifact(
        config, method, num_few_shot_examples, iter_num, artifact_type="criteria"
    )


def maybe_load_description(
    config: Dict[str, Any], method: str, num_few_shot_examples: int, iter_num: int
) -> Optional[str]:
    """
    Load description from disk if it already exists.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number

    Returns:
        Loaded description or None if not found
    """
    return maybe_load_artifact(
        config, method, num_few_shot_examples, iter_num, artifact_type="description"
    )


def sample_labels_by_distribution(
    dataset_df: pd.DataFrame, num_few_shot_examples: int, label_column: str
) -> pd.DataFrame:
    """
    Ensures sampling from all labels while maintaining the total number of few-shot examples.

    Args:
        dataset_df: The dataset containing examples
        num_few_shot_examples: The number of examples to sample
        label_column: The column name representing the label

    Returns:
        The sampled few-shot examples DataFrame
    """
    dataset_df_copy = dataset_df.sample(frac=1).copy()
    label_counts = dataset_df_copy[label_column].value_counts()
    non_zero_labels = label_counts[label_counts > 0]
    label_distribution = (
        non_zero_labels / non_zero_labels.sum()
        if non_zero_labels.sum() > 0
        else non_zero_labels
    )
    min_examples_per_label = 1
    remaining_examples = num_few_shot_examples - len(non_zero_labels)
    if remaining_examples < 0:
        sample_sizes = pd.Series(
            {label: (1) for label in non_zero_labels.index[:num_few_shot_examples]}
        )
    else:
        additional_samples = (
            (label_distribution * remaining_examples).round().astype(int)
        )
        sample_sizes = pd.Series(
            {
                label: (min_examples_per_label + additional)
                for label, additional in additional_samples.items()
            }
        )
        while sample_sizes.sum() != num_few_shot_examples:
            if sample_sizes.sum() < num_few_shot_examples:
                label_to_adjust = label_distribution.idxmax()
                sample_sizes[label_to_adjust] += 1
            else:
                label_to_adjust = sample_sizes.idxmax()
                sample_sizes[label_to_adjust] -= 1
    sampled_dfs = []
    for label, size in sample_sizes.items():
        group = dataset_df_copy[dataset_df_copy[label_column] == label]
        actual_size = min(len(group), size)
        sampled_dfs.append(group.sample(actual_size, replace=False))
    few_shot_example_df = pd.concat(sampled_dfs)
    return few_shot_example_df


def random_sampling(
    dataset_df: pd.DataFrame, num_few_shot_examples: int
) -> pd.DataFrame:
    """
    Randomly samples few-shot examples from the dataset.

    Args:
        dataset_df: The dataset containing examples
        num_few_shot_examples: The number of examples to sample

    Returns:
        The sampled few-shot examples DataFrame
    """
    if dataset_df.empty:
        return pd.DataFrame(columns=dataset_df.columns)
    if num_few_shot_examples <= 0:
        return pd.DataFrame(columns=dataset_df.columns)
    dataset_df_copy = dataset_df.sample(frac=1).copy()
    num_few_shot_examples = min(num_few_shot_examples, len(dataset_df_copy))
    sample_indices = random.sample(range(len(dataset_df_copy)), num_few_shot_examples)
    few_shot_example_df = dataset_df_copy.iloc[sample_indices]
    return few_shot_example_df


def create_few_shot_examples(
    dataset_df: pd.DataFrame,
    num_few_shot_examples: int = 5,
    mix_examples: bool = False,
    example_column: str = "text",
    label_column: str = "label",
    sampling_method: str = "random",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create few-shot examples from the dataset.

    Args:
        dataset_df: Dataset DataFrame
        num_few_shot_examples: Number of few-shot examples to sample
        mix_examples: Whether to mix the examples or group them by label
        example_column: Name of the column containing the examples
        label_column: Name of the column containing the labels
        sampling_method: Sampling method to use (random or distribution)

    Returns:
        Tuple containing the few-shot examples DataFrame and a dictionary of formatted examples
    """
    if num_few_shot_examples == 0:
        return None, None
    if sampling_method == "label_distribution":
        few_shot_example_df = sample_labels_by_distribution(
            dataset_df, num_few_shot_examples, label_column
        )
    else:
        few_shot_example_df = random_sampling(dataset_df, num_few_shot_examples)
    few_shot_example_df = few_shot_example_df[[example_column, "textual_label"]].rename(
        columns={"textual_label": label_column}
    )
    few_shot_example_df = few_shot_example_df.reset_index().rename(
        columns={example_column: "text", label_column: "label"}
    )
    few_shot_examples = format_examples(few_shot_example_df, mix_examples)
    return few_shot_example_df, few_shot_examples


def save_few_shot_examples(
    few_shot_examples_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save few-shot examples to disk and optionally to WandB.

    Args:
        few_shot_examples_df: DataFrame containing few-shot examples
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    if few_shot_examples_df is None or len(few_shot_examples_df) <= 0:
        return
    few_shot_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    few_shot_examples_file_path = os.path.join(
        config["output_dir"], f"{few_shot_prefix}.csv"
    )
    few_shot_examples_df.to_csv(few_shot_examples_file_path, index=False)
    logger.info("Few-shot examples saved to %s", few_shot_examples_file_path)
    few_shot_examples_model_file_path = os.path.join(
        config["model_output_dir"], f"{few_shot_prefix}.csv"
    )
    few_shot_examples_df.to_csv(few_shot_examples_model_file_path, index=False)
    logger.info("Few-shot examples saved to %s", few_shot_examples_model_file_path)
    if wandb_run is not None:
        few_shot_examples_table = wandb.Table(dataframe=few_shot_examples_df)
        few_shot_examples_table_name = f"{few_shot_prefix}_table"
        wandb_run.log({few_shot_examples_table_name: few_shot_examples_table})
        save_file_to_wandb(wandb_run, few_shot_examples_file_path, few_shot_prefix)


def save_file_to_wandb(
    wandb_run: Any,
    file_path: Union[str, Path],
    artifact_name: str,
    artifact_type: str = "dataset",
) -> None:
    """
    Save a file as a WandB artifact.

    Args:
        wandb_run: WandB run object
        file_path: Path to the file to save
        artifact_name: Name of the artifact
        artifact_type: Type of the artifact
    """
    if not os.path.exists(file_path):
        logger.warning("File not found for W&B logging: %s", file_path)
        return
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(file_path))
    wandb_run.log_artifact(artifact)


def fillna_to_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in a DataFrame based on column data types.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with NaN values filled
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=["float", "int"]).columns
    if not numeric_cols.empty:
        df_copy.loc[:, numeric_cols] = df_copy.loc[:, numeric_cols].fillna(0)
    string_cols = df_copy.select_dtypes(include=["object", "string"]).columns
    if not string_cols.empty:
        df_copy.loc[:, string_cols] = df_copy.loc[:, string_cols].fillna("")
    bool_cols = df_copy.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df_copy.loc[:, bool_cols] = df_copy.loc[:, bool_cols].fillna(False)
    return df_copy


def save_artifact(
    artifact: Any,
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    artifact_type: str,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save an artifact (criteria or description) to disk and optionally to WandB.

    Args:
        artifact: The artifact to save
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        artifact_type: Type of artifact ('criteria' or 'description')
        wandb_run: WandB run object or None
    """
    if artifact is None:
        return
    os.makedirs(config["model_output_dir"], exist_ok=True)
    artifact_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    artifact_file_path = os.path.join(
        config["model_output_dir"], f"{artifact_prefix}.json"
    )
    with open(artifact_file_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    logger.info("%s saved to %s", artifact_type.capitalize(), artifact_file_path)
    if wandb_run is not None:
        artifact_name = f"{artifact_prefix}_json"
        save_file_to_wandb(wandb_run, artifact_file_path, artifact_name)


def save_criteria(
    criteria: Optional[Dict[str, str]],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save criteria to disk and optionally to WandB.

    Args:
        criteria: Dictionary containing criteria
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    save_artifact(
        criteria,
        config,
        method,
        num_few_shot_examples,
        iter_num,
        artifact_type="criteria",
        wandb_run=wandb_run,
    )


def save_description(
    description: Optional[str],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save description to disk and optionally to WandB.

    Args:
        description: Description string
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    save_artifact(
        description,
        config,
        method,
        num_few_shot_examples,
        iter_num,
        artifact_type="description",
        wandb_run=wandb_run,
    )


def maybe_load_classification_report(
    config: Dict[str, Any], method: str
) -> Optional[Dict[str, Any]]:
    """
    Load a classification report if it exists and override is not specified.

    Args:
        config: Configuration dictionary
        method: Method name

    Returns:
        Classification report dictionary or None if it doesn't exist or override is True
    """
    if not config.get("override", False):
        classification_report_path = os.path.join(
            config["model_output_dir"], f"{method}_classification_report.json"
        )
        if os.path.exists(classification_report_path):
            logger.info(
                "Classification report already exists for %s. Skipping...", method
            )
            with open(classification_report_path, "r", encoding="utf-8") as f:
                classification_report_res = json.load(f)
            return classification_report_res
    return None


def get_classification_report(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
    labels: List[int] = [0, 1],
    target_names: List[str] = ["Negative", "Positive"],
) -> Dict[str, Any]:
    """
    Generate a classification report with metrics.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        labels: List of label integers
        target_names: List of label names

    Returns:
        Dictionary containing classification metrics
    """
    classification_report_res = classification_report(
        list(true_labels),
        list(predicted_labels),
        labels=labels,
        target_names=target_names,
        output_dict=True,
    )
    if "accuracy" not in classification_report_res:
        classification_report_res["accuracy"] = accuracy_score(
            true_labels, predicted_labels
        )
    if "micro avg" not in classification_report_res:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels, average="micro"
        )
        classification_report_res["micro avg"] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": len(true_labels),
        }
    return classification_report_res


async def run_experiment(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    classifier: ClassifierBase,
    samples_df: pd.DataFrame,
    method: str,
    few_shot_examples: Optional[Dict[str, str]] = None,
    criteria: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a classification experiment and collect results.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        classifier: Classifier instance
        samples_df: DataFrame containing test samples
        method: Method name
        few_shot_examples: Dictionary of few-shot examples or None
        criteria: Dictionary of criteria or None
        description: Description string or None

    Returns:
        Classification report dictionary
    """
    tic = time()
    responses = await classifier.get_llm_responses(
        samples_df,
        few_shot_examples=few_shot_examples,
        criteria=criteria,
        description=description,
    )
    responses_df = pd.DataFrame(responses).set_index("index")
    toc = time()
    logger.info("Time for method %s: %s seconds.", method, toc - tic)
    samples_w_responses_df = pd.merge(
        samples_df, responses_df, left_index=True, right_index=True
    )
    samples_w_responses_df = fillna_to_all_types(samples_w_responses_df)
    experiment_file_name = f"{method}_samples_w_responses_df.csv"
    samples_w_responses_path = os.path.join(
        config["model_output_dir"], experiment_file_name
    )
    os.makedirs(config["model_output_dir"], exist_ok=True)
    samples_w_responses_df.to_csv(samples_w_responses_path, index=False)
    logger.info(
        "Samples with responses dataframe saved to %s", samples_w_responses_path
    )
    classification_report_res = get_classification_report(
        samples_w_responses_df["label"],
        samples_w_responses_df["predicted_label"],
        labels=classifier.labels,
        target_names=classifier.target_names,
    )
    classification_report_path = os.path.join(
        config["model_output_dir"], f"{method}_classification_report.json"
    )
    with open(classification_report_path, "w") as f:
        json.dump(classification_report_res, f, indent=2)
        logger.info("Classification report saved to %s", classification_report_path)
    if wandb_run is not None:
        experiment_classification_report_name = f"{method}_classification_report"
        wandb_run.log(
            {
                experiment_classification_report_name: classification_report_res,
                "time": toc - tic,
            }
        )
        dataset_with_responses_table = wandb.Table(dataframe=samples_w_responses_df)
        experiment_table_name = f"{method}_samples_w_responses_table"
        wandb_run.log({experiment_table_name: dataset_with_responses_table})
        save_file_to_wandb(wandb_run, samples_w_responses_path, experiment_table_name)
    return classification_report_res


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator for joining keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_average_metrics(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    reports: List[Dict[str, Any]],
    method: str,
    target_names: List[str] = ["Negative", "Positive"],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate and log average metrics across multiple reports.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        reports: List of classification reports
        method: Method name
        target_names: List of label names

    Returns:
        Tuple containing dictionaries of mean and standard deviation metrics
    """
    aggregated_metrics = {
        target_name: {"precision": [], "recall": [], "f1-score": [], "support": []}
        for target_name in target_names
    }
    aggregated_metrics.update(
        {
            "accuracy": [],
            "macro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
            "micro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
            "weighted avg": {
                "precision": [],
                "recall": [],
                "f1-score": [],
                "support": [],
            },
        }
    )
    for report in reports:
        aggregated_metrics["accuracy"].append(report["accuracy"])
        all_labels = target_names + ["macro avg", "micro avg", "weighted avg"]
        for label in all_labels:
            aggregated_metrics[label]["precision"].append(report[label]["precision"])
            aggregated_metrics[label]["recall"].append(report[label]["recall"])
            aggregated_metrics[label]["f1-score"].append(report[label]["f1-score"])
            aggregated_metrics[label]["support"].append(report[label]["support"])
    mean_report = {}
    std_report = {}
    for label, metrics in aggregated_metrics.items():
        if label == "accuracy":
            mean_report[label] = sum(metrics) / len(metrics) if len(metrics) > 0 else 0
            std_report[label] = pd.Series(metrics).std()
        else:
            mean_report[label] = {
                "precision": (
                    sum(metrics["precision"]) / len(metrics["precision"])
                    if len(metrics["precision"]) > 0
                    else 0
                ),
                "recall": (
                    sum(metrics["recall"]) / len(metrics["recall"])
                    if len(metrics["recall"]) > 0
                    else 0
                ),
                "f1-score": (
                    sum(metrics["f1-score"]) / len(metrics["f1-score"])
                    if len(metrics["f1-score"]) > 0
                    else 0
                ),
                "support": (
                    sum(metrics["support"]) / len(metrics["support"])
                    if len(metrics["support"]) > 0
                    else 0
                ),
            }
            std_report[label] = {
                "precision": pd.Series(metrics["precision"]).std(),
                "recall": pd.Series(metrics["recall"]).std(),
                "f1-score": pd.Series(metrics["f1-score"]).std(),
                "support": pd.Series(metrics["support"]).std(),
            }
    mean_file_path = os.path.join(
        config["model_output_dir"], f"{method}_average_metrics.json"
    )
    os.makedirs(config["model_output_dir"], exist_ok=True)
    with open(mean_file_path, "w") as f:
        json.dump(mean_report, f, indent=2)
    std_file_path = os.path.join(
        config["model_output_dir"], f"{method}_std_metrics.json"
    )
    with open(std_file_path, "w") as f:
        json.dump(std_report, f, indent=2)
    logger.info("Average %s report: %s", method, json.dumps(mean_report, indent=2))
    logger.info("Std %s report: %s", method, json.dumps(std_report, indent=2))
    if wandb_run is not None:
        wandb_run.log({f"{method}_mean": mean_report, f"{method}_std": std_report})
        flatten_mean_dict = flatten_dict(mean_report)
        for key, value in flatten_mean_dict.items():
            wandb_run.summary[f"mean_{key}"] = value
        flatten_std_dict = flatten_dict(std_report)
        for key, value in flatten_std_dict.items():
            wandb_run.summary[f"std_{key}"] = value
        save_file_to_wandb(wandb_run, mean_file_path, f"{method}_mean_json")
        save_file_to_wandb(wandb_run, std_file_path, f"{method}_std_json")
    return mean_report, std_report


def compute_average_metrics_dict(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    reports: Dict[str, Dict[int, List[Dict[str, Any]]]],
    target_names: List[str] = ["Negative", "Positive"],
) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Compute average metrics for all methods and few-shot configurations.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        reports: Nested dictionary containing classification reports
        target_names: List of label names

    Returns:
        Nested dictionary containing average metrics
    """
    average_metrics_dict = {}
    for method, method_reports in reports.items():
        average_metrics_dict[method] = {}
        for num_few_shot_examples, report in method_reports.items():
            mean_report, std_report = log_average_metrics(
                config,
                wandb_run,
                report,
                f"{method}_{num_few_shot_examples}",
                target_names=target_names,
            )
            average_metrics_dict[method][num_few_shot_examples] = {
                "mean": mean_report,
                "std": std_report,
            }
    return average_metrics_dict


def plot_performance(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    methods: List[str],
    average_metrics_dict: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    target_names: List[str] = ["Negative", "Positive"],
) -> None:
    """
    Plot performance metrics across different methods and few-shot configurations.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        methods: List of method names
        average_metrics_dict: Nested dictionary containing average metrics
        target_names: List of label names
    """
    categories = target_names + ["macro avg", "micro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score"]
    for category in categories:
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            markers = ["^", "*", "o", "s", "D", "v", "p"]
            for i, method in enumerate(methods):
                means = []
                stds = []
                for num_few_shot_examples in config["num_few_shot_examples"]:
                    mean_value = average_metrics_dict[method][num_few_shot_examples][
                        "mean"
                    ][category][metric]
                    std_value = average_metrics_dict[method][num_few_shot_examples][
                        "std"
                    ][category][metric]
                    means.append(mean_value)
                    stds.append(std_value)
                plt.errorbar(
                    config["num_few_shot_examples"],
                    means,
                    yerr=stds,
                    label=method,
                    marker=markers[i % len(markers)],
                    capsize=5,
                )
            plt.title(f"{category.title()} - {metric.title()}", fontsize=14)
            plt.xlabel("Number of Few-Shots", fontsize=12)
            plt.ylabel(f"{metric.title()}", fontsize=12)
            plt.legend(loc="best")
            plt.grid(True)
            os.makedirs(config["model_output_dir"], exist_ok=True)
            for ext in ["pdf", "png"]:
                plot_filename = f"{category.replace(' ', '_').lower()}_{metric}.{ext}"
                plot_path = os.path.join(config["model_output_dir"], plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            if wandb_run is not None:
                save_file_to_wandb(
                    wandb_run,
                    plot_path,
                    f"{category.replace(' ', '_').lower()}_{metric}_plot",
                )
            plt.close()


async def main(config: Dict):
    """
    Main function to run experiments.

    Args:
        config: Configuration dictionary
    """
    wandb_run = None
    if config.get("use_wandb", False):
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run"],
            config=config,
        )
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
    if config["model_id"].startswith("arn:"):
        config["model_output_dir"] = os.path.join(
            config["output_dir"], config["model_id"].split("/")[-1]
        )
    else:
        config["model_output_dir"] = os.path.join(
            config["output_dir"], config["model_id"]
        )
    if not os.path.exists(config["model_output_dir"]):
        os.makedirs(config["model_output_dir"], exist_ok=True)
    setup_logger(config["model_output_dir"])
    logger.info("Starting experiment with config: %s", json.dumps(config, indent=2))
    dataset_loader = DatasetLoader(config)
    logger.info("Loaded dataset loader: %s", dataset_loader)
    train_df, test_df = dataset_loader.load_datasets()

    logger.info(
        "Loaded train and test datasets: train_df: %s, test_df: %s",
        train_df.shape,
        test_df.shape,
    )
    target_names = list(map(lambda label: label.title(), config["label_map"].values()))
    logger.info("Target names: %s", target_names)
    llm = get_llm(config)
    if config["model_source"] == "azure":
        logger.info("Loaded Azure LLM: %s", llm.deployment_name)
    else:
        logger.info("Loaded Bedrock LLM: %s", llm.model_id)
    if len(config["label_map"]) <= 2:
        classifier = BinaryClassifier(
            llm=llm,
            objective=config["objective"],
            label_map=config["label_map"],
            target_names=target_names,
            text_column=config["text_column"],
            batch_size=config["batch_size"],
        )
        criteria_creator = BinaryCriteriaCreator(llm=llm, objective=config["objective"])
        description_creator = BinaryDescriptionCreator(
            llm=llm, objective=config["objective"]
        )
    else:
        raise ValueError(
            "Only binary classification is supported in this version. "
            "Please update the config to use a binary label map."
        )

    methods = [
        "Examples",
        "Criteria-Ex",
        "Criteria-De",
        "Description-Ex",
        "Description-Cr",
    ]
    config["num_few_shot_examples"] = sorted(
        [int(num_few_shots) for num_few_shots in config["num_few_shot_examples"]]
    )
    if 0 not in config["num_few_shot_examples"]:
        config["num_few_shot_examples"].insert(0, 0)
    reports = init_reports_dict(methods, config["num_few_shot_examples"])
    for iter_num in range(1, config["num_experiments"] + 1):
        logger.info(
            "Running experiment iteration %s/%s...", iter_num, config["num_experiments"]
        )
        for num_few_shot_examples in config["num_few_shot_examples"]:
            logger.info(
                "Iteration %s - Few-shot examples: %s", iter_num, num_few_shot_examples
            )

            method = "Examples"
            few_shot_examples_df, few_shot_examples = maybe_load_few_shot_examples(
                config,
                method="Examples",
                num_few_shot_examples=num_few_shot_examples,
                iter_num=iter_num,
                mix_examples=config["mix_examples"],
            )
            if few_shot_examples_df is None:
                if num_few_shot_examples > 0:
                    logger.info(
                        "Few-shot examples not found for %s examples. Generating new examples...",
                        num_few_shot_examples,
                    )
                few_shot_examples_df, few_shot_examples = create_few_shot_examples(
                    train_df,
                    num_few_shot_examples=num_few_shot_examples,
                    mix_examples=config["mix_examples"],
                    example_column=config["example_column"],
                    label_column=config["label_column"],
                    sampling_method=config["sampling_method"],
                )
                if num_few_shot_examples > 0:
                    save_few_shot_examples(
                        few_shot_examples_df,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info(
                        "Few-shot examples saved for %s examples.",
                        num_few_shot_examples,
                    )
            samples_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if samples_report is None:
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                samples_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    few_shot_examples=few_shot_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(samples_report)
            method = "Criteria-Ex"
            criteria_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            criteria_from_examples = maybe_load_criteria(
                config, method, num_few_shot_examples, iter_num
            )
            if criteria_report is None:
                if criteria_from_examples is None:
                    logger.info(
                        "Criteria not found for %s. Generating new criteria...", method
                    )
                    criteria_from_examples = await criteria_creator.get_llm_responses(
                        few_shot_examples=few_shot_examples
                    )
                    save_criteria(
                        criteria_from_examples,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info("Generated and saved criteria for %s", method)
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                criteria_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    criteria=criteria_from_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(criteria_report)
            method = "Description-Ex"
            description_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            description_from_examples = maybe_load_description(
                config, method, num_few_shot_examples, iter_num
            )
            if description_report is None:
                if description_from_examples is None:
                    logger.info(
                        "Description not found for %s. Generating new description...",
                        method,
                    )
                    description_from_examples = (
                        await description_creator.get_llm_responses(
                            few_shot_examples=few_shot_examples
                        )
                    )
                    save_description(
                        description_from_examples,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info("Generated and saved description for %s", method)
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                description_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    description=description_from_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(description_report)
            method = "Criteria-De"
            criteria_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if num_few_shot_examples == 0:
                criteria_from_description = criteria_from_examples
                criteria_report = reports["Criteria-Ex"][int(num_few_shot_examples)][
                    iter_num - 1
                ]
                save_criteria(
                    criteria_from_description,
                    config,
                    method,
                    num_few_shot_examples,
                    iter_num,
                    wandb_run,
                )
                logger.info(
                    "Using and saving zero-shot criteria for %s from Criteria-Ex",
                    method,
                )
            else:
                criteria_from_description = maybe_load_criteria(
                    config, method, num_few_shot_examples, iter_num
                )
                if criteria_report is None:
                    if criteria_from_description is None:
                        logger.info(
                            "Criteria not found for %s. Generating from description...",
                            method,
                        )
                        criteria_from_description = (
                            await criteria_creator.get_llm_responses(
                                description=description_from_examples
                            )
                        )
                        save_criteria(
                            criteria_from_description,
                            config,
                            method,
                            num_few_shot_examples,
                            iter_num,
                            wandb_run,
                        )
                        logger.info("Generated and saved criteria for %s", method)
                    logger.info(
                        "Classification report not found for %s. Running experiment...",
                        method,
                    )
                    criteria_report = await run_experiment(
                        config,
                        wandb_run,
                        classifier,
                        test_df,
                        criteria=criteria_from_description,
                        method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                    )
                logger.info(
                    "Finished running experiment for %s; with %s examples. (iter %s)",
                    method,
                    num_few_shot_examples,
                    iter_num,
                )
            reports[method][int(num_few_shot_examples)].append(criteria_report)
            method = "Description-Cr"
            description_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if num_few_shot_examples == 0:
                description_from_criteria = description_from_examples
                description_report = reports["Description-Ex"][
                    int(num_few_shot_examples)
                ][iter_num - 1]
                save_description(
                    description_from_criteria,
                    config,
                    method,
                    num_few_shot_examples,
                    iter_num,
                    wandb_run,
                )
                logger.info(
                    "Using and saving zero-shot description for %s from Description-Ex",
                    method,
                )
            else:
                description_from_criteria = maybe_load_description(
                    config, method, num_few_shot_examples, iter_num
                )
                if description_report is None:
                    if description_from_criteria is None:
                        logger.info(
                            "Description not found for %s. Generating from criteria...",
                            method,
                        )
                        description_from_criteria = (
                            await description_creator.get_llm_responses(
                                criteria=criteria_from_examples
                            )
                        )
                        save_description(
                            description_from_criteria,
                            config,
                            method,
                            num_few_shot_examples,
                            iter_num,
                            wandb_run,
                        )
                        logger.info("Generated and saved description for %s", method)
                    logger.info(
                        "Classification report not found for %s. Running experiment...",
                        method,
                    )
                    description_report = await run_experiment(
                        config,
                        wandb_run,
                        classifier,
                        test_df,
                        description=description_from_criteria,
                        method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                    )
                logger.info(
                    "Finished running experiment for %s; with %s examples. (iter %s)",
                    method,
                    num_few_shot_examples,
                    iter_num,
                )
            reports[method][int(num_few_shot_examples)].append(description_report)
    logger.info("Computing average metrics across all iterations...")
    average_metrics_dict = compute_average_metrics_dict(
        config, wandb_run, reports, target_names=target_names
    )
    logger.info("Creating performance visualizations...")
    plot_performance(
        config, wandb_run, methods, average_metrics_dict, target_names=target_names
    )
    logger.info("Experiment completed successfully")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = vars(args)
        config.pop("config", None)
        config.pop("env_file", None)
    config = process_label_map(config)
    asyncio.run(main(config))
