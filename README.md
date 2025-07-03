# From Examples to Knowledge: Enhancing In-Context Learning for Long-Context Classification

This repository contains code for the paper **"From Examples to Knowledge: Enhancing In-Context Learning for Long-Context Classification."** This framework explores how various prompting techniques influence classification accuracy in B2B conversation transcripts, with a focus on practical efficiency when using LLMs. In particular, it examines how to generate criteria and descriptions from few-shot examples, and then use those for in-context classification.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Datasets](#datasets)
  - [Dataset Structure](#dataset-structure)
  - [Dataset Format](#dataset-format)
  - [Data Privacy and Anonymization](#data-privacy-and-anonymization)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Configuration File](#configuration-file)
  - [Command Line Arguments](#command-line-arguments)
- [Project Structure](#project-structure)
- [Expected Outputs and Visualization](#expected-outputs-and-visualization)
  - [Output Directory Structure](#output-directory-structure)
  - [Visualization Outputs](#visualization-outputs)
- [Integrations](#integrations)
  - [Weights & Biases](#weights--biases)
- [License](#license)
- [Citation](#citation)

---

## Overview

The framework evaluates different methods for instructing LLMs to perform classification tasks:

- **Examples**: Direct few-shot learning by showing examples  
- **Criteria-Ex**: Generate classification criteria based on examples, then classify using criteria  
- **Criteria-De**: Generate criteria from task descriptions, then classify using criteria  
- **Description-Ex**: Generate task descriptions from examples, then classify using descriptions  
- **Description-Cr**: Generate descriptions from criteria, then classify using descriptions  

Each method is evaluated across different configurations of few-shot examples to understand how the amount of demonstration data affects performance.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/criteria-description-in-context-learning.git
cd criteria-description-in-context-learning

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root directory with your API credentials:

```bash
# For Azure OpenAI
AZURE_OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "your_api_version"
AZURE_OPENAI_MODEL_VERSION = "your_model_version"
AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your_azure_api_key"

# For AWS Bedrock
AWS_PROFILE = "your_profile"
AWS_REGION = "your_region"
AWS_BEDROCK_ENDPOINT= "https://your-resource-name.amazonaws.com"
```

## Datasets

We provide 5 datasets for B2B binary classification tasks extracted from real customer conversations:

- **Business Goals**: Classification of statements as business goals or not
- **Decision Criteria**: Identification of decision criteria in customer conversations
- **Decision Makers**: Recognition of decision maker mentions in conversations
- **Decision Making Process**: Classification of statements describing decision-making processes
- **Pain Points**: Identification of customer pain points in conversations

These datasets can be obtained for research purposes under a data usage agreement at:
[https://github.com/yourusername/b2b-classification-datasets](https://github.com/yourusername/b2b-classification-datasets)

### Dataset Structure

```
data/
├── business_goals/
│   ├── train.csv
│   └── test.csv
├── decision_criteria/
│   ├── train.csv
│   └── test.csv
...
```

### Dataset Format

The framework expects datasets in CSV format with at least:

- A text column (specified by `--text_column`)
- A label column (specified by `--label_column`)

Example:

```
id,text,label
1,"[PROSPECT_A] We aim to reduce our customer acquisition costs by 15% this year. [SELLER_A] Our product is exactly what you need.",1
2,"[SELLER_A] Let me check my calendar for our next meeting. [PROSPECT_A] Sure, no problem.",0
3,"[SELLER_A] What is your main objective? [PROSPECT_A] Our main objective is to expand into European markets by Q4.",1
```

### Data Privacy and Anonymization
To protect sensitive business information in the provided datasets, all personal and organizational data has been anonymized. We have systematically replaced real entities with fictional alternatives while preserving the conversational context and linguistic patterns essential for classification tasks.
The replacements.json file contains the complete set of replacement entities used in the anonymization process, organized by entity type:

- ORGANIZATION (120+ entries): Fictional company names like "Quantum Solutions", "Zenith Innovations", "Nebula Technologies"
- PERSON (130+ entries): Gender-neutral names such as "Alex", "Jordan", "Casey", "Taylor"
- PRODUCT (80+ entries): Fictional product/service names like "CodeCraft", "QuantaQuery", "NebulaNet"
- LOCATION (180+ entries): Fictional countries, cities, and places including "Varthevia", "Brindmere", "Corswick"
- OTHER (35+ entries): Languages and miscellaneous categories like "English", "Spanish", "French"
- URL (15+ entries): Anonymized website formats such as "sparkle magic dot com", "tech wizard dot net"
- ID (1 entry): Generic ID placeholder format
- PHONE (1 entry): Generic phone number placeholder format
- EMAIL (4 entries): Anonymized email formats like "john.doe at example dot com"

## Usage

### Command Line

```bash
python run_classification.py \
  --model_id gpt-4o \
  --model_source azure \
  --objective "the prospect discusses their business goals in the context of the purchasing process." \
  --data_dir ./data/business_goals \
  --output_dir ./results \
  --num_few_shot_examples 0 10 25 50 75 100 \
  --num_experiments 5 \
  --batch_size 10 \
  --label_map '{"0": "Negative", "1": "Positive"}' \
  --env_file .env
```

### Configuration File

Create a JSON configuration file:

```json
{
    "model_id": "gpt-4o",
    "model_source": "azure",
    "objective": "the prospect discusses their business goals in the context of the purchasing process",
    "data_dir": "./data/business_goals",
    "num_few_shot_examples": [
        0,
        10,
        25,
        50,
        75,
        100
    ],
    "num_experiments": 5,
    "batch_size": 10,
    "output_dir": "./results/business_goals",
    "use_wandb": true,
    "override": false,
    "mix_examples": false,
    "sampling_method": "label_distribution",
    "label_column": "label",
    "text_column": "text",
    "example_column": "text",
    "label_map": {
      "0": "Negative",
      "1": "Positive"
    },
    "wandb_entity": "nlp",
    "wandb_project": "b2b-classification-research",
    "wandb_run": "experiment-business_goals-gpt4o"
}
```

Then run:

```bash
python run_classification.py --config config.json --env_file .env
```

### Command Line Arguments

| Argument                  | Description |
|---------------------------|-------------|
| `--model_id`              | ID of the model to use. |
| `--model_source`          | Source of the model: azure, bedrock. |
| `--objective`             | The user intent: a short instruction of the classification task. |
| `--data_dir`              | Directory containing the dataset. |
| `--batch_size`            | (Optional) Batch size for model inference (default: 10). |
| `--num_few_shot_examples` | (Optional) List of few-shot example counts to use (default: `[0, 10, 25, 50, 75, 100]`). |
| `--num_experiments`       | (Optional) Number of experiments to run for statistical significance (default: 5). |
| `--mix_examples`          | (Optional) Whether to mix examples across classes. |
| `--sampling_method`       | (Optional) Sampling method: `random` or `label_distribution` (default: `label_distribution`). |
| `--output_dir`            | Directory to save the outputs. |
| `--use_wandb`             | (Optional) Enable logging with Weights & Biases. |
| `--override`              | (Optional) Override existing outputs. |
| `--label_column`          | (Optional) Name of the label column (default: `label`). |
| `--text_column`           | (Optional) Name of the text column (default: `text`). |
| `--example_column`        | (Optional) Name of the example column (default: `text`). |
| `--label_map`             | (Optional) Label map as a JSON string (default: `'{"0": "Negative", "1": "Positive"}'`). |
| `--wandb_entity`          | (Optional) Weights & Biases entity name (default: `nlp`). |
| `--wandb_project`         | (Optional) Weights & Biases project name (default: `criteria-classification-research`). |
| `--wandb_run`             | (Optional) Weights & Biases run name (default: `criteria-experiment`). |
| `--config`                | (Optional) Path to the configuration JSON file. |
| `--env_file`              | (Optional) Path to the .env file (default: `.env`). |

## Project Structure

```
.
├── data/                       # Data Folder
├── results/                    # Results Folder
├── run_classification.py       # Entry point for running classification experiments
├── dataset_loader.py           # Loads and prepares datasets
├── classifiers.py              # Runs classification based on different prompting methods
├── criteria_creator.py         # Generates classification criteria from examples or descriptions
├── description_creator.py      # Generates classification description from examples or criteria
├── replacements.json           # Entity replacement mappings used for data anonymization
├── config.json                 # Example configuration file
├── .env                        # Environment file with API credentials
├── requirements.txt            # Python dependencies
├── LICENSE                     # Project license
└── README.md                   # Project documentation and usage guide
```

## Expected Outputs and Visualization

### Output Directory Structure
```
output_dir/
└── model_id/
├── few_shot_examples/
│   ├── Examples_<n>iter<i>.csv       # Few-shot examples for each configuration
│   └── Examples_<n>iter<i>.json      # Examples experiment results
│
├── criteria_generation/
│   ├── Criteria-Ex_<n>iter<i>.json   # Criteria generated from examples
│   └── Criteria-De_<n>iter<i>.json   # Criteria generated from descriptions
│
├── description_generation/
│   ├── Description-Ex_<n>iter<i>.json # Descriptions generated from examples
│   └── Description-Cr_<n>iter<i>.json # Descriptions generated from criteria
│
├── prediction_results/
│   └── *_samples_w_responses_df.csv    # Full prediction results for each method
│
├── metrics/
│   ├── classification_report.json    # Detailed metrics for each method
│   ├── average_metrics.json          # Mean metrics across iterations
│   └── std_metrics.json              # Standard deviation of metrics
│
├── visualizations/
│   ├── macro_avg.png                 # Performance plots for macro average metrics
│   ├── micro_avg.png                 # Performance plots for micro average metrics
│   └── weighted_avg.png              # Performance plots for weighted average metrics
│
└── experiment.log                      # Detailed experiment log
```
Where `<n>` is the number of few-shot examples and `<i>` is the iteration number.

### Visualization Outputs
The framework automatically generates performance visualization plots:

Few-shot Learning Curves: These plots show how performance metrics change as the number of few-shot examples increases. For each method (Examples, Criteria-Ex, etc.), we include:

- X-axis: Number of few-shot examples (0, 10, 25, 50, 75, 100)
- Y-axis: Performance metric (precision, recall, f1-score)
- Error bars: Standard deviation across iterations


Metrics Visualized:

- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were identified correctly
- F1-Score: Harmonic mean of precision and recall


Categories of Metrics:

- Per-class metrics: Individual metrics for each class (e.g., "Positive", "Negative")
- Macro Average: Average of metrics computed for each label (treats all classes equally)
- Micro Average: Aggregates contributions of all classes (favors larger classes)
- Weighted Average: Weighted by support (number of true instances for each label)

## Integrations

### Weights & Biases

The framework supports logging metrics to [Weights & Biases](https://wandb.ai/):

- Enable with `--use_wandb`
- Requires `wandb_entity`, `wandb_project`, and optionally `wandb_run`

## License

This code is licensed under the [MIT](LICENSE).

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{llm_classification_2025,
  author = {Anonimyzed},
  title = {From Examples to Knowledge: Enhancing In-Context Learning for Long-Context Classification},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/criteria-description-in-context}}
}
```
