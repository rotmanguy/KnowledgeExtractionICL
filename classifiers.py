# MIT License
# Copyright (c) [2025] [Anonymized]
# See LICENSE file for full license text

from abc import ABC, abstractmethod
import asyncio
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Union


class ClassifierBase(ABC):
    """
    Abstract base class for text classifiers that use language models for classification.

    This class provides the base structure for creating classifiers that can:
    1. Process text inputs using language models
    2. Handle both zero-shot and few-shot classification
    3. Support criteria-based and description-based classification
    4. Process inputs in batches with retry mechanisms
    """

    def __init__(
        self,
        llm: Union[ChatBedrock, AzureChatOpenAI],
        objective: str,
        batch_size: int = 10,
        text_column: str = "text",
    ) -> None:
        """
        Initialize the classifier with model and processing parameters.

        Args:
            llm: Language model instance (BedrockChat or AzureChat)
            objective: Classification objective or task description
            batch_size: Number of inputs to process in a single batch
            text_column: Name of the column containing text data in the dataset
        """
        self.llm = llm
        self.objective = objective
        self.batch_size = batch_size
        self.text_column = text_column
        self.system_prompt: Optional[str] = None
        self.ai_prompt: Optional[str] = None

    def get_system_prompt(self) -> str:
        """
        Generate the system prompt for multi-class classification.

        Returns:
            System prompt string
        """
        return f"""<description>
You are a knowledgeable annotator with experience in classifying diverse types of textual inputs.
</description>"""

    def get_human_prompt(
        self,
        text_snippet: str,
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Generate a human prompt based on the classification approach.

        Args:
            text_snippet: Text to classify
            few_shot_examples: Few-shot examples organized by class
            criteria: Classification criteria for each class
            description: Detailed description of the classification task

        Returns:
            Human prompt string
        """
        if few_shot_examples is not None:
            return self.create_human_prompt_few_shot(text_snippet, few_shot_examples)
        elif criteria is not None:
            return self.create_human_prompt_criteria(text_snippet, criteria)
        elif description is not None:
            return self.create_human_prompt_description(text_snippet, description)
        else:
            return self.create_human_prompt_zero_shot(text_snippet)

    def get_ai_prompt(self) -> str:
        """
        Generate the AI prompt for multi-class classification.

        Returns:
            AI prompt string
        """
        return f"<rationale>"

    def create_inputs(
        self,
        text_snippets: List[str],
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> List[List[Union[SystemMessage, HumanMessage, AIMessage]]]:
        """
        Create input prompts for the language model based on the classification approach.

        Args:
            text_snippets: List of texts to classify
            few_shot_examples: Examples for few-shot learning, organized by class
            criteria: Classification criteria for each class
            description: Detailed description of the classification task

        Returns:
            List of message sequences for the language model
        """
        inputs = [
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.get_human_prompt(
                        text_snippet,
                        few_shot_examples=few_shot_examples,
                        criteria=criteria,
                        description=description,
                    )
                ),
                AIMessage(content=self.ai_prompt),
            ]
            for text_snippet in text_snippets
        ]
        return inputs

    async def process_batches(
        self, inputs: List[List[Union[SystemMessage, HumanMessage, AIMessage]]]
    ) -> List[str]:
        """
        Process batches of inputs using the LLM with simple retry mechanism.

        This method handles throttling errors by retrying batches with a fixed delay.

        Args:
            inputs: List of message sequences to process

        Returns:
            List of model responses as strings

        Raises:
            RuntimeError: If batch processing fails after maximum retries
        """
        responses = []
        len_inputs = len(inputs)
        max_retries = 3
        for start in range(0, len_inputs, self.batch_size):
            inputs_batch = inputs[start : start + self.batch_size]
            retries = 0
            while retries < max_retries:
                try:
                    responses_batch = await self.llm.abatch(
                        inputs_batch, config={"max_concurrency": self.batch_size // 2}
                    )
                    responses.extend(responses_batch)
                    break
                except Exception as e:
                    if "ThrottlingException" in str(e) and retries < max_retries - 1:
                        delay = 60
                        print(
                            f"Throttling detected. Retrying batch in {delay} seconds..."
                        )
                        await asyncio.sleep(delay)
                        retries += 1
                    else:
                        print(f"Batch processing failed after {retries} retries.")
                        raise RuntimeError(
                            f"Batch starting at index {start} failed: {str(e)}"
                        )
        responses = [response.content for response in responses]
        if len(responses) != len_inputs:
            raise RuntimeError(
                f"Mismatch in responses: Expected {len_inputs}, but got {len(responses)}."
            )
        return responses

    @abstractmethod
    def create_few_shot_example_str(self, examples: Dict[str, str]) -> str:
        """
        Format few-shot examples for inclusion in the prompt.

        Args:
            examples: Dictionary of examples organized by class

        Returns:
            Formatted examples string
        """
        pass

    @abstractmethod
    def create_criteria_str(self, criteria: Dict[str, str]) -> str:
        """
        Format classification criteria for inclusion in the prompt.

        Args:
            criteria: Dictionary of criteria organized by class

        Returns:
            Formatted criteria string
        """
        pass

    @abstractmethod
    def create_description_str(self, description: str) -> str:
        """
        Format task description for inclusion in the prompt.

        Args:
            description: Detailed description of the classification task

        Returns:
            Formatted description string
        """
        pass

    @abstractmethod
    def create_human_prompt_zero_shot(self, text_snippet: str) -> str:
        """
        Create a zero-shot prompt for classification.

        Args:
            text_snippet: Text to classify

        Returns:
            Zero-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_few_shot(
        self, text_snippet: str, few_shot_examples: Dict[str, str]
    ) -> str:
        """
        Create a few-shot prompt for classification using examples.

        Args:
            text_snippet: Text to classify
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_criteria(
        self, text_snippet: str, criteria: Dict[str, str]
    ) -> str:
        """
        Create a criteria-based prompt for classification.

        Args:
            text_snippet: Text to classify
            criteria: Dictionary of criteria organized by class

        Returns:
            Criteria-based prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_description(
        self, text_snippet: str, description: str
    ) -> str:
        """
        Create a description-based prompt for classification.

        Args:
            text_snippet: Text to classify
            description: Detailed description of the classification task

        Returns:
            Description-based prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract classification information.

        Args:
            response: Raw response from the language model

        Returns:
            Dictionary with parsed response data
        """
        pass

    @abstractmethod
    def process_parsed_responses(
        self, parsed_responses: List[Dict[str, str]]
    ) -> Dict[str, List]:
        """
        Process a list of parsed responses to extract structured information.

        Args:
            parsed_responses: List of dictionaries containing parsed responses

        Returns:
            Dictionary containing classification results and metadata
        """
        pass

    async def get_llm_responses(
        self,
        dataset: pd.DataFrame,
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get language model responses for a dataset of text snippets.

        Args:
            dataset: Dataset containing text to classify
            few_shot_examples: Few-shot examples organized by class
            criteria: Classification criteria for each class
            description: Detailed description of the classification task

        Returns:
            Dictionary containing classification results and metadata

        Raises:
            ValueError: If multiple classification approaches are provided simultaneously
        """
        if sum(x is not None for x in [few_shot_examples, criteria, description]) > 1:
            raise ValueError(
                "Only one of few_shot_examples, criteria, or description should be provided."
            )
        texts = [text for text in dataset[self.text_column]]
        inputs = self.create_inputs(
            texts,
            few_shot_examples=few_shot_examples,
            criteria=criteria,
            description=description,
        )
        responses = await self.process_batches(inputs)
        final_responses = {}
        final_responses["index"] = list(dataset.index)
        final_responses["system_prompt"] = [i[0].content for i in inputs]
        final_responses["human_prompt"] = [i[1].content for i in inputs]
        final_responses["ai_prompt"] = [i[2].content for i in inputs]
        final_responses["raw_response"] = responses
        parsed_responses = [self.parse_response(response) for response in responses]
        parsed_responses = self.process_parsed_responses(parsed_responses)
        final_responses.update(parsed_responses)
        return final_responses


class BinaryClassifier(ClassifierBase):
    """
    Classifier for binary classification tasks using language models.

    This class specializes the base classifier for binary (positive/negative) classification
    with support for few-shot examples, criteria-based, and description-based approaches.
    """

    def __init__(
        self,
        llm: Union[ChatBedrock, AzureChatOpenAI],
        objective: str,
        label_map: Dict[Union[int, str], str],
        target_names: List[str],
        text_column: str = "text",
        batch_size: int = 10,
    ) -> None:
        """
        Initialize the binary classifier.

        Args:
            llm: Language model instance
            objective: Classification objective
            label_map: Mapping from label IDs to label names
            target_names: List of target class names
            text_column: Name of the text column in the dataset
            batch_size: Number of inputs to process in a single batch
        """
        super().__init__(
            llm=llm, objective=objective, text_column=text_column, batch_size=batch_size
        )
        self.label_map = {k: v.title() for k, v in label_map.items()}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.labels = list(map(int, self.label_map.keys()))
        self.target_names = target_names
        self.system_prompt = self.get_system_prompt()
        self.ai_prompt = self.get_ai_prompt()

    def create_few_shot_example_str(self, examples: Dict[str, str]) -> str:
        """
        Format few-shot examples for inclusion in the prompt.

        Args:
            examples: Dictionary of examples organized by class

        Returns:
            Formatted examples string
        """
        if "mixed" in examples:
            return f"""Below is a shuffled list of examples with their labels indicating whether the objective is present or not in the text snippet:

<examples>
{examples['mixed']}
</examples>"""
        else:
            positive_examples = examples.get("Positive", "")
            negative_examples = examples.get("Negative", "")
            if positive_examples == "" and negative_examples == "":
                return ""
            elif positive_examples == "":
                return f"""Below is a list of negative examples that would not indicate that the objective is present in the text snippet:
<negative_examples>
{negative_examples}
</negative_examples>"""
            elif negative_examples == "":
                return f"""Below is a list of positive examples that would indicate that the objective is present in the text snippet:
<positive_examples>
{positive_examples}
</positive_examples>"""
            else:
                return f"""Below is a list of positive examples that would indicate that the objective is present in the text snippet:
<positive_examples>
{positive_examples}
</positive_examples>

Below is a list of negative examples that would not indicate that the objective is present in the text snippet:
<negative_examples>
{negative_examples}
</negative_examples>"""

    def create_criteria_str(self, criteria: Dict[str, str]) -> str:
        """
        Format classification criteria for inclusion in the prompt.

        Args:
            criteria: Dictionary of criteria organized by class

        Returns:
            Formatted criteria string
        """
        positive_criteria = criteria.get("Positive", "")
        negative_criteria = criteria.get("Negative", "")
        if positive_criteria == "" and negative_criteria == "":
            return ""
        elif positive_criteria == "":
            return f"""Below is a list of negative criteria that would not indicate that the objective is present in the text snippet:
<negative_criteria>
{negative_criteria}
</negative_criteria>"""
        elif negative_criteria == "":
            return f"""Below is a list of positive criteria that would indicate that the objective is present in the text snippet:
<positive_criteria>
{positive_criteria}
</positive_criteria>"""
        else:
            return f"""Below is a list of positive criteria that would indicate that the objective is present in the text snippet:
<positive_criteria>
{positive_criteria}
</positive_criteria>

Below is a list of negative criteria that would not indicate that the objective is present in the text snippet:
<negative_criteria>
{negative_criteria}
</negative_criteria>"""

    def create_description_str(self, description: str) -> str:
        """
        Format task description for inclusion in the prompt.

        Args:
            description: Detailed description of the classification task

        Returns:
            Formatted description string
        """
        return f"""Below is a detailed description of the classification task:
<description>
{description}
</description>"""

    def create_human_prompt_zero_shot(self, text_snippet: str) -> str:
        """
        Create a zero-shot prompt for binary classification.

        Args:
            text_snippet: Text to classify

        Returns:
            Zero-shot prompt string
        """
        return f"""<instructions>
Analyze the following text snippet and identify whether {self.objective}.
Provide a detailed reasoning for your decision (chain of thoughts) before delivering the final classification.
Label the snippet as either Positive (if {self.objective}) or Negative (if the snippet does not relate or contain relevant information).

Base your classification on the objective provided above.
</instructions>

<snippet>
{text_snippet}
</snippet>

<format>
<rationale> [Your reasoning] </rationale>
<label> [Positive or Negative] </label>
</format>
"""

    def create_human_prompt_few_shot(
        self, text_snippet: str, few_shot_examples: Dict[str, str]
    ) -> str:
        """
        Create a few-shot prompt for binary classification using examples.

        Args:
            text_snippet: Text to classify
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        few_shot_examples_str = self.create_few_shot_example_str(few_shot_examples)
        return f"""<instructions>
Analyze the following text snippet and identify whether {self.objective}.
Provide a detailed reasoning for your decision (chain of thoughts) before delivering the final classification.
Label the snippet as either Positive (if {self.objective}) or Negative (if the snippet does not relate or contain relevant information).

{few_shot_examples_str}

Base your classification on the objective and the examples provided above.
</instructions>

<snippet>
{text_snippet}
</snippet>

<format>
Your answer must be in the following format:
<rationale> [Your reasoning] </rationale>
<label> [Positive or Negative] </label>
</format>
"""

    def create_human_prompt_criteria(
        self, text_snippet: str, criteria: Dict[str, str]
    ) -> str:
        """
        Create a criteria-based prompt for binary classification.

        Args:
            text_snippet: Text to classify
            criteria: Dictionary of criteria organized by class

        Returns:
            Criteria-based prompt string
        """
        criteria_str = self.create_criteria_str(criteria)
        return f"""<instructions>
Analyze the following text snippet and identify whether {self.objective}.
Provide a detailed reasoning for your decision (chain of thoughts) before delivering the final classification.
Label the snippet as either Positive (if {self.objective}) or Negative (if the snippet does not relate or contain relevant information).

{criteria_str}

Base your classification on the objective and the criteria provided above.
</instructions>

<snippet>
{text_snippet}
</snippet>

<format>
<rationale> [Your reasoning] </rationale>
<label> [Positive or Negative] </label>
</format>
"""

    def create_human_prompt_description(
        self, text_snippet: str, description: str
    ) -> str:
        """
        Create a description-based prompt for binary classification.

        Args:
            text_snippet: Text to classify
            description: Detailed description of the classification task

        Returns:
            Description-based prompt string
        """
        description_str = self.create_description_str(description)
        return f"""<instructions>
Analyze the following text snippet and identify whether {self.objective}.
Provide a detailed reasoning for your decision (chain of thoughts) before delivering the final classification.
Label the snippet as either Positive (if {self.objective}) or Negative (if the snippet does not relate or contain relevant information).

{description_str}

Base your classification on the objective and the description provided above.
</instructions>

<snippet>
{text_snippet}
</snippet>

<format>
<rationale> [Your reasoning] </rationale>
<label> [Positive or Negative] </label>
</format>
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract rationale and label.

        Handles various formatting issues and extracts key information
        using regular expressions.

        Args:
            response: Raw response from the language model

        Returns:
            Dictionary with 'rationale' and 'label' keys
        """
        rationale = ""
        label = ""
        response = response.strip()
        if not response.startswith("<rationale>"):
            response = "<rationale> " + response
        if "</rationale>" not in response:
            response = response.replace("<label>", "</rationale> <label>")
        response_l = response.lower()
        try:
            rationale_match = re.search(
                "<rationale>(.*?)</rationale>", response, re.DOTALL | re.IGNORECASE
            )
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            label_match = re.search(
                "<label>(.*?)</label>", response, re.DOTALL | re.IGNORECASE
            )
            if label_match:
                label_value = label_match.group(1).strip().title()
                if label_value in self.target_names:
                    label = label_value
                elif "positive" in response_l and not "negative" in response_l:
                    label = "Positive"
                elif "negative" in response_l and not "positive" in response_l:
                    label = "Negative"
            return {"rationale": rationale, "label": label}
        except Exception as e:
            print(f"Parsing error: {e}")
            return {"rationale": "", "label": ""}

    def process_parsed_responses(
        self, parsed_responses: List[Dict[str, str]]
    ) -> Dict[str, List]:
        """
        Process a list of parsed responses to extract structured information.

        Args:
            parsed_responses: List of dictionaries containing parsed responses

        Returns:
            Dictionary containing lists of rationales, labels, and metadata
        """
        rationale_list = []
        predicted_raw_label_list = []
        predicted_label_list = []
        is_empty = []
        for pred in parsed_responses:
            rationale = pred["rationale"]
            rationale_list.append(rationale)
            raw_label = pred["label"]
            predicted_raw_label_list.append(raw_label)
            label = self.reverse_label_map.get(raw_label.title(), -1)
            predicted_label_list.append(label)
            if label == -1:
                is_empty.append(1)
            else:
                is_empty.append(0)
        final_responses = {
            "parsed_response": parsed_responses,
            "rationale": rationale_list,
            "predicted_raw_label": predicted_raw_label_list,
            "predicted_label": predicted_label_list,
            "is_empty": is_empty,
        }
        return final_responses
