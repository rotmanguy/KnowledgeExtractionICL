# MIT License
# Copyright (c) [2025] [Anonymized]
# See LICENSE file for full license text

from abc import ABC, abstractmethod
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re
from typing import Dict, List, Optional, Union


class CriteriaCreatorBase(ABC):
    """
    Base class for creating criteria for classification tasks using language models.

    This abstract class provides common functionality for generating criteria
    for both binary and multi-class classification tasks.
    """

    def __init__(
        self, llm: Union[ChatBedrock, AzureChatOpenAI], objective: str
    ) -> None:
        """
        Initialize the criteria creator base class.

        Args:
            llm: Language model instance (BedrockChat or AzureChat)
            objective: Classification objective
        """
        self.llm = llm
        self.objective = objective
        self.system_prompt = self.get_system_prompt()
        self.ai_prompt = self.get_ai_prompt()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Generate the system prompt for criteria creation.

        Returns:
            System prompt string
        """
        pass

    def get_human_prompt(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Generate a human prompt based on the criteria creation approach.

        Args:
            few_shot_examples: Few-shot examples organized by class
            description: Detailed description of the classification task

        Returns:
            Human prompt string
        """
        if few_shot_examples is not None:
            return self.create_human_prompt_few_shot(few_shot_examples)
        elif description is not None:
            return self.create_human_prompt_description(description)
        else:
            return self.create_human_prompt_zero_shot()

    def get_ai_prompt(self) -> str:
        """
        Generate the AI prompt for criteria creation.

        Returns:
            AI prompt string
        """
        return "<criteria>"

    def create_inputs(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> List[List[Union[SystemMessage, HumanMessage, AIMessage]]]:
        """
        Create input prompts for the language model based on the criteria creation approach.

        Args:
            few_shot_examples: Few-shot examples organized by class
            description: Detailed description of the classification task

        Returns:
            List of message sequences for the language model
        """
        inputs = [
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.get_human_prompt(
                        few_shot_examples=few_shot_examples, description=description
                    )
                ),
                AIMessage(content=self.ai_prompt),
            ]
        ]
        return inputs

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

    @abstractmethod
    def create_human_prompt_zero_shot(self) -> str:
        """
        Create a zero-shot prompt for criteria creation.

        Returns:
            Zero-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_few_shot(self, few_shot_examples: Dict[str, str]) -> str:
        """
        Create a few-shot prompt for criteria creation using examples.

        Args:
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_description(self, description: str) -> str:
        """
        Create a description-based prompt for criteria creation.

        Args:
            description: Detailed description of the classification task

        Returns:
            Description-based prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract criteria.

        Args:
            response: Raw response from the language model

        Returns:
            Dictionary with criteria organized by category
        """
        pass

    async def get_llm_responses(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate criteria for classification.

        Args:
            few_shot_examples: Few-shot examples organized by class
            description: Detailed description of the classification task

        Returns:
            Dictionary containing criteria

        Raises:
            ValueError: If multiple criteria creation approaches are provided simultaneously
        """
        if sum(x is not None for x in [few_shot_examples, description]) > 1:
            raise ValueError(
                "Only one of few_shot_examples or description should be provided."
            )
        inputs = self.create_inputs(
            few_shot_examples=few_shot_examples, description=description
        )
        responses = await self.llm.agenerate(inputs)
        responses = [response[0].text for response in responses.generations][0]
        parsed_responses = self.parse_response(responses)
        return parsed_responses


class BinaryCriteriaCreator(CriteriaCreatorBase):
    """
    Creates criteria for binary classification tasks using language models.

    This class generates positive and negative criteria for determining
    whether a given objective is present in text snippets.
    """

    def __init__(
        self, llm: Union[ChatBedrock, AzureChatOpenAI], objective: str
    ) -> None:
        """
        Initialize the binary criteria creator.

        Args:
            llm: Language model instance (BedrockChat or AzureChat)
            objective: Classification objective
        """
        self.llm = llm
        self.objective = objective
        self.system_prompt = self.get_system_prompt()
        self.ai_prompt = self.get_ai_prompt()

    def get_system_prompt(self) -> str:
        """
        Generate the system prompt for criteria creation.

        Returns:
            System prompt string
        """
        return f""""<description>
You are a knowledgeable annotator with experience in processing diverse types of textual inputs. Your task is to establish criteria for determining the presence of the following objective in text snippets: {self.objective}. The criteria should include positive criteria indicating the objective is present and negative criteria indicating the objective is not present in the text snippets.
</description>"""

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

    def create_human_prompt_zero_shot(self) -> str:
        """
        Create a zero-shot prompt for binary criteria creation.

        Returns:
            Zero-shot prompt string
        """
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate two lists: 
1. A list of positive criteria: Conditions that indicate the presence of the objective.
2. A list of negative criteria: Conditions that indicate the absence of the objective.

Base your criteria on the objective provided above.
The criteria should be as general as possible and should be applicable to any text snippet. 
The criteria should be clear and concise.
Each list of criteria should include at least five criteria and no more than ten criteria.
Each criterion should be self-explanatory and not require an example.
</instructions>

<format>
Your answer must be in the following format:
<criteria>
<positive>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</positive>
<negative>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</negative>
</criteria>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the criteria could be:
<criteria>
<positive>
Criterion 1: Questions or inquiries from the prospect about the pricing of the product or solution, explicitly considering their organization's size, needs, or purchasing capacity. 
Criterion 2: Comparisons made by the prospect regarding pricing across competitors or mentions of affordability or discounts, specifically in the context of their budget and organizational scale. 
Criterion 3: Negotiation attempts initiated by the prospect, such as requests for price reductions or additional value, aligned with their business size or operational priorities.
</positive>
<negative>
Criterion 1: Discussions initiated by the prospect that are unrelated to pricing, such as product features, timelines, or technical specifications, without tying back to cost or value for their organization.
Criterion2: General expressions of interest or dissatisfaction from the prospect that do not mention cost, pricing, or relevance to their business size.
Criterion 3: Comments from the prospect on financial or contractual matters focused on process or terms, rather than pricing details tailored to their organization's needs.
Criterion 4: Statements from the seller regarding pricing that are not directly responding to the prospect's inquiries or addressing their specific organizational context.
</negative>
</criteria>
</format>
"""

    def create_human_prompt_few_shot(self, few_shot_examples: Dict[str, str]) -> str:
        """
        Create a few-shot prompt for binary criteria creation using examples.

        Args:
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        few_shot_examples_str = self.create_few_shot_example_str(few_shot_examples)
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate two lists: 
1. A list of positive criteria: Conditions that indicate the presence of the objective.
2. A list of negative criteria: Conditions that indicate the absence of the objective.

{few_shot_examples_str}

Base your criteria on the objective and the labeled examples provided above.
The criteria should be as general as possible and should be applicable to any text snippet. 
The criteria should be clear and concise.
Each list of criteria should include at least five criteria and no more than ten criteria.
Each criterion should be self-explanatory and not require an example.
Each criterion should be based on at least two of the examples provided above.
</instructions>

<format>
Your answer must be in the following format:
<criteria>
<positive>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</positive>
<negative>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</negative>
</criteria>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the criteria could be:
<criteria>
<positive>
Criterion 1: Questions or inquiries from the prospect about the pricing of the product or solution, explicitly considering their organization's size, needs, or purchasing capacity. 
Criterion 2: Comparisons made by the prospect regarding pricing across competitors or mentions of affordability or discounts, specifically in the context of their budget and organizational scale. 
Criterion 3: Negotiation attempts initiated by the prospect, such as requests for price reductions or additional value, aligned with their business size or operational priorities.
</positive>
<negative>
Criterion 1: Discussions initiated by the prospect that are unrelated to pricing, such as product features, timelines, or technical specifications, without tying back to cost or value for their organization.
Criterion2: General expressions of interest or dissatisfaction from the prospect that do not mention cost, pricing, or relevance to their business size.
Criterion 3: Comments from the prospect on financial or contractual matters focused on process or terms, rather than pricing details tailored to their organization's needs.
Criterion 4: Statements from the seller regarding pricing that are not directly responding to the prospect's inquiries or addressing their specific organizational context.
</negative>
</criteria>
</format>
"""

    def create_human_prompt_description(self, description: str) -> str:
        """
        Create a description-based prompt for binary criteria creation.

        Args:
            description: Detailed description of the classification task

        Returns:
            Description-based prompt string
        """
        description_str = self.create_description_str(description)
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate two lists: 
1. A list of positive criteria: Conditions that indicate the presence of the objective.
2. A list of negative criteria: Conditions that indicate the absence of the objective.

{description_str}

Base your criteria on the objective and the description provided above.
The criteria should be as general as possible and should be applicable to any text snippet. 
The criteria should be clear and concise.
Each list of criteria should include at least five criteria and no more than ten criteria.
Each criterion should be self-explanatory and not require an example.
</instructions>

<format>
Your answer must be in the following format:
<criteria>
<positive>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</positive>
<negative>
Criterion 1: [Criterion 1]
Criterion 2: [Criterion 2]
...
</negative>
</criteria>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the criteria could be:
<criteria>
<positive>
Criterion 1: Questions or inquiries from the prospect about the pricing of the product or solution, explicitly considering their organization's size, needs, or purchasing capacity. 
Criterion 2: Comparisons made by the prospect regarding pricing across competitors or mentions of affordability or discounts, specifically in the context of their budget and organizational scale. 
Criterion 3: Negotiation attempts initiated by the prospect, such as requests for price reductions or additional value, aligned with their business size or operational priorities.
</positive>
<negative>
Criterion 1: Discussions initiated by the prospect that are unrelated to pricing, such as product features, timelines, or technical specifications, without tying back to cost or value for their organization.
Criterion2: General expressions of interest or dissatisfaction from the prospect that do not mention cost, pricing, or relevance to their business size.
Criterion 3: Comments from the prospect on financial or contractual matters focused on process or terms, rather than pricing details tailored to their organization's needs.
Criterion 4: Statements from the seller regarding pricing that are not directly responding to the prospect's inquiries or addressing their specific organizational context.
</negative>
</criteria>
</format>
"""

    def parse_response(self, response: str) -> Dict[str, str]:
        """
                Parses the response from the LLM to extract the positive and negative criteria.
                Assumes the response format:
                <criteria>
                <positive>
        Criterion 1: [Criterion 1]
        Criterion 2: [Criterion 2]
        ...</positive>
                <negative>
        Criterion 1: [Criterion 1]
        Criterion 2: [Criterion 2]
        ...</negative>
                </criteria>

                Returns a dictionary with 'positive' and 'negative' keys, or '' if parsing fails.
        """
        positive_criteria = ""
        negative_criteria = ""
        response = response.strip()
        if not response.startswith("<criteria>"):
            response = "<criteria>\n" + response
        if not response.endswith("</criteria>"):
            response = response + "\n</criteria>"
        try:
            positive_criteria_match = re.search(
                "<positive>(.*?)</positive>", response, re.DOTALL | re.IGNORECASE
            )
            negative_criteria_match = re.search(
                "<negative>(.*?)</negative>", response, re.DOTALL | re.IGNORECASE
            )
            if positive_criteria_match:
                positive_criteria = positive_criteria_match.group(1).strip()
            if negative_criteria_match:
                negative_criteria = negative_criteria_match.group(1).strip()
        except (IndexError, ValueError):
            return {"Positive": "", "Negative": ""}
        return {"Positive": positive_criteria, "Negative": negative_criteria}
