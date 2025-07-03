# MIT License
# Copyright (c) [2025] [Anonymized]
# See LICENSE file for full license text

from abc import ABC, abstractmethod
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re
from typing import Dict, List, Optional, Union


class DescriptionCreatorBase(ABC):
    """
    Base class for creating detailed descriptions for classification tasks using language models.

    This abstract class provides common functionality for generating task descriptions
    for both binary and multi-class classification tasks.
    """

    def __init__(
        self, llm: Union[ChatBedrock, AzureChatOpenAI], objective: str
    ) -> None:
        """
        Initialize the description creator base class.

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
        Generate the system prompt for description creation.

        Returns:
            System prompt string
        """
        pass

    def get_human_prompt(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a human prompt based on the description creation approach.

        Args:
            few_shot_examples: Few-shot examples organized by class
            criteria: Classification criteria organized by class

        Returns:
            Human prompt string
        """
        if few_shot_examples is not None:
            return self.create_human_prompt_few_shot(few_shot_examples)
        elif criteria is not None:
            return self.create_human_prompt_criteria(criteria)
        else:
            return self.create_human_prompt_zero_shot()

    def get_ai_prompt(self) -> str:
        """
        Generate the AI prompt for description creation.

        Returns:
            AI prompt string
        """
        return "<description>"

    def create_inputs(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> List[List[Union[SystemMessage, HumanMessage, AIMessage]]]:
        """
        Create input prompts for the language model based on the description creation approach.

        Args:
            few_shot_examples: Few-shot examples organized by class
            criteria: Classification criteria organized by class

        Returns:
            List of message sequences for the language model
        """
        inputs = [
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.get_human_prompt(
                        few_shot_examples=few_shot_examples, criteria=criteria
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
    def create_human_prompt_zero_shot(self) -> str:
        """
        Create a zero-shot prompt for description creation.

        Returns:
            Zero-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_few_shot(self, few_shot_examples: Dict[str, str]) -> str:
        """
        Create a few-shot prompt for description creation using examples.

        Args:
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        pass

    @abstractmethod
    def create_human_prompt_criteria(self, criteria: Dict[str, str]) -> str:
        """
        Create a criteria-based prompt for description creation.

        Args:
            criteria: Dictionary of criteria organized by class

        Returns:
            Criteria-based prompt string
        """
        pass

    def parse_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the description.

        Args:
            response: Raw response from the language model

        Returns:
            Description string
        """
        description = ""
        response = response.strip()
        if not response.startswith("<description>"):
            response = "<description> " + response
        if not response.endswith("</description>"):
            response = response + " </description>"
        try:
            description_match = re.search(
                "<description>\\s*(.*?)\\s*</description>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if description_match:
                description = description_match.group(1).strip()
            if not description:
                return ""
        except (IndexError, ValueError, AttributeError):
            return ""
        return description

    async def get_llm_responses(
        self,
        few_shot_examples: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate description for classification.

        Args:
            few_shot_examples: Few-shot examples organized by class
            criteria: Classification criteria organized by class

        Returns:
            Description string

        Raises:
            ValueError: If multiple description creation approaches are provided simultaneously
        """
        if sum(x is not None for x in [few_shot_examples, criteria]) > 1:
            raise ValueError(
                "Only one of few_shot_examples or criteria should be provided."
            )
        inputs = self.create_inputs(
            few_shot_examples=few_shot_examples, criteria=criteria
        )
        responses = await self.llm.agenerate(inputs)
        responses = [response[0].text for response in responses.generations][0]
        parsed_responses = self.parse_response(responses)
        return parsed_responses


class BinaryDescriptionCreator(DescriptionCreatorBase):
    """
    Creates detailed descriptions for binary classification tasks using language models.

    This class generates descriptions that help determine whether a given objective
    is present in text snippets.
    """

    def __init__(self, llm, objective):
        """
        Initialize the binary description creator.

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
        Generate the system prompt for binary description creation.

        Returns:
            System prompt string
        """
        return f"""<description>
You are a knowledgeable annotator with experience in processing diverse types of textual inputs. Your task is to establish a detailed description for determining the presence of the following objective in text snippets: {self.objective}. The description should include clear guidelines to help distinguish whether the objective is present or not in the text snippets.
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

    def create_human_prompt_zero_shot(self) -> str:
        """
        Create a zero-shot prompt for binary description creation.

        Returns:
            Zero-shot prompt string
        """
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate a detailed description of the classification task allowing for the identification of the objective in text snippets.

Base your description on the objective provided above.
The description should be as general as possible and should be applicable to any text snippet. 
The description should be clear and concise.
</instructions>

<format>
Your answer must be in the following format:
<description> ... </description>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the criteria could be:
<description> The task is to determine whether any part of the conversation reflects the prospect's inquiries, comparisons, or negotiations related to the pricing of a product or solution, with explicit consideration of their organization's size, budget, or needs. This includes identifying questions, statements, or requests from the prospect that highlight their interest in cost, affordability, or value. Annotate based on the presence of pricing-related discussions initiated by the prospect or closely tied to their organizational context. Exclude statements focused solely on non-pricing topics or generic comments from the seller unrelated to the prospect's pricing concerns. </description>
</format>
"""

    def create_human_prompt_few_shot(self, few_shot_examples: Dict[str, str]) -> str:
        """
        Create a few-shot prompt for binary description creation using examples.

        Args:
            few_shot_examples: Dictionary of examples organized by class

        Returns:
            Few-shot prompt string
        """
        few_shot_examples_str = self.create_few_shot_example_str(few_shot_examples)
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate a detailed description of the classification task allowing for the identification of the objective in text snippets.

{few_shot_examples_str}

Base your description on the objective and the labeled examples provided above.
The description should be as general as possible and should be applicable to any text snippet. 
The description should be clear and concise.
</instructions>

<format>
Your answer must be in the following format:
<description> [Your description] </description>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the description could be:
<description> The task is to determine whether any part of the conversation reflects the prospect's inquiries, comparisons, or negotiations related to the pricing of a product or solution, with explicit consideration of their organization's size, budget, or needs. This includes identifying questions, statements, or requests from the prospect that highlight their interest in cost, affordability, or value. Annotate based on the presence of pricing-related discussions initiated by the prospect or closely tied to their organizational context. Exclude statements focused solely on non-pricing topics or generic comments from the seller unrelated to the prospect's pricing concerns. </description>
</format>
"""

    def create_human_prompt_criteria(self, criteria: Dict[str, str]) -> str:
        """
        Create a criteria-based prompt for binary description creation.

        Args:
            criteria: Dictionary of criteria organized by class

        Returns:
            Criteria-based prompt string
        """
        criteria_str = self.create_criteria_str(criteria)
        return f"""<instructions>
You are tasked with annotating text snippets.
The end-goal task is to analyze text snippets and determine whether {self.objective}.

Your task is to generate a detailed description of the classification task allowing for the identification of the objective in text snippets.

{criteria_str}

Base your description on the objective and the criteria provided above.
The description should be as general as possible and should be applicable to any text snippet. 
The description should be clear and concise.
</instructions>

<format>
Your answer must be in the following format:
<description> [Your description] </description>

For example, for the objective of determining whether "the prospect discusses pricing in the context of the purchasing process", the description could be:
<description> The task is to determine whether any part of the conversation reflects the prospect's inquiries, comparisons, or negotiations related to the pricing of a product or solution, with explicit consideration of their organization's size, budget, or needs. This includes identifying questions, statements, or requests from the prospect that highlight their interest in cost, affordability, or value. Annotate based on the presence of pricing-related discussions initiated by the prospect or closely tied to their organizational context. Exclude statements focused solely on non-pricing topics or generic comments from the seller unrelated to the prospect's pricing concerns. </description>
</format>
"""
