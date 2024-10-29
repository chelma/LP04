from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List

from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from summary_expert.prompting import get_page_to_markdown_prompt_template, get_page_refine_prompt_template
from summary_expert.tools import TOOLS_CONVERSION, TOOLS_REFINEMENT, store_converted_page_tool, store_refined_page_summary_tool
from utilities.scraping import ScrapedPage


logger = logging.getLogger(__name__)

# Define a boto Config to use w/ our LLMs that's more resilient to long waits and frequent throttling
config = Config(
    read_timeout=120, # Wait 2 minutes for a response from the LLM
    retries={
        'max_attempts': 10,  # Increase the number of retry attempts
        'mode': 'adaptive'   # Use adaptive retry strategy for better throttling handling
    }
)

# Define our LLMs
llm_convert = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2",
    config=config
)
llm_convert_w_tools = llm_convert.bind_tools(TOOLS_CONVERSION)

llm_refine = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2",
    config=config
)
llm_refine_w_tools = llm_refine.bind_tools(TOOLS_REFINEMENT)

@dataclass
class SummarizationPass:
    """A dataclass to store the results of a summarization pass and feed it into the next pass"""
    url: str
    text: str
    turns: List[BaseMessage]

    def to_json(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "text": self.text,
            "turns": [turn.to_json() for turn in self.turns]
        }

@dataclass
class SummarizationPassBatch:
    """A dataclass to store the results of a batch of summarization passes"""
    passes: List[SummarizationPass]

    def to_json(self) -> Dict[str, Any]:
        return {
            "sections": [pass_.to_json() for pass_ in self.passes]
        }

def perform_initial_conversion_batched(scraped_page: ScrapedPage) -> SummarizationPassBatch:
    # Create the inference context for each section of the page
    batched_context = []
    for section_key, section_contents in scraped_page.content.items():
        section_turns = []
        section_turns.append(
            get_page_to_markdown_prompt_template(json.dumps({section_key: section_contents}))
        )
        section_turns.append(
            HumanMessage(content="Please convert the source text into markdown and store it.")
        )
        batched_context.append(section_turns)
    
    # Execute the initial conversion by section
    # Ideally, we'd use a batched call here, but Bedrock's approach to that is an asynchronous process that writes
    # the results to S3 and returns a URL to the results.  This is not implemented by default in the ChatBedrockConverse
    # class, so we'll skip batch processing for now.
    response_batched = []
    for section_turns in batched_context:
        response = llm_convert_w_tools.invoke(section_turns)
        response_batched.append(response)

    # Process the results of the batched conversion
    conversion_passes = []
    for section_id in range(len(batched_context)):
        section_turns = batched_context[section_id]

        section_response = response_batched[section_id]
        section_turns.append(section_response)

        # Execute the tool call to conform to the ReAct format specification
        tool_call = section_response.tool_calls[-1]
        converted_text = store_converted_page_tool(tool_call["args"]) # Currently a no-op
        section_turns.append(
            ToolMessage(
                name="StoreConvertedPage",
                content="Stored the converted text",
                tool_call_id=tool_call["id"]
            )
        )

        conversion_passes.append(
            SummarizationPass(
                url=scraped_page.url,
                text=converted_text,
                turns=section_turns
            )
        )

    return SummarizationPassBatch(passes=conversion_passes)

def perform_initial_refinement_batch(url: str, sections: List[str]) -> SummarizationPassBatch:
    # Create the inference context for each section of the page
    batched_context = []
    for section in sections:
        section_turns = []
        section_turns.append(
            get_page_refine_prompt_template(section)
        )
        section_turns.append(
            HumanMessage(content="Please refine the source text and store it.")
        )
        batched_context.append(section_turns)
    
    # Execute the initial refinement by section
    # Ideally, we'd use a batched call here, but Bedrock's approach to that is an asynchronous process that writes
    # the results to S3 and returns a URL to the results.  This is not implemented by default in the ChatBedrockConverse
    # class, so we'll skip batch processing for now.
    response_batched = []
    for section_turns in batched_context:
        response = llm_refine_w_tools.invoke(section_turns)
        response_batched.append(response)

    # Process the results of the batched conversion
    conversion_passes = []
    for section_id in range(len(batched_context)):
        section_turns = batched_context[section_id]

        section_response = response_batched[section_id]
        section_turns.append(section_response)

        # Execute the tool call to conform to the ReAct format specification
        tool_call = section_response.tool_calls[-1]
        converted_text = store_refined_page_summary_tool(tool_call["args"]) # Currently a no-op
        section_turns.append(
            ToolMessage(
                name="StoreRefinedPageSummary",
                content="Stored the refined text",
                tool_call_id=tool_call["id"]
            )
        )

        conversion_passes.append(
            SummarizationPass(
                url=url,
                text=converted_text,
                turns=section_turns
            )
        )

    return SummarizationPassBatch(passes=conversion_passes)