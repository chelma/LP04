from dataclasses import dataclass
import logging
from typing import Any, Dict, List

from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from summary_expert.prompting import get_page_to_markdown_prompt_template, get_page_refine_prompt_template, get_pages_combine_prompt_template
from summary_expert.tools import TOOLS_COMBINED, TOOLS_CONVERSION, TOOLS_REFINEMENT, store_combined_page_summary_tool, store_converted_page_tool, store_refined_page_summary_tool
from utilities.scraping import ScrapedPage


logger = logging.getLogger(__name__)

# Define a boto Config to use w/ our LLMs
config = Config(
    read_timeout=120 # Wait 2 minutes for a response from the LLM
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

llm_combine = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2",
    config=config
)
llm_combine_w_tools = llm_combine.bind_tools(TOOLS_COMBINED)

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

def perform_initial_conversion(initial_state: SummarizationPass) -> SummarizationPass:
    # Define some initial variables
    conversion_turns = initial_state.turns

    # Define the initial task for the conversion Agent
    conversion_turns.append(
        get_page_to_markdown_prompt_template(initial_state.text)
    )
    conversion_turns.append(
        HumanMessage(content="Please convert the source text into markdown and store it.")
    )

    # Execute the initial conversion
    response = llm_convert_w_tools.invoke(conversion_turns)
    conversion_turns.append(response)

    # Execute the tool call to conform to the ReAct format specification
    tool_call = response.tool_calls[-1]
    converted_text = store_converted_page_tool(tool_call["args"]) # Currently a no-op
    conversion_turns.append(
        ToolMessage(
            name="StoreConvertedPage",
            content="Stored the converted text",
            tool_call_id=tool_call["id"]
        )
    )

    return SummarizationPass(
        url=initial_state.url,
        text=converted_text,
        turns=conversion_turns
    )

def perform_conversion_qc(previous_pass: SummarizationPass) -> SummarizationPass:
    # Add the QC task to the turns
    qc_turns = previous_pass.turns

    qc_turns.append(
        AIMessage(
            content=(
                "I stored the converted text.  A copy of the converted text is pasted below.  I will review it carefully and ensure that the following criteria are met:"
                + "\n* The text follows markdown conventions"
                + "\n* No important details were lost from the source text in the conversion"
                + "\n\nAfter performing this review, I will store the updated, converted text."
                + f"\n\n<converted_text>{previous_pass.text}<\\converted_text>"
            )
        )
    )
    qc_turns.append(
        HumanMessage(content="Please review the converted text and store it.")
    )

    # Execute the QC pass on conversion
    response = llm_convert_w_tools.invoke(qc_turns)
    qc_turns.append(response)

    # Execute the tool call to conform to the ReAct format specification
    tool_call = response.tool_calls[-1]
    converted_text = store_converted_page_tool(tool_call["args"]) # Currently a no-op
    qc_turns.append(
        ToolMessage(
            name="StoreConvertedPage",
            content="Stored the converted text",
            tool_call_id=tool_call["id"]
        )
    )

    return SummarizationPass(
        url=previous_pass.url,
        text=converted_text,
        turns=qc_turns
    )

def perform_initial_refinement(initial_state: SummarizationPass) -> SummarizationPass:
    # Define the initial task for the summary Agent
    refinement_turns = initial_state.turns

    refinement_turns.append(
        get_page_refine_prompt_template(initial_state.text)
    )
    refinement_turns.append(
        HumanMessage(content="Please refine the source text and store it.")
    )

    # Execute the initial conversion
    response = llm_refine_w_tools.invoke(refinement_turns)
    refinement_turns.append(response)

    # Execute the tool call to conform to the ReAct format specification
    tool_call = response.tool_calls[-1]
    refined_text = store_refined_page_summary_tool(tool_call["args"]) # Currently a no-op
    refinement_turns.append(
        ToolMessage(
            name="StoreRefinedPageSummary",
            content="Stored the refined text",
            tool_call_id=tool_call["id"]
        )
    )

    return SummarizationPass(
        url=initial_state.url,
        text=refined_text,
        turns=refinement_turns
    )

def perform_refinement_qc(previous_pass: SummarizationPass) -> SummarizationPass:
    # Add the QC task to the turns
    qc_turns = previous_pass.turns

    qc_turns.append(
        AIMessage(
            content=(
                "I stored the refined text.  A copy of the refined text is pasted below.  I will review it carefully and ensure that the following criteria are met:"
                + "\n* The text follows markdown conventions"
                + "\n* No important details were lost from the source text in the refinement"
                + "\n\nAfter performing this review, I will store the updated, refined text."
                + f"\n\n<refined_text>{previous_pass.text}<\\refined_text>"
            )
        )
    )
    qc_turns.append(
        HumanMessage(content="Please review the refined text and store it.")
    )

    # Execute the QC pass on conversion
    response = llm_refine_w_tools.invoke(qc_turns)
    qc_turns.append(response)

    # Execute the tool call to conform to the ReAct format specification
    tool_call = response.tool_calls[-1]
    refined_text = store_refined_page_summary_tool(tool_call["args"]) # Currently a no-op
    qc_turns.append(
        ToolMessage(
            name="StoreRefinedPageSummary",
            content="Stored the refined text",
            tool_call_id=tool_call["id"]
        )
    )

    return SummarizationPass(
        url=previous_pass.url,
        text=refined_text,
        turns=qc_turns
    )

def perform_combined_summary(initial_state: SummarizationPass) -> SummarizationPass:
    # Define some initial variables
    combine_turns = initial_state.turns

    # Define the initial task for the summary Agent
    combine_turns.append(
        get_pages_combine_prompt_template(initial_state.text)
    )
    combine_turns.append(
        HumanMessage(content="Please combine the refined texts and store the combined summary.")
    )

    # Execute the initial conversion
    response = llm_combine_w_tools.invoke(combine_turns)
    combine_turns.append(response)

    # Execute the tool call to conform to the ReAct format specification
    tool_call = response.tool_calls[-1]
    combined_text = store_combined_page_summary_tool(tool_call["args"]) # Currently a no-op
    combine_turns.append(
        ToolMessage(
            name="StoreCombinedPageSummary",
            content="Stored the combined text",
            tool_call_id=tool_call["id"]
        )
    )

    return SummarizationPass(
        url=initial_state.url,
        text=combined_text,
        turns=combine_turns
    )