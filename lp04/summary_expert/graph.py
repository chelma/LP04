from functools import wraps
import json
import logging
from typing import Annotated, Any, Callable, Dict, List, Literal
from typing_extensions import TypedDict
import uuid

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledGraph

from summary_expert.prompting import get_page_to_markdown_prompt_template, get_page_summary_prompt_template
from summary_expert.tools import TOOLS_CONVERSION, TOOLS_REFINEMENT, store_converted_page_tool, store_refined_page_summary_tool


logger = logging.getLogger(__name__)

# Define our LLMs
llm_convert = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_convert_w_tools = llm_convert.bind_tools(TOOLS_CONVERSION)

llm_refine = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_refine_w_tools = llm_refine.bind_tools(TOOLS_REFINEMENT)

# Define the state our graph will be operating on
class SummarizePageState(TypedDict):
    # Store the page content in its various stages of conversion
    raw_page: Dict[str, Any]
    converted_page: str
    refined_page: str

    # Hold the internal conversation of the Python expert
    convert_turns: Annotated[List[BaseMessage], add_messages]
    refine_turns: Annotated[List[BaseMessage], add_messages]

    # Flags used to constrol the flow of the graph
    convert_complete: bool
    refine_1st_complete: bool
    refine_2nd_complete: bool

def summarize_page_state_to_json(state: SummarizePageState) -> Dict[str, Any]:
    return {
        "raw_page": state.get("raw_page", {}),
        "convert_complete": state.get("convert_complete", False),
        "converted_page": state.get("converted_page", ""),
        "convert_turns": [turn.to_json() for turn in state.get("convert_turns", [])],
        "refine_1st_complete": state.get("refine_1st_complete", False),
        "refine_2nd_complete": state.get("refine_2nd_complete", False),
        "refined_page": state.get("refined_page", ""),
        "refine_turns": [turn.to_json() for turn in state.get("refine_turns", [])]
    }
    
def trace_summarize_node(func: Callable[[SummarizePageState], Dict[str, Any]]) -> Callable[[SummarizePageState], Dict[str, Any]]:
    @wraps(func)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Entering node: {func.__name__}")
        state_json = summarize_page_state_to_json(state)
        logging.debug(f"Starting state: {str(state_json)}")
        
        result = func(state)
        
        logging.debug(f"Output of {func.__name__}: {result}")
        
        return result
    
    return wrapper

# Set up our tools
# N/A

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Define our graph
summarize_page_graph = StateGraph(SummarizePageState)

# Set up our graph nodes
class InvalidStateError(Exception):
    pass

@trace_summarize_node
def node_validate_starting_state(state: SummarizePageState):
    # Ensure we have the raw page content
    if not state.get("raw_page", {}):
        raise InvalidStateError("State 'raw_page' is missing.  You must provide the structured text to be summarized.")
    
    # Ensure both of the turn lists are empty
    if state.get("convert_turns", []) or state.get("refine_turns", []):
        raise InvalidStateError("State 'convert_turns' and 'refine_turns' must be empty.")
    
    # Ensure the completion flags are set to False
    if state.get("convert_complete", False) or state.get("refine_1st_complete", False) or state.get("refine_2nd_complete", False):
        raise InvalidStateError("State 'convert_complete', 'refine_1st_complete', and 'refine_2nd_complete' must be False.")

    return {"convert_turns": [], "refine_turns": [], "convert_complete": False, "refine_1st_complete": False, "refine_2nd_complete": False}

@trace_summarize_node
def node_invoke_llm_convert_initial(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to perform the initial conversation of structured text to markdown
    """
    convert_turns = state["convert_turns"]
    convert_turns.append(
        get_page_to_markdown_prompt_template(state["raw_page"])
    )
    convert_turns.append(
        HumanMessage(content="Please convert the source text into markdown and store it.")
    )

    response = llm_convert_w_tools.invoke(convert_turns)
    convert_turns.append(response)

    return {"convert_turns": convert_turns}

@trace_summarize_node
def node_invoke_llm_convert_2nd_pass(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to perform a second pass on the conversion of structured text to markdown for quality control
    """
    new_turns = []
    new_turns.append(
        AIMessage(
            content=(
                "I stored the converted text.  A copy of the converted text is pasted below.  I will review it carefully and ensure that the following criteria are met:"
                + "\n* The text follows markdown conventions"
                + "\n* No important details were lost from the source text in the conversion"
                + "\n\nAfter performing this review, I will store the updated, converted text."
                + f"\n\n<converted_text>{state['converted_page']}<\\converted_text>"
            )
        )
    )
    new_turns.append(
        HumanMessage(content="Please review the converted text and store it.")
    )

    all_turns = state["convert_turns"] + new_turns

    response = llm_convert_w_tools.invoke(all_turns)
    new_turns.append(response)

    return {"convert_turns": new_turns, "convert_complete": True}

@trace_summarize_node
def node_invoke_llm_refine_initial(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to perform refine the generated markdown text
    """
    refine_turns = state["refine_turns"]
    refine_turns.append(
        get_page_summary_prompt_template(state["converted_page"])
    )
    refine_turns.append(
        HumanMessage(content="Please refine the source text and store it.")
    )

    response = llm_refine_w_tools.invoke(refine_turns)
    refine_turns.append(response)

    return {"refine_turns": refine_turns, "refine_1st_complete": True}

@trace_summarize_node
def node_invoke_llm_refine_2nd_pass(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to perform a second pass on the refining the markdown text for quality control
    """
    new_turns = []
    new_turns.append(
        AIMessage(
            content=(
                "I stored the refined text.  A copy of the refined text is pasted below.  I will review it carefully and ensure that the following criteria are met:"
                + "\n* The text follows markdown conventions"
                + "\n* No important details were lost from the source text in the refinement"
                + "\n\nAfter performing this review, I will store the updated, refined text."
                + f"\n\n<refined_text>{state["refined_page"]}<\\refined_text>"
            )
        )
    )
    new_turns.append(
        HumanMessage(content="Please review the refined text and store it.")
    )

    all_turns = state["refine_turns"] + new_turns

    response = llm_refine_w_tools.invoke(all_turns)
    new_turns.append(response)

    return {"refine_turns": new_turns, "refine_2nd_complete": True}

@trace_summarize_node
def node_store_text_convert(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to store the converted text
    """
    outcome = {}
    tool_call = state["convert_turns"][-1].tool_calls[-1]

    # We're storing the converted text
    if tool_call["name"] == "StoreConvertedPage":
        text = store_converted_page_tool(tool_call["args"])
        outcome["converted_page"] = text
        outcome["convert_turns"] = [
            ToolMessage(
                name="StoreConvertedPage",
                content="Stored the converted text",
                tool_call_id=tool_call["id"]
            )
        ]

    else:
        raise ValueError(f"Unexpected tool call name: {tool_call['name']}")

    return outcome

@trace_summarize_node
def node_store_text_refine(state: SummarizePageState) -> Dict[str, any]:
    """
    Node to store the refined text
    """
    outcome = {}
    tool_call = state["refine_turns"][-1].tool_calls[-1]

    # We're storing the refined text
    if tool_call["name"] == "StoreRefinedPageSummary":
        text = store_refined_page_summary_tool(tool_call["args"])
        outcome["refined_page"] = text
        outcome["refine_turns"] = [
            ToolMessage(
                name="StoreRefinedPageSummary",
                content="Stored the refined text",
                tool_call_id=tool_call["id"]
            )
        ]

    else:
        raise ValueError(f"Unexpected tool call name: {tool_call['name']}")

    return outcome

summarize_page_graph.add_node("node_validate_starting_state", node_validate_starting_state)
summarize_page_graph.add_node("node_invoke_llm_convert_initial", node_invoke_llm_convert_initial)
summarize_page_graph.add_node("node_invoke_llm_convert_2nd_pass", node_invoke_llm_convert_2nd_pass)
summarize_page_graph.add_node("node_invoke_llm_refine_initial", node_invoke_llm_refine_initial)
summarize_page_graph.add_node("node_invoke_llm_refine_2nd_pass", node_invoke_llm_refine_2nd_pass)
summarize_page_graph.add_node("node_store_text_convert", node_store_text_convert)
summarize_page_graph.add_node("node_store_text_refine", node_store_text_refine)

# Define our graph edges

def next_node(state: SummarizePageState) -> Literal[
            "node_invoke_llm_convert_2nd_pass", "node_invoke_llm_refine_initial", "node_invoke_llm_refine_2nd_pass", END
        ]:
    if not state["convert_complete"]:
        return "node_invoke_llm_convert_2nd_pass"
        
    if state["convert_complete"] and not state["refine_1st_complete"]:
        return "node_invoke_llm_refine_initial"
    
    if state["convert_complete"] and not state["refine_2nd_complete"]:
        return "node_invoke_llm_refine_2nd_pass"
    
    return END

summarize_page_graph.add_edge(START, "node_validate_starting_state")
summarize_page_graph.add_edge("node_validate_starting_state", "node_invoke_llm_convert_initial")
summarize_page_graph.add_edge("node_invoke_llm_convert_initial", "node_store_text_convert")
summarize_page_graph.add_edge("node_invoke_llm_convert_2nd_pass", "node_store_text_convert")
summarize_page_graph.add_edge("node_invoke_llm_refine_initial", "node_store_text_refine")
summarize_page_graph.add_edge("node_invoke_llm_refine_2nd_pass", "node_store_text_refine")
summarize_page_graph.add_conditional_edges("node_store_text_convert", next_node)
summarize_page_graph.add_conditional_edges("node_store_text_refine", next_node)

# Finally, compile the graph into a LangChain Runnable
SUMMARIZE_GRAPH = summarize_page_graph.compile(checkpointer=checkpointer)

def _create_runner(workflow: CompiledGraph):
    def run_workflow(summarize_state: SummarizePageState, thread: int) -> SummarizePageState:
        states = workflow.stream(
            summarize_state,
            config={"configurable": {"thread_id": thread}},
            stream_mode="values"
        )

        final_state = None
        for state in states:
            # if "summarize_turns" in state:
            #     state["summarize_turns"][-1].pretty_print()
            #     logger.info(state["summarize_turns"][-1].to_json())
            final_state = state

        return final_state

    return run_workflow

SUMMARIZE_GRAPH_RUNNER = _create_runner(SUMMARIZE_GRAPH)