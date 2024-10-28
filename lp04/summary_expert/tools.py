import logging
from typing import Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StoreConvertedPage(BaseModel):
    """Stores the web page content once it has been converted from structured text to markdown"""
    markdown_text: str = Field(description="The human-readible, markdown text of the page")

def store_converted_page(markdown_text: str) -> None:
    # No-op; exists to provide a structured tool interface that forces the model to comform to the
    # expected result format
    return markdown_text

store_converted_page_tool = StructuredTool.from_function(
    func=store_converted_page,
    name="StoreConvertedPage",
    args_schema=StoreConvertedPage
)

class StoreRefinedPageSummary(BaseModel):
    """Stores the web page content once it has been refined to ensure it adheres to the summary guidelines"""
    refined_text: str = Field(description="The refined, summarized text of the page")

def store_refined_page_summary(refined_text: str) -> None:
    # No-op; exists to provide a structured tool interface that forces the model to comform to the
    # expected result format
    return refined_text

store_refined_page_summary_tool = StructuredTool.from_function(
    func=store_refined_page_summary,
    name="StoreRefinedPageSummary",
    args_schema=StoreRefinedPageSummary
)

TOOLS_CONVERSION = [store_converted_page_tool]
TOOLS_REFINEMENT = [store_refined_page_summary_tool]
TOOLS_ALL = TOOLS_CONVERSION + TOOLS_REFINEMENT
