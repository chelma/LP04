import logging

from langchain_core.messages import HumanMessage

from summary_expert.graph import SummarizePageState, summarize_page_state_to_json, SUMMARIZE_GRAPH, SUMMARIZE_GRAPH_RUNNER
from utilities.logging import configure_logging
from utilities.scraping import extract_text_from_page

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)

# urls = [
#     "https://opensearch.org/docs/2.17/api-reference/index-apis/create-index/",
#     # "https://opensearch.org/docs/2.17/install-and-configure/configuring-opensearch/index-settings/",
#     # "https://opensearch.org/docs/2.17/field-types/",
#     # "https://opensearch.org/docs/2.17/im-plugin/index-alias/"
# ]

url = "https://opensearch.org/docs/2.17/api-reference/index-apis/create-index/"
raw_page = extract_text_from_page(url)

summarize_state = SummarizePageState(
    raw_page=raw_page
)
final_state = SUMMARIZE_GRAPH_RUNNER(summarize_state, 42)
logger.info(f"Final state: {summarize_page_state_to_json(final_state)}")
# logger.info(f"Converted Text: {final_state['converted_page']}")
logger.info(f"Refined Text: {final_state['refined_page']}")