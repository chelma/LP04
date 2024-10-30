import json
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage

page_to_markdown_prompt_template = """
You are an AI assistant whose goal is to assist in creating detailed summaries of web pages.  You will
be provided with a structured representation of the content of a web page (listed below as source_text),
and your task is to generate a human-readable, markdown representation of the content.

While working towards the goal of creating the markdown representation, will ALWAYS follow the below
general guidelines:
<guidelines>
- Do not attempt to be friendly in your responses.  Be as direct and succint as possible.
- Think through the problem, extract all data from the task and the previous conversations before creating any plan.
- Never assume any parameter values while invoking a tool or function.
- You may not ask clarifying questions; if you need more information, indicate that in your output and stop.
</guidelines>

Additionally, you must ALWAYS follow these conversion_guidelines for the summaries you produce:
<conversion_guidelines>
- You will try to retain as much of the original detail as possible
- You will try to retain the exact wording and structure of the source_text
- You will not add any information that is not present in the source_text
- You MUST produce output in a human-readable format following markdown conventions
</conversion_guidelines>

The source text is <source_text>{source_text}</source_text>.
"""

def get_page_to_markdown_prompt_template(source_text: str) -> SystemMessage:
    return SystemMessage(
        content=page_to_markdown_prompt_template.format(
            source_text=source_text
        )
    )

page_refine_prompt_template = """
You are an AI assistant whose goal is to refine technical documentation.  You will be provided with an existing
document in a markdown format (listed below as source_text).  Your task is to restructure the content for clarity
and readability.

While working towards the goal of creating the refined document, will ALWAYS follow the below
general guidelines:
<guidelines>
- Think through the problem, extract all data from the task and the previous conversations before creating any plan.
- Never assume any parameter values while invoking a tool or function.
- You may not ask clarifying questions; if you need more information, indicate that in your output and stop.
</guidelines>

Additionally, you must ALWAYS follow these refinement_guidelines for the changes you produce:
<refinement_guidelines>
- You MUST retain all details about technical content such as API specifications, configuration values, code snippets, etc
- You MUST NOT add any information that is not present in the source_text
- You MUST produce output in a human-readable format following markdown conventions
- You should try to avoid using the exact same wording and structure as the source_text, unless it is required to retain technical detail
</refinement_guidelines>

The source text is <source_text>{source_text}</source_text>.
"""

def get_page_refine_prompt_template(source_text: Dict[str, Any]) -> SystemMessage:
    return SystemMessage(
        content=page_refine_prompt_template.format(
            source_text=json.dumps(source_text, indent=4)
        )
    )

page_qc_prompt_template = """
You are an AI assistant whose goal is to refine technical documentation as part of a multi-step process.  The previous
step in the process refined the documentation for clarity and readability.  You task is perform final quality control
on the refined document before it is ready for publishing.

You will be provided the original, unrefined document (listed below as source_text) and the refined document (listed
below as refined_text).  Both should be in markdown format.  Your task is to carefully review the source_text and
refined_text and produce an updated version of the refined_text that meets the following quality_control_guidelines:

<quality_control_guidelines>
- Retain the style and structure of the refined_text
- If you find technical details present in the source_text but missing from the refined_text, add them back in.
    By technical details, we mean API specifications, configuration values and options, code snippets, etc
- Do not add any information that is not present in the source_text
- Ensure the final output follows markdown conventions
</quality_control_guidelines>

While working towards the goal of producing a final, will ALWAYS follow the below general guidelines:
<guidelines>
- Think through the problem, extract all data from the task and the previous conversations before creating any plan.
- Never assume any parameter values while invoking a tool or function.
- You may not ask clarifying questions; if you need more information, indicate that in your output and stop.
</guidelines>

The source text is <source_text>{source_text}</source_text>.

The refined text is <refined_text>{refined_text}</refined_text>.
"""

def get_page_qc_prompt_template(source_text: str, refined_text: str) -> SystemMessage:
    return SystemMessage(
        content=page_qc_prompt_template.format(
            source_text=source_text,
            refined_text=refined_text
        )
    )