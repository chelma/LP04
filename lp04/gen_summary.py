import json
import logging

import click

from summary_expert.chain import perform_initial_conversion_batched, perform_initial_refinement, perform_refinement_qc, SummarizationPass
from utilities.logging import configure_logging
from utilities.scraping import extract_text_from_page

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)


@click.command()
@click.option('--url', type=str, required=True, help='The URL to summarize.')
@click.option('--output', type=str, required=True, help='The output file location to save the generated markdown.')
def main(url, output):
    # Log the supplied user inputs
    logger.info(f"URL: {url}")
    logger.info(f"Output file location: {output}")

    # Scrape the web pages
    scraped_page = extract_text_from_page(url)

    # 1st pass on converting the structured text to markdown
    logger.info(f"1st pass on converting page: {scraped_page.url}")
    pass_results = perform_initial_conversion_batched(scraped_page)
    logger.debug(f"Conversion results: {pass_results.to_json()}")

    # 1st pass on refining the markdown summaries
    converted_text = "\n".join([pass_result.text for pass_result in pass_results.passes])
    logger.debug(f"Converted text: \n{converted_text}")

    initial_state_refinement = SummarizationPass(
        scraped_page.url,
        converted_text,
        []
    )
    logger.info(f"1st pass on refining page: {scraped_page.url}")
    pass_results = perform_initial_refinement(initial_state_refinement)
    logger.debug(f"Refinement results: {pass_results.to_json()}")

    # 2nd pass on refining the markdown summaries for quality control
    logger.info(f"2nd pass on refining page: {scraped_page.url}")
    pass_results = perform_refinement_qc(initial_state_refinement)
    logger.debug(f"Refinement results: {pass_results.to_json()}")

    logger.info(f"Summary completed for page: {scraped_page.url}")
    logger.info(f"Final refined text: \n{pass_results.text}")

    # Store the results in the output file
    with open(output, 'w') as f:
        f.write(pass_results.text)

if __name__ == '__main__':
    main()