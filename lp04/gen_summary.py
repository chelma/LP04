import json
import logging

import click

from summary_expert.chain import perform_initial_conversion, perform_conversion_qc, perform_initial_refinement, perform_refinement_qc, perform_combined_summary, SummarizationPass
from utilities.logging import configure_logging
from utilities.scraping import extract_text_from_page

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)


@click.command()
@click.option('--urls', type=str, required=True, help='A comma-separated list of URLs to summarize.')
@click.option('--output', type=str, required=True, help='The output file location to save the generated markdown.')
def main(urls, output):
    # Log the supplied user inputs
    logger.info(f"URLs: {urls}")
    logger.info(f"Output file location: {output}")

    # Scrape the web pages
    url_list = urls.split(',')
    scraped_pages = [extract_text_from_page(url) for url in url_list]

    # Convert and summarize each page individually
    page_results = []
    for page in scraped_pages:
        # 1st pass on converting the structured text to markdown
        initial_state_conversion = SummarizationPass(
            url=page.url,
            text=json.dumps(page.content, indent=4),
            turns=[]
        )

        logger.info(f"1st pass on converting page: {page.url}")
        pass_results = perform_initial_conversion(initial_state_conversion)
        logger.debug(f"Conversion results: {pass_results.to_json()}")

        # QC pass on the converting markdown
        logger.info(f"QC pass on converting page: {page.url}")
        pass_results = perform_conversion_qc(pass_results)
        logger.debug(f"Conversion QC results: {pass_results.to_json()}")

        # 1st pass on refining the markdown summaries
        initial_state_refinement = SummarizationPass(
            url=page.url,
            text=pass_results.text,
            turns=[]
        )
        logger.info(f"1st pass on refining page: {page.url}")
        pass_results = perform_initial_refinement(initial_state_refinement)
        logger.debug(f"Refinement results: {pass_results.to_json()}")

        # QC pass on the refining summaries
        logger.info(f"QC pass on refining page: {page.url}")
        pass_results = perform_refinement_qc(pass_results)
        logger.debug(f"Refinement QC results: {pass_results.to_json()}")

        page_results.append(pass_results)
        logger.info(f"Summary completed for page: {page.url}")
        logger.info(f"Final refined text: \n{pass_results.text}")

    # Combine the summaries if there are multiple pages
    final_summary = None

    if len(page_results) > 1:
        logger.info("Combining summaries...")
        initial_state_combine = SummarizationPass(
            url=None,
            text=json.dumps([result.text for result in page_results]),
            turns=[]
        )

        combined_summary = perform_combined_summary(initial_state_combine)
        logger.debug(f"Combined summary: \n{combined_summary.to_json()}")

        final_summary = combined_summary.text
    else:
        final_summary = page_results[0].text

    # Store the results in the output file
    with open(output, 'w') as f:
        f.write(final_summary)

if __name__ == '__main__':
    main()