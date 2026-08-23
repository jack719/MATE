import os
import json
import argparse

from src.logging import Logger
from src.sciencedirect_scraping import *

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--doi_file",
        type=str,
        required=True,
        help="Path to the file containing the dois to download the papers for. Must be a text file with a comma separated list of dois. E.g. 'doi1,doi2,doi3'.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./downloaded_papers",
        help="Directory to save the downloaded papers to. Default: ./downloaded_papers",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default='5eb5e9140dd5fc877a65bd5cfa8bdad9',
        help="API key to use for the Elsevier API. ",
    )

    return parser.parse_args()


def main():
    args = get_cli_args()

    logger = Logger()
    logger.init(args.save_dir, "download_papers", print_=True, save=True, continue_logging=True)

    assert os.path.exists(args.doi_file), f"File {args.doi_file} does not exist."

    with open(args.doi_file, "r") as f:
        dois = f.read().strip().split(",")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    already_saved = [f.replace(".json", "") for f in os.listdir(args.save_dir) if f.endswith(".json")]
    dois = [doi for doi in dois if doi.replace("/", "_") not in already_saved]

    no_results_json = os.path.join(args.save_dir, "no_results.json")

    n_exceptions_streak = 0
    for i, doi in enumerate(dois):
        logger.log(f"Downloading {i+1}/{len(dois)}: {doi}...")

        if os.path.exists(no_results_json):
            with open(no_results_json, "r") as f:
                no_results = json.load(f)
        else:
            no_results = {"doi": []}

        if doi in no_results["doi"]:
            logger.log(f"    Skipping {i+1}/{len(dois)}: {doi} -> no results found previously")
            continue

        doi_escaped = doi.replace("/", "_")
        doi_result_json = os.path.join(args.save_dir, f"{doi_escaped}.json")

        try:
            results = get_text_abstract_via_elsevier_api(doi, args.api_key)
        except CriticalElsevierError as e:
            logger.log(f"Error: {e}")
            raise e
        except Exception as e:
            logger.log(f"Error: {e}")

            add_doi_to_no_results(doi, args.save_dir)
            n_exceptions_streak += 1

            #if n_exceptions_streak > 50:
            #    logger.log("Too many exceptions in a row. Exiting.")
            #    break

            continue

        n_exceptions_streak = 0

        logger.log(f"    Saving {doi}...")
        with open(doi_result_json, "w+") as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
