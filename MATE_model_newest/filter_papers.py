import json
import os
import re
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Filter papers")
    parser.add_argument("input_dir", type=str, help="Directory containing downloaded papers")
    parser.add_argument("--output_dir", type=str, help="Output file", default="./downloaded_papers_filtered")
    return parser.parse_args()

def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print(f"Output directory {args.output_dir} already exists. Overwrite? (y/n)")
        response = input()
        if response.lower() == "y":
            shutil.rmtree(args.output_dir)
            os.mkdir(args.output_dir)
        else:
            print("Exiting...")
            return

    files = [fname for fname in os.listdir(args.input_dir) if fname.endswith(".json") and not "no_results" in fname]
    matches = 0
    for fname in files:
        with open(f"{args.input_dir}/{fname}", "r") as f:
            data = json.load(f)
        
        body = data["body"]

        possibilities = ["wt.%", "wt %", "wt%", "weight percent", "chemical composition"]
        if not any([re.search(possibility, body, flags=re.IGNORECASE) for possibility in possibilities]):
            continue
        
        possibilities = ["yield strength", "yield stress", r"\bys\b", r"\byts\b"]
        if not any([re.search(possibility, body, flags=re.IGNORECASE) for possibility in possibilities]):
            continue

        possibilities = [r"\bmpa\b"]
        if not any([re.search(possibility, body, flags=re.IGNORECASE) for possibility in possibilities]):
            continue

        matches += 1
        shutil.copyfile(f"{args.input_dir}/{fname}", f"{args.output_dir}/{fname}")

    print(f"Found {matches} papers matching criteria and saved them to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
