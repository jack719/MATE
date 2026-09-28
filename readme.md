# MATE: Two-Stage Alloy Information Extraction

This repository contains the parsing code used to extract titanium-alloy samples, compositions, processing routes, and mechanical properties from paper text. The maintained workflow starts with Markdown papers that have already been produced by a document-conversion tool such as MinerU.

Only the `unconstrained` parser is exposed because it is the only active path in this repository.

## Installation

Use Python 3.10 or newer and install the dependencies:

```bash
pip install -r requirements.txt
```

A local Hugging Face causal language model and its tokenizer are required. The model must support `generate`, and the tokenizer must provide `encode` and `decode`; `apply_chat_template` is used when available.

## Input format

Place one or more `.md` files in an input directory.

## Quickstart

```bash
python parse_downloaded_papers.py \
  --modelpath /path/to/model \
  --downloaded_papers_dir /path/to/markdown_papers \
  --save_dir ./outputs
```

The default CLI values point to the original authors' machine paths, so pass explicit paths on a new machine. To continue an earlier run, use `--continue_csv_path latest`; parsed filenames and exceptions are stored in the save directory.

## Outputs

The save directory contains:

- `results_<parameters>.csv`: flattened extraction records;
- `exceptions_<parameters>.json`: files that failed during reading, generation, or JSON conversion;
- `parsed_files_<parameters>.txt`: filenames already attempted;
- `log_<parameters>.log`: timestamped execution log.

## Citation

If you use this code, cite the associated MATE paper and the original repository. Replace the placeholder below with the final bibliographic record before publication:

```bibtex
@software{mate,
  author  = {MATE authors},
  title   = {MATE: Two-Stage Alloy Information Extraction},
  year    = {2026},
  url     = {https://github.com/jack719/MATE}
}
```

## License

This project keeps the GNU General Public License v3.0 from the upstream repository. See `LICENSE`.
