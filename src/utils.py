import os
import json
import pandas as pd

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
import torch

def get_save_name_stem(
    modelpath: str, max_tokens_body: int, n_ctx: int, temperature: float, parsing_method: str
) -> str:
    modelname = Path(modelpath).stem.replace(" ", "_")
    return f"{modelname}_maxtokens{max_tokens_body}_nctx{n_ctx}_temp{temperature}_method-{parsing_method}"

def get_exceptions_name(save_name_stem: str) -> str:
    return f"exceptions_{save_name_stem}.json"

def get_csv_name(save_name_stem: str) -> str:
    return f"results_{save_name_stem}.csv"

def get_log_name(save_name_stem: str) -> str:
    return f"log_{save_name_stem}.log"

def get_parsedfiles_name(save_name_stem: str) -> str:
    return f"parsed_files_{save_name_stem}.txt"

def load_model_and_tokenizer(
    modelpath: str, hf_modelname: Optional[str] = None, **kwargs
) -> Tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = hf_modelname if hf_modelname is not None else modelpath
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        **kwargs,
    )
    if device == "cpu":
        model = model.to(device)
    return model, tokenizer

def save_data(all_data, df_to_continue, exceptions, save_name_stem, save_dir, parsing_method):
    create_df_func = get_create_df_func(parsing_method)
    csv_name = get_csv_name(save_name_stem)
    exceptions_name = get_exceptions_name(save_name_stem)
    try:
        df = create_df_func(all_data)
        df_all = pd.concat([df_to_continue, df], ignore_index=True)
        df_all = df_all[list(df.columns) + [c for c in df_all.columns if c not in df.columns]]
        df_all.to_csv(os.path.join(save_dir, csv_name), index=False)
        with open(os.path.join(save_dir, exceptions_name), "w+") as f:
            json.dump(exceptions, f)
    except:
        return None

def get_continue_csv_path(
    continue_csv_path: Optional[str], save_dir: str, continue_latest_same_params: bool, save_name_stem: str, logger: Any
) -> Optional[str]:
    if continue_csv_path and continue_csv_path.lower() == "latest":
        files = [
            f for f in os.scandir(save_dir) if (
                f.is_file()
                and f.name.endswith(".csv")
                and ((not continue_latest_same_params) or save_name_stem in f.name))
        ]
        if len(files) == 0:
            logger.log(f"No files found in {save_dir}")
            return None
        else:
            latest = max(files, key=lambda x: x.stat().st_ctime)
            continue_csv_path = str(Path(latest.path).resolve())
            logger.log(f"Using latest csv file: {continue_csv_path}")

    return continue_csv_path

def load_df_to_continue(continue_csv_path: Optional[str]) -> pd.DataFrame:
    if continue_csv_path:
        df_to_continue = pd.read_csv(continue_csv_path)
    else:
        df_to_continue = pd.DataFrame()

    return df_to_continue

def get_relevant_files(downloaded_papers_dir: str, file_extension: str = ".md") -> List[str]:
    """
    获取相关文件列表
    Args:
        downloaded_papers_dir: 论文目录
        file_extension: 文件扩展名,默认 .json,也可以是 .md
    """
    relevant_files = [
        f for f in os.listdir(downloaded_papers_dir)
        if f.endswith(file_extension) and "no_results" not in f
    ]
    return relevant_files

def load_exceptions(exceptions_path: str) -> Dict[str, Dict[str, str]]:
    if os.path.exists(exceptions_path):
        with open(exceptions_path, "r") as f:
            exceptions = json.load(f)
    else:
        exceptions = {}

    return exceptions

def filter_relevant_files(
    continue_training: bool,
    relevant_files: List[str],
    save_dir: str,
    downloaded_papers_dir: str,
    parsedfiles_name: str,
    logger: Logger,
    file_extension: str = ".md",
) -> List[str]:
    len_before = len(relevant_files)
    removed_already_parsed = 0

    if continue_training:
        if os.path.exists(os.path.join(save_dir, parsedfiles_name)):
            with open(os.path.join(save_dir, parsedfiles_name), "r") as file:
                parsed_files = file.read().splitlines()

            parsed_files = [el.strip() for el in parsed_files]
            relevant_files = [f for f in relevant_files if f not in parsed_files]

        removed_already_parsed = len_before - len(relevant_files)

    files_to_remove = []
    for f in relevant_files:
        file_path = os.path.join(downloaded_papers_dir, f)

        if file_extension == ".json":

            with open(file_path, "r", encoding='utf-8') as file:
                data = json.load(file)
            if len(data.get("body", "")) < 100:
                files_to_remove.append(f)

        elif file_extension == ".md":

            try:
                with open(file_path, "r", encoding='utf-8') as file:
                    content = file.read()

                if len(content.strip()) < 100:
                    files_to_remove.append(f)
            except Exception as e:
                logger.log(f"Error reading {f}: {e}")
                files_to_remove.append(f)

    for f in files_to_remove:
        relevant_files.remove(f)

    removed_empty_body = len_before - removed_already_parsed - len(relevant_files)
    len_after = len(relevant_files)

    logger.log(
        f"Filtered out {len_before - len_after} files that were already checked "
        f"({removed_already_parsed} files) or which had content with less than 100 "
        f"characters ({removed_empty_body} files)"
    )

    return relevant_files

def get_system_prompt(parsing_method: str) -> str:
    if parsing_method == "unconstrained":
        from src.freeform_parsing import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_model_wrapper(model, tokenizer, save_name, logger, parsing_method: str) -> Any:
    if parsing_method == "unconstrained":
        return model
    raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_parsing_func(parsing_method: str) -> Any:
    if parsing_method == "unconstrained":
        from src.freeform_parsing import parse_paper
        return parse_paper
    raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_create_df_func(parsing_method: str) -> Any:
    if parsing_method == "unconstrained":
        from src.freeform_parsing import build_df
        return build_df
    raise ValueError(f"Unknown parsing method: {parsing_method}")
