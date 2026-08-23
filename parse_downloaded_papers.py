#TODO markdown版本
import os
import json
import argparse
import traceback
from src.logging import Logger
from src.utils import *
from datetime import datetime, timedelta

LOGGER = Logger()

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpath",
        type=str,
        default='/mnt/data2/home/liujj/model/llama/Llama-3.1-8B-Instruct',
        help="Path to local model file to use for parsing (assumed format: .gguf)",
    )
    parser.add_argument(
        "--downloaded_papers_dir",
        type=str,
        default="/mnt/data2/home/liujj/dax_paper_process/markdown_no_ref",
        help="Directory containing the downloaded papers (markdown files). Default: /home/pichlmannl/downloaded_papers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/data2/home/liujj/ai4M_data_H20/modelResult/llama31_CODE_2Stages_check/markdownNEW_no_ref",
        help="Directory to save the parsed data to (as csv). Default: /home/pichlmannl/data",
    )
    parser.add_argument(
        "--continue_csv_path",
        type=str,
        default=None,
        help="Path to the saved csv file containing results of already parsed papers to continue at. If set to 'latest', the latest csv file in save_dir is used. Default: None",
    )
    parser.add_argument(
        "--continue_latest_same_params",
        action="store_false",
        help='Only relevant if --continue_csv_path="latest". If set to true, continues with the latest csv file which had exactly the same parsing-parameters, otherwise parameters are not checked and simply the latest csv file is used (parsing parameters are modelpath, max_tokens_body, n_ctx, temperature, parsing_method). Default: True',
    )
    parser.add_argument(
        "--max_tokens_body",
        type=int,
        default=None,
        help="Maximum number of tokens at which the paper is truncated. Default: None",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="Main GPU to use. Default: None"
    )
    parser.add_argument(
        "--n_ctx", type=int, default=None, help="Context window size. Default: None"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for model. Default: None",
    )
    parser.add_argument(
        "--save_freq_minutes",
        type=int,
        default=10,
        help="Frequency in minutes at which to save the parsed data to csv. Default: 10",
    )
    parser.add_argument(
        "--skip_prev_exceptions",
        action="store_true",
        help="Skip previously encountered exceptions. Default: False",
    )
    parsing_choices = ["outlines", "guidance", "unconstrained"]
    parser.add_argument(
        "--parsing_method",
        type=str,
        default="unconstrained",
        choices=parsing_choices,
        help=f"Method to use for parsing. Must be one of {parsing_choices}. Default: guidance.",
    )
    parser.add_argument(
        "--hf_modelname",
        type=str,
        default=None,
        help="Name of the corresponding huggingface model, this is used to load the corresponding tokenizer. If not set, the model's included tokenizer is used. Default: None",
    )
    return parser.parse_args()

def parse_markdown_file(filepath):
    """
    解析Markdown文件,提取标题、摘要和正文
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 初始化
    title = ""
    abstract = ""
    body = content
    
    lines = content.split('\n')
    
    # 尝试提取标题 (第一个 # 标题)
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            title = line.strip()[2:].strip()
            # 移除标题行
            lines = lines[i+1:]
            break
    
    # 尝试提取摘要 (查找 Abstract 或 ## Abstract 部分)
    abstract_start = -1
    abstract_end = -1
    
    for i, line in enumerate(lines):
        lower_line = line.strip().lower()
        if ('abstract' in lower_line or '摘要' in lower_line) and (line.strip().startswith('#') or line.strip().startswith('**')):
            abstract_start = i + 1
            continue
        
        # 如果找到了摘要开始,寻找摘要结束(下一个标题或空行后的标题)
        if abstract_start != -1 and i > abstract_start:
            if line.strip().startswith('##') or line.strip().startswith('# '):
                abstract_end = i
                break
    
    # 提取摘要
    if abstract_start != -1:
        if abstract_end != -1:
            abstract = '\n'.join(lines[abstract_start:abstract_end]).strip()
            # 正文从摘要结束后开始
            body = '\n'.join(lines[abstract_end:]).strip()
        else:
            # 如果没有找到明确的结束,取前几段作为摘要
            abstract_lines = []
            for i in range(abstract_start, min(abstract_start + 10, len(lines))):
                if lines[i].strip().startswith('##') or lines[i].strip().startswith('# '):
                    break
                abstract_lines.append(lines[i])
            abstract = '\n'.join(abstract_lines).strip()
            body = '\n'.join(lines[abstract_start:]).strip()
    
    return {
        "title": title,
        "abstract": abstract,
        "body": body
    }

def main():
    ########### Setup
    args = get_cli_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_name_stem = get_save_name_stem(
        args.modelpath, args.max_tokens_body, args.n_ctx, args.temperature, args.parsing_method
    )
    exceptions_name = get_exceptions_name(save_name_stem)
    parsedfiles_name = get_parsedfiles_name(save_name_stem)
    continue_training = args.continue_csv_path is not None

    LOGGER.init(
        args.save_dir,
        save_name_stem,
        continue_logging=continue_training,
    )
    LOGGER.log(
        f"\n\n--------------------\nStarting parsing with model: {args.modelpath}\n--------------------\n\n"
    )

    continue_csv_path = get_continue_csv_path(args.continue_csv_path, args.save_dir, args.continue_latest_same_params, save_name_stem, LOGGER)
    df_to_continue = load_df_to_continue(continue_csv_path)

    def get_files(exceptions=None):
        # 修改为查找 .md 文件而不是 .json 文件
        relevant_files = get_relevant_files(args.downloaded_papers_dir, file_extension='.md')
        if exceptions is None:
            if not continue_training or args.skip_prev_exceptions:
                load_exc_name = ""
            else:
                load_exc_name = os.path.join(args.save_dir, exceptions_name)
            exceptions = load_exceptions(load_exc_name)
        relevant_files = filter_relevant_files(
            continue_training if exceptions is None else True, relevant_files, args.save_dir, args.downloaded_papers_dir, parsedfiles_name, LOGGER, ".md"
        )
        return relevant_files, exceptions

    relevant_files, exceptions = get_files()
    if not len(relevant_files):
        LOGGER.log("No files to parse, exiting...")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.modelpath, args.hf_modelname, temperature=args.temperature
    )
    LOGGER.warn("Using chatml format - no other formats currently supported")
    if args.parsing_method == "guidance":
        LOGGER.warn("Loading ChatMLTemplate into the guidance model - no other formats currently supported")

    ########### start parsing files
    parsing_func = get_parsing_func(args.parsing_method)
    all_data = []
    parsed_files = []
    last_save_time = datetime.now()
    n_exceptions_streak = 0
    mean_time_per_file = 0
    n_files = 0
    projected_finish_time = None
    system_prompt = get_system_prompt(args.parsing_method)
    idx = 0

    while len(relevant_files):
        relevant_files, _ = get_files(exceptions)
        fname = relevant_files.pop(0)
        if fname in parsed_files:
            raise ValueError(f"File {fname} already parsed, this should not happen!")
        parsed_files.append(fname)

        n_entries = 0
        if len(all_data):
            n_entries = sum([len(d) for d in all_data])
        LOGGER.log(
            f"{len(relevant_files)} files left ({idx} already parsed): {fname} - projected finish time: {projected_finish_time} - current number of entries: {n_entries}"
        )

        # 修改:读取 Markdown 文件而不是 JSON
        try:
            data = parse_markdown_file(os.path.join(args.downloaded_papers_dir, fname))
        except Exception as e:
            LOGGER.log(f"Error reading markdown file {fname}: {e}")
            exceptions[fname] = {"exception": f"Error reading file: {str(e)}"}
            continue

        with open(os.path.join(args.save_dir, parsedfiles_name), "a+") as f:
            f.write(fname + "\n")

        ########### prepare the prompt
        body = data["body"]
        abstract = data["abstract"]
        title = data["title"]
        
        paper = ""
        if len(title):
            paper += title + "\n\n"
        if len(abstract):
            paper += "Abstract:\n" + abstract + "\n\n"
        if len(body):
            paper += "Full text:\n" + body + "\n\n"

        tokens = tokenizer.encode(paper)
        user_prompt = (
            "\nHere is a paper:\n-- START OF PAPER --\n"
            + paper
            + "...\n-- END OF PAPER --\n"
        )

        if (args.max_tokens_body is not None) and (len(tokens) > args.max_tokens_body):
            to_cut = len(tokens) - args.max_tokens_body
            LOGGER.log(
                f"-------------> Paper too long ({len(tokens)} > {args.max_tokens_body} tokens), truncating the **FIRST** {to_cut} tokens\n"
            )
            paper = "..." + tokenizer.decode(tokens[to_cut:])
            user_prompt = (
                "\nHere is a paper (NOTE: the beginning of the paper is truncated because it was too long):\n-- START OF PAPER --\n"
                + paper
                + "...\n-- END OF PAPER --\n"
            )
        elif args.max_tokens_body is None:
            LOGGER.log("    WARNING: args.max_tokens_body is None!")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        llm = get_model_wrapper(model, tokenizer, save_name_stem, LOGGER, args.parsing_method)
        num_tokens_prompt = len(tokenizer.encode(system_prompt)) + len(tokenizer.encode(user_prompt))
        
        if args.n_ctx is not None:
            tokens_left = args.n_ctx - num_tokens_prompt
            LOGGER.log(
                f"Tokens of paper with prompt: {num_tokens_prompt} - Tokens left for model: {tokens_left}"
            )
        else:
            LOGGER.log(
                f"Tokens of paper with prompt: {num_tokens_prompt}"
            )

        ########### save current data and exceptions to csv
        if (datetime.now() - last_save_time).seconds > args.save_freq_minutes * 60:
            save_data(all_data, df_to_continue, exceptions, save_name_stem, args.save_dir, args.parsing_method)
            last_save_time = datetime.now()
            LOGGER.log("Saved data to csv...")

        start_time = datetime.now()

        ########### parse the paper
        try:
            if args.parsing_method == "guidance":
                out = parsing_func(llm, messages, LOGGER)
            elif args.parsing_method == "outlines":
                out = parsing_func(llm, messages, LOGGER)
            elif args.parsing_method == "unconstrained":
                out = parsing_func(llm, tokenizer, messages, LOGGER)
            else:
                out = parsing_func(llm, messages, LOGGER)
        except Exception as e:
            n_exceptions_streak += 1
            LOGGER.log(f"Exception: {e}")
            LOGGER.log(f"Stack trace:\n{traceback.format_exc()}")
            exceptions[fname] = {"exception": str(e), "prompt": messages[0]["content"] + messages[1]["content"]}
            if n_exceptions_streak > 30:
                LOGGER.log("Too many exceptions in a row, stopping...")
                break
            continue

        if out is None:
            continue

        ########### parse the model output
        try:
            if args.parsing_method == "unconstrained":
                if isinstance(out, tuple) and len(out) == 2:
                    code_content, saved_path = out
                else:
                    code_content = out
                    saved_path = "unknown"
                data = [{
                    "parsed_code": code_content,
                    "saved_filepath": saved_path,
                    "status": "success"
                }]
            else:
                data = json.loads(out)
        except Exception as e:
            LOGGER.log(f"Exception: {e}")
            LOGGER.log(f"Failed output content: {str(out)[:200]}...")
            exceptions[fname] = {"exception": str(e), "out": str(out)}
            if n_exceptions_streak > 10:
                LOGGER.log("Too many exceptions in a row, stopping...")
                break
            continue

        n_exceptions_streak = 0
        idx += 1

        ########### add metadata to dictionaries
        path = "file://" + os.path.abspath(
            os.path.join(args.downloaded_papers_dir, fname)
        )
        [d.update({"file": path}) for d in data]
        [d.update({"params": save_name_stem}) for d in data]
        all_data.append(data)

        ########### update time variables and projected finish time
        parsing_time = datetime.now() - start_time
        n_files += 1
        mean_time_per_file = (
            mean_time_per_file * (n_files - 1) + parsing_time.total_seconds()
        ) / n_files
        files_left = len(relevant_files) - idx
        projected_time_left = int(mean_time_per_file * files_left)
        projected_finish_time = (
            datetime.now() + timedelta(seconds=projected_time_left)
        ).strftime("%Y-%m-%d %H:%M:%S")
        LOGGER.log(f"Time taken: {parsing_time}")

    save_data(all_data, df_to_continue, exceptions, save_name_stem, args.save_dir, args.parsing_method)

if __name__ == "__main__":
    main()