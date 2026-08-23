import os
import json
import pandas as pd

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy
from bs4 import BeautifulSoup
import openai
import torch

from src.logging import Logger

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


import certifi
def load_model_and_tokenizer(
    modelpath: str, hf_modelname: Optional[str] = None, api_key: Optional[str] = None, use_api: bool = False, **kwargs 
) -> Tuple[Any, Any]:
    """
    加载模型和tokenizer - 支持本地模型和API
    """
    if use_api:
        # API模式
        # 从环境变量或参数获取API密钥
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if api_key is None:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
        
        # 创建DeepSeek客户端
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            http_client=openai.DefaultHttpxClient(verify=certifi.where())
        )
        
        # 使用tiktoken作为tokenizer（兼容性更好）
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 compatible encoding
        except:
            # 如果tiktoken不可用，使用简单实现
            class SimpleTokenizer:
                def encode(self, text: str) -> list:
                    # 简单的token估算：大约4个字符 = 1个token
                    return list(range(len(text) // 4))
                
                def decode(self, tokens: list) -> str:
                    # 简化版本
                    return ""
            tokenizer = SimpleTokenizer()
        
        return client, tokenizer
    else:
        # 本地模型模式（保持原有逻辑）
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = hf_modelname if hf_modelname is not None else modelpath
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    #     # todo 1/11 应对gollie修改
    #     if tokenizer.chat_template is None:
    #         print("⚠️ 检测到缺少 Chat Template，正在手动应用 Llama-3 标准模板...")
    #         tokenizer.chat_template = (
    #             "{% set loop_messages = messages %}"
    #             "{% for message in loop_messages %}"
    #             "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    #             "{% if loop.index0 == 0 %}"
    #             "{% set content = bos_token + content %}"
    #             "{% endif %}"
    #             "{{ content }}"
    #             "{% endfor %}"
    #             "{% if add_generation_prompt %}"
    #             "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    #             "{% endif %}"
    #     )
    # # ===【插入这段修复代码 End】===








        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            **kwargs
        )
        
        if device == "cpu":
            model = model.to(device)
        
        return model, tokenizer

# DeepSeek API包装器
class DeepSeekWrapper:
    def __init__(self, client, model_name="deepseek-chat"):
        self.client = client
        self.model_name = model_name
        
    def generate(self, messages: list, max_tokens: int = 4000, temperature: float = 0.1, stop: Optional[list] = None) -> str:
        """
        调用DeepSeek API生成响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"DeepSeek API call failed: {str(e)}")


def save_data(all_data, df_to_continue, exceptions, save_name_stem, save_dir, parsing_method):
    create_df_func = get_create_df_func(parsing_method)
    csv_name = get_csv_name(save_name_stem)
    exceptions_name = get_exceptions_name(save_name_stem)
    try:
        df = create_df_func(all_data)
        df_all = pd.concat([df_to_continue, df], ignore_index=True)
        df_all = df_all[list(df.columns) + [c for c in df_all.columns if c not in df.columns]] # keep order of columns like in the original df
        df_all.to_csv(os.path.join(save_dir, csv_name), index=False)
        with open(os.path.join(save_dir, exceptions_name), "w+") as f:
            json.dump(exceptions, f)
    except:
        return None


def get_continue_csv_path(
    continue_csv_path: Optional[str], save_dir: str, continue_latest_same_params: str, save_name_stem: str, logger: Logger
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


def get_relevant_files(downloaded_papers_dir: str, file_extension: str = ".json") -> List[str]:
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


# def filter_relevant_files(
#     continue_training: bool,
#     relevant_files: List[str],
#     save_dir: str,
#     downloaded_papers_dir: str,
#     parsedfiles_name: str,
#     logger: Logger,
# ) -> List[str]:
#     len_before = len(relevant_files)
#     removed_already_parsed = 0
#     if continue_training:
#         if os.path.exists(os.path.join(save_dir, parsedfiles_name)):
#             with open(os.path.join(save_dir, parsedfiles_name), "r") as file:
#                 parsed_files = file.read().splitlines()

#             parsed_files = [el.strip() for el in parsed_files]
#             relevant_files = [f for f in relevant_files if f not in parsed_files]

#         removed_already_parsed = len_before - len(relevant_files)

#     for f in relevant_files:
#         with open(os.path.join(downloaded_papers_dir, f), "r") as file:
#             data = json.load(file)

#         if len(data.get("body", "")) < 100:
#             relevant_files.remove(f)

#     removed_empty_body = (
#         len_before - removed_already_parsed - len(relevant_files)
#     )
#     len_after = len(relevant_files)

#     logger.log(
#         f"Filtered out {len_before - len_after} files that were already checked "
#         f"({removed_already_parsed} files) or which had a text with less than 100 "
#         f"characters ({removed_empty_body} files)"
#     )

#     return relevant_files


def filter_relevant_files(
    continue_training: bool,
    relevant_files: List[str],
    save_dir: str,
    downloaded_papers_dir: str,
    parsedfiles_name: str,
    logger: Logger,
    file_extension: str = ".json",  # 新增参数
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

    # 修改这部分以支持 Markdown
    files_to_remove = []
    for f in relevant_files:
        file_path = os.path.join(downloaded_papers_dir, f)
        
        if file_extension == ".json":
            # JSON 文件处理逻辑(原有)
            with open(file_path, "r", encoding='utf-8') as file:
                data = json.load(file)
            if len(data.get("body", "")) < 100:
                files_to_remove.append(f)
        
        elif file_extension == ".md":
            # Markdown 文件处理逻辑(新增)
            try:
                with open(file_path, "r", encoding='utf-8') as file:
                    content = file.read()
                # 检查内容长度
                if len(content.strip()) < 100:
                    files_to_remove.append(f)
            except Exception as e:
                logger.log(f"Error reading {f}: {e}")
                files_to_remove.append(f)
    
    # 移除标记的文件
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


def get_benchmark_papers():
    """
    for testing new models, here are some papers which should contain relevant information
    """
    papers_with_info = [
        "10.1016/j.msea.2003.10.237",
        "10.1016/j.msea.2009.11.069",
        "10.1016/j.msea.2012.12.046",
        "10.1016/j.msea.2003.10.237",
        "10.1016/j.jallcom.2016.07.177",
        "10.1016/j.msea.2014.08.038",
        "10.1016/j.jmatprotec.2017.04.024",
        "10.1016/j.msea.2015.12.101",
        "10.1016/j.jmatprotec.2005.02.006",
        "10.1016/j.msea.2008.03.017",
        "10.1016/j.matlet.2013.08.093",
        "10.1016/j.jmatprotec.2008.04.020",
    ]
    papers_with_info = [p.replace("/", "_") + ".json" for p in papers_with_info]
    return papers_with_info


def process_html(html):
    # extract the text from the html, but keep the raw code for the tables
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    tables_copy = [deepcopy(t) for t in tables]

    for table in tables:
        table.insert_before(" PLACEHOLDER_TABLE ")
        table.decompose()

    text = soup.get_text()
    for table in tables_copy:
        text = text.replace(" PLACEHOLDER_TABLE ", table.prettify(), 1)

    return text

def get_system_prompt(parsing_method: str) -> str:
    if parsing_method == "guidance":
        from src.guidance_parsing import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    elif parsing_method == "unconstrained":
        from src.freeform_parsing import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_model_wrapper(model, tokenizer, save_name, logger, parsing_method: str) -> Any:
    if parsing_method == "guidance":
        from guidance import models
        # 对于 transformers 模型，需要使用不同的 guidance 模型包装器
        if "llama" in save_name.lower():
            from guidance.chat import Llama3ChatTemplate
            logger.log("Using Llama3ChatTemplate for guidance parsing.")
            return models.Transformers(model=model, tokenizer=tokenizer, echo=False, chat_template=Llama3ChatTemplate)
        else:
            from guidance.chat import ChatMLTemplate
            logger.log("Using ChatMLTemplate for guidance parsing.")
            return models.Transformers(model=model, tokenizer=tokenizer, echo=False, chat_template=ChatMLTemplate)
    elif parsing_method == "unconstrained":
        return model
    else:
        raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_parsing_func(parsing_method: str) -> Any:
    if parsing_method == "guidance":
        from src.guidance_parsing import parse_paper
        return parse_paper
    elif parsing_method == "unconstrained":
        from src.freeform_parsing import parse_paper
        return parse_paper
    else:
        raise ValueError(f"Unknown parsing method: {parsing_method}")

def get_create_df_func(parsing_method: str) -> Any:
    if parsing_method == "guidance":
        from src.guidance_parsing import build_df
        return build_df
    elif parsing_method == "unconstrained":
        from src.freeform_parsing import build_df
        return build_df
    else:
        raise ValueError(f"Unknown parsing method: {parsing_method}")