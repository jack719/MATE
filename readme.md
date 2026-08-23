# MATE

## 数据集

论文主要来自dax-ti-alloy数据集提到的论文，使用开源方法MinerU处理为markdown

## 结构以及核心部分

入口：parse_downloaded_papers.py，该文件为启动文件，传入主要运行参数，并收集生成的csv

核心文件：src\freeform_parsing.py，包含方法的全部部分

配置文件：utils.py，读取输入等配置，目前配置允许处理markdown文件

## Quickstart

``````cmd
python parse_downloaded_papers.py
``````

