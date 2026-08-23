import argparse
import json
from pathlib import Path
from src.xml_cleaning import clean_xml_table


def process_json_file(src_path, target_path):
    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'tables' in data:
        for table_key, table_content in data['tables'].items():
            data['tables'][table_key] = clean_xml_table(table_content)
    
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description='Clean XML tables in JSON files')
    parser.add_argument('src', help='Source directory containing JSON files')
    parser.add_argument('target', help='Target directory for processed JSON files')
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    target_dir = Path(args.target)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for json_file in src_dir.glob('*.json'):
        target_file = target_dir / json_file.name
        process_json_file(json_file, target_file)
        print(f"Processed: {json_file.name}")


if __name__ == '__main__':
    main()
