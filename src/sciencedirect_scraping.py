import requests
import os
import time
import re
import json
from datetime import datetime

from typing import Dict, Literal, Optional, Union

import xml.etree.ElementTree as ET
import re

# create new error class for a case where the response status code is not 200
class CriticalElsevierError(Exception):
    def __init__(self, message):
        super().__init__(message)

def get_text_abstract_via_elsevier_api(doi, api_key, sleep_time=3) -> Optional[Dict[Literal["abstract", "body", "title", "tables"], Union[str, Dict[str, str]]]]:
    time.sleep(sleep_time)
    url = f"https://api.elsevier.com/content/article/doi/{doi}?apiKey={api_key}&httpAccept=text/xml"
    response = requests.get(url)

    headers = response.headers
    if "X-RateLimit-Remaining" in headers.keys():
        print(f"    Rate limit remaining: {headers['X-RateLimit-Remaining']}")

        reset_time_unix = response.headers["X-RateLimit-Reset"]
        reset_time = datetime.strftime(datetime.fromtimestamp(int(reset_time_unix)), format="%Y-%m-%d %H:%M:%S")
        if int(response.headers["X-RateLimit-Remaining"]) <= 0:
            print(f"no more api calls left. reset at {reset_time}")
            exit()

    if headers['X-ELS-Status'] not in ["OK", "RESOURCE_NOT_FOUND - The resource specified cannot be found."]:
        raise CriticalElsevierError(f"Unexpected response status: {headers['X-ELS-Status']} -> {response.text}")
    elif response.status_code not in [200,404]:
        raise CriticalElsevierError(f"Response status code is {response.status_code} -> {response.text}")
    elif response.status_code == 404:
        raise ValueError(f"Response status code is 404 -> {response.text}")

    # Parse the XML
    xml_string = response.text
    root = ET.fromstring(xml_string)

    # Define namespaces
    namespaces = {
        'ce': 'http://www.elsevier.com/xml/common/dtd',
        'ja': 'http://www.elsevier.com/xml/ja/dtd',
        'tb': 'http://www.elsevier.com/xml/common/table/dtd'
    }

    # Extract tables
    tables = {}
    for table in root.findall(".//ce:table", namespaces):
        label = table.find("ce:label", namespaces)
        if label is None:
            print("Warning: Table without label found...")
            label = ""
        else:
            label = label.text

        content = ET.tostring(table, encoding='unicode', method='xml')
        content = re.sub(r"\n\s+", "\n", content)
        tables[str(label)] = content

    # Extract title
    title = root.find(".//ce:title", namespaces)
    if title is not None:
        title = title.text
    else:
        print("Warning: No title found...")
        title = ""

    # Extract abstract
    abstract = root.find(".//ce:abstract-sec", namespaces)
    if abstract is not None:
        abstract_text = '\n'.join([el.strip() for el in abstract.itertext()])
        abstract_text: str = re.sub(r'\n+', r'\n', abstract_text).strip()
    else:
        print("Warning: No abstract found...")
        abstract_text = ""

    # Extract body
    body = root.find(".//ja:body", namespaces)
    if body is not None:
        body_text = '\n'.join([el.strip() for el in body.itertext()])
        body_text: str = re.sub(r'\n+', r'\n', body_text).strip()
    else:
        print("Warning: No body found...")
        body_text = ""

    return {"abstract": abstract_text, "body": body_text, "title": title, "tables": tables}

def add_doi_to_no_results(doi, save_dir):
    if not os.path.exists(f"{save_dir}/no_results.json"):
        with open(f"{save_dir}/no_results.json", "w+") as f:
            json.dump({"doi": [doi]}, f)
    else:
        with open(f"{save_dir}/no_results.json", "r") as f:
            data = json.load(f)
        if doi not in data["doi"]:
            data["doi"].append(doi)
            with open(f"{save_dir}/no_results.json", "w") as f:
                json.dump(data, f)
