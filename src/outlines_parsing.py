
class Cols:
    """
    column/key names for the dict/dataframe
    """
    ys = "YS"
    ys_unit = "YS (unit)"
    uts = "UTS"
    uts_unit = "UTS (unit)"
    elong = "elong"
    elong_unit = "elong (unit)"
    temper_designation = "temper designation"
    other_info = "other info"
    alloy_designation = "alloy designation"
    manufacturing_type = "manufacturing type"
    heat_treatment_info = "heat treatment/aging info"
    testing_temperature = "testing temperature"


SYSTEM_PROMPT = f"""You are a paper information extractor. Given a paper about alloys, extract the chemical composition of the used alloys and ONLY the following 3 mechanical properties: yield strength, ultimate tensile strength and elongation. Do not extract any other mechanical properties! Also include the temper designation, heat treatment and cooling specifications, as well as the testing temperature under which the results were obtained. Save results in json format. If no relevant information about atleast chemical composition and mechanichal properties is found in the paper or the paper is not about alloys, just make the json empty. 

One chemical composition can have multiple measurements of mechanical properties with different test setup, heat treatment, and so on. So each different chemical composition can have multiple mechanical properties dictionaries. Include any not yet mentioned information that led to a given mechanical property dictionary as "other info"! (e.g. the type of sample or some novel sample treatment techniques!)

upfront, here are the keys that will be present in the json with a short explanation:
- {Cols.alloy_designation}: the designation of the alloy as stated in the paper
- chemical composition: cotnains a sub-dictionary, where the keys are the names of the elements and the values are lists with the first element being the value and the second element the unit, e.g. "Mg": ["2", "wt %"]
- mechanical properties: a list of dictionaries (one for each different result/test with the given alloy) containing the following keys:
    - {Cols.ys}: yield strength, if stated in the paper. should be a list with the first element being the value and the second the unit, e.g.: ["150.2", "MPa"]
    - {Cols.uts}: ultimate tensile strength, if stated in the paper. should be a list with the first element being the value and the second the unit, e.g.: ["210.3", "MPa"]
    - {Cols.elong}: elongation, if stated in the paper. should be a list with the first element being the value and the second the unit, e.g.: ["2.1", "%"]
    - {Cols.temper_designation}: temper designation, if explicitly stated in the paper. if not, put in "null"
    - {Cols.manufacturing_type}: is it for example a wrought alloy, cast alloy, sintered alloy, etc. if not stated, put "null"
    - {Cols.heat_treatment_info}: heat treatment/aging information. this is a list of dictionaries, where each dictionary contains the keys "time", "temperature" and "cooling method". Values for "time" and "temperature" should be lists where the first value is the time/temperature and the second is the unit, e.g.: ["2", "hours"]. if not stated, put ["null", "null"]. Cooling method is a string, e.g. "water quenching". if no cooling method was applied or it is not stated, put "null". If temperature is stated e.g. as "room temperature" in the text, put in the corresponding numerical value, i.e. ["20", "C"]. example: an alloy was heat treted first at 100°C for 2 hours, then at 180°C for 2 hours, and air cooling applied after the last step: [{{"time": ["2", "hours"], "temperature": ["100", "C"], "cooling method": "null"}}, {{"time": ["2", "hours"], "temperature": ["180", "C"], "cooling method": "air cooling"}}]. Another example: an alloy was 
    - {Cols.testing_temperature}: temperature at the time of epxeriment/measurement. always use numeric values, e.g. if room temperature is stated, put ["20", "C"]. should be a list where the first value is the temperature and the second is the unit, e.g.: ["100", "C"]. if not stated, put "null"
    - {Cols.other_info}: any other information that you deem relevant for the given mechanical property that has not been mentioned, e.g. the shape of sample, the testing conditions, etc. if not stated, put "null"

Here is an example of the desired json format. This example is from a paper where two different alloys are described, and the second was tested under 3 different times using different heat treatment durations:
```json
[
    {{
        "{Cols.alloy_designation}": "Al-0.88Ag-2.0C-0.5Fe",
        "chemical composition": {{
            "Ag": ["0.88", "wt %"],
            "C": ["2.0", "wt %"],
            "Fe": ["0.5", "wt %"],
            "Al": ["Bal.", "wt %"]
        }}, "mechanical properties": [
            {{
                "{Cols.ys}": ["150.2", "MPa"],
                "{Cols.uts}": ["210.3", "MPa"],
                "{Cols.elong}": ["null", "null"],
                "{Cols.temper_designation}": "H6",
                "{Cols.manufacturing_type}": "wrought",
                "{Cols.heat_treatment_info}": [{{"time": ["2", "hours"], "temperature": ["200", "C"], "cooling method": "water quenching"}}],
                "{Cols.testing_temperature}": ["50", "C"],
                "{Cols.other_info}": "null"
            }}
        ]
    }},
    {{
        "{Cols.alloy_designation}": "A6061",
        "chemical composition": {{
            "Mg": ["0.9", "wt %"],
            "Si": ["0.6", "wt %"],
            "Cu": ["0.15", "wt %"],
            "Ti": ["50", "ppm"],
            "Al": ["Bal.", "wt %"]
        }}, "mechanical properties": [
            {{
               "{Cols.ys}": ["170.6", "MPa"],
               "{Cols.uts}": ["190.5", "MPa"],
               "{Cols.elong}": ["null", "null"],
               "{Cols.temper_designation}": "T10",
               "{Cols.manufacturing_type}": "cast",
               "{Cols.heat_treatment_info}": [{{"time": ["2", "hours"], "temperature": ["200", "C"], "cooling method": "water quenching"}}],
               "{Cols.testing_temperature}": ["50", "C"],
               "{Cols.other_info}": "null"
            }},
            {{
               "{Cols.ys}": ["143.4", "MPa"],
               "{Cols.uts}": ["173.5", "MPa"],
               "{Cols.elong}": ["2.1", "%"],
               "{Cols.temper_designation}": "T8",
               "{Cols.manufacturing_type}": "cast",
               "{Cols.heat_treatment_info}": [{{"time": ["5", "hours"], "temperature": ["200", "C"], "cooling method": "water quenching"}}],
               "{Cols.testing_temperature}": ["50", "C"],
               "{Cols.other_info}": "null"
            }},
            {{
               "{Cols.ys}": ["139.5", "MPa"],
               "{Cols.uts}": ["301.4", "MPa"],
               "{Cols.elong}": ["7.2", "%"],
               "{Cols.temper_designation}": "T8",
               "{Cols.manufacturing_type}": "cast",
               "{Cols.heat_treatment_info}": [{{"time": ["10", "hours"], "temperature": ["200", "C"], "cooling method": "water quenching"}}],
               "{Cols.testing_temperature}": ["50", "C"],
               "{Cols.other_info}": "null"
            }}
        ]
    }}
]
```

Here is an example of a paper where only the ultimate tensile strength is given for only one alloy in one test, and the chemical composition in the paper was stated as Al-2Mg-0.5Si-0.22Fe-1.25Mn. Also, no temper designation was explicitly stated in the paper and thus one was assigned based on the information given in the paper:
```json
[
    {{
        "{Cols.alloy_designation}": "Al-2Mg-0.5Si-0.22Fe-1.25Mn",
        "chemical composition": {{
            "Mg": ["2", "wt %"],
            "Si": ["0.55", "wt %"],
            "Fe": ["0.22", "wt %"],
            "Mn": ["1.25", "wt %"],
            "Al": ["Bal.", "wt %"]
        }}, "mechanical properties": [
            {{
                "{Cols.ys}": ["null", "null"],
                "{Cols.uts}": ["300.0", "MPa"],
                "{Cols.elong}": ["null", "null"],
                "{Cols.temper_designation}": "null",
                "{Cols.manufacturing_type}": "wrought",
                "{Cols.heat_treatment_info}": [{{"time": ["null", "null"], "temperature": ["null", "null"], "cooling method": "null"}}],
                "{Cols.testing_temperature}": ["null", "null"],
                "{Cols.other_info}": "The alloy was tested in a cylindrical shape."
            }}
        ]
    }}
]
```

In the following example a 3-step aging process was applied, where the sample was first heat treated for 2 hours at 200 C and then aged for 5 hours at 250 C and then naturally aged (i.e. at room temperature) for 10 days:
```json
[
    {{
        "{Cols.alloy_designation}": "Al-2Si-1.5Mg",
        "chemical composition": {{
            "Al": ["Bal.", "wt %"],
            "Si": ["2", "wt %"],
            "Mg": ["1.5", "wt %"]
        }}, "mechanical properties": [
            {{
                "{Cols.ys}": ["250.2", "MPa"],
                "{Cols.uts}": ["310.3", "MPa"],
                "{Cols.elong}": ["null", "null"],
                "{Cols.temper_designation}": "T4",
                "{Cols.manufacturing_type}": "null",
                "{Cols.heat_treatment_info}": [{{"time": ["2", "hours"], "temperature": ["200", "C"], "cooling method": "null"}}, {{"time": ["5", "hours"], "temperature": ["250", "C"], "cooling method": "air cooling"}}, {{"time": ["10", "days"], "temperature": ["20", "C"], "cooling method": "null"}}],
                "{Cols.testing_temperature}": ["20", "C"],
                "{Cols.other_info}": "null"
            }}
        ]
    }}
]
```

Make sure to correctly extract the chemical composition of the alloy. It could be given in a table or in free text, or a combination of both. It could also for example be stated as e.g. Al-2Si-0.5Mg, or Al-2%Si-0.5%Mg, or Al2Si0.5Mg, which would correspond to the following dictionary:
```json
"chemical composition": {{
    "Al": ["Bal.", "wt %"],
    "Si": ["2", "wt %"],
    "Mg": ["0.5", "wt %"]
}}
```

Or for example Al–6Mg–9Si–8Cu–7Zn–3Ni would correspond to:
```json
"chemical composition": {{
    "Al": ["Bal.", "wt %"],
    "Mg": ["6", "wt %"],
    "Si": ["9", "wt %"],
    "Cu": ["8", "wt %"],
    "Zn": ["7", "wt %"],
    "Ni": ["3", "wt %"]
}}
```

Sometimes also ranges are stated in the paper, e.g.:
```json
"chemical composition": {{
    "Al": ["Bal.", "wt %"],
    "Si": ["2.0-2.5", "wt %"],
    "Mg": ["1.5-3.0", "wt %"]
}}
```

Sometimes no clear composition is found in the paper, e.g.:
```json
"chemical composition": {{
    "null": ["null", "null"],
}}
```

The "Bal."-Element could also be stated as "Rest", or "Remaining", or "rem", etc., but we always write it as "Bal." in the json.

Important: Only state the chemical composition if the exact numbers are given in the paper! If the paper only states that the alloy contains e.g. Si and Mg, but not the amount as a number, then do not include the chemical composition in the json!
We only want to extract information that is explicitly stated in the paper, and we only want to extract information for alloys where the chemical composition AS WELL AS one of the measurements (yield strength, ultimate tensile strength, or elongation) is given in the paper. If there is no information about the chemical composition or any of these measurements in the paper, just write an empty json.
If there's no information about some fields (e.g. {Cols.ys}), just insert "null" as value and unit.
Do not make up any information, if something is not stated in the paper, put "null".
Values should always be strings!
Make sure that the results you put in actually match the key! I.e. only put an actual yield strength value in the "{Cols.ys}" field! Do NOT for example insert a hardness result under "{Cols.ys}"!
Make sure to include ALL chemical elements contained in the alloy in the chemical composition!
"""

MAX_ITEMS_CHEMICAL = 30
JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            f"{Cols.alloy_designation}": {"type": "string"},
            "chemical composition": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2 
                },
                "maxProperties": MAX_ITEMS_CHEMICAL
            },
            "mechanical properties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        f"{Cols.ys}": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        f"{Cols.uts}": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        f"{Cols.elong}": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        f"{Cols.temper_designation}": {"type": "string"},
                        f"{Cols.manufacturing_type}": {"type": "string"},
                        f"{Cols.heat_treatment_info}": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                                    "temperature": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                                    "cooling method": {"type": "string"}
                                },
                                "required": ["time", "temperature", "cooling method"]
                            },
                            "minItems": 1
                        },
                        f"{Cols.testing_temperature}": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        f"{Cols.other_info}": {"type": "string"}
                    },
                    "required": [
                        f"{Cols.ys}",
                        f"{Cols.uts}",
                        f"{Cols.elong}",
                        f"{Cols.temper_designation}",
                        f"{Cols.manufacturing_type}",
                        f"{Cols.heat_treatment_info}",
                        f"{Cols.testing_temperature}",
                        f"{Cols.other_info}"
                    ]
                }
            }
        },
        "required": [
            f"{Cols.alloy_designation}",
            "chemical composition",
            "mechanical properties"
        ]
    }
}

raise NotImplementedError("""Outlines doesn't seem to work with our long prompts and complex schema's. Here an example which does not work as of Sep. 2024:

import os
import json
import outlines
import torch
from outlines import models
from llama_cpp import Llama
import llama_cpp

gpu_id = 1

modelpath="/home/pichlmannl/models/Meta-Llama-3.1-70B-Instruct-Q8_0-00001-of-00002.gguf"
llama = Llama(
    model_path=modelpath,
    n_gpu_layers=-1,
    gpu_id=1,
    n_ctx=50000,
    temperature=0.0,
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
)
llama.verbose = False

paper = "10.1016_j.ijfatigue.2020.105951"
fpath = os.path.join("./downloaded_papers_old", paper + ".json")

with open(fpath, "r") as f:
    data = json.load(f)

text = data["body"]

#############
# this works:

out = llama(SYSTEM_PROMPT + "\n\nHere is the paper:\n" + text + "\n\n\nYour json here:\n", max_tokens=4096)
len(out["choices"])
import json
generated = out["choices"][0]["text"]
start_json = generated.find("{")
end_json = generated.rfind("}") + 1
generated = generated[start_json:end_json]
data = json.loads(generated)

###############
# this doesn't:

@outlines.prompt
def build_prompt(prompt, paper):
    \"""
    {{ prompt }}

    Here is the paper:

    -PAPER_START-
    {{ paper }}
    -PAPER_END-

    Your json here:
    \"""

model = models.LlamaCpp(llama)
from src.outlines_parsing import Cols, SYSTEM_PROMPT, JSON_SCHEMA
json_schema = JSON_SCHEMA
schema_str = json.dumps(json_schema)
generator = outlines.generate.json(model, schema_str)

generator = outlines.generate.json(model, schema_str)
output = generator(build_prompt(SYSTEM_PROMPT, text))
output
"""
)
