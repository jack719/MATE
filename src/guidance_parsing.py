import guidance
import re
import pandas as pd
from typing import List, Dict, Optional
from guidance import models, gen, select, user, system, assistant
import json

from src.logging import Logger

# TODO: incorporate the temperature from the cli args

MAX_TOKENS_FIELDS = 50
RELEVANT_INFORMATION_ADDED = True
CHEM_VALS_PREV = []


gen_limited = lambda *args, **kwargs: (
    gen(*args, max_tokens=MAX_TOKENS_FIELDS, stop_regex=r'["}\]]', **kwargs)
    if not (kwargs.get("stop") or kwargs.get("stop_regex"))
    else gen(*args, max_tokens=MAX_TOKENS_FIELDS, **kwargs)
)


class Cols:
    """
    Column/key names for the dict/dataframe - Titanium Alloy Specific
    """
    # Chemical composition (13 elements)
    ti = "Ti (wt%)"
    mo = "Mo (wt%)"
    al = "Al (wt%)"
    sn = "Sn (wt%)"
    v = "V (wt%)"
    zr = "Zr (wt%)"
    cr = "Cr (wt%)"
    nb = "Nb (wt%)"
    ta = "Ta (wt%)"
    fe = "Fe (wt%)"
    w = "W (wt%)"
    si = "Si (wt%)"
    o = "O (wt%)"
    
    # Transition temperature
    transition_temp = "transition temperature (°C)"
    
    # Condition and Process
    condition = "Condition"
    process = "Process"
    
    # Mechanical properties
    strength = "Strength (MPa)"
    elongation = "Elongation (%)"
    
    # Process parameters
    forging_temp = "Forging temperature (°C)"
    processing_temp = "Processing temperature (°C)"
    deformation = "Deformation amount (%)"
    solution_temp = "solution temperature (°C)"
    solution_time = "time (h)"
    solution_cooling = "cooling method"
    second_solution_temp = "second solution (°C)"
    second_solution_time = "time.1 (h)"
    second_solution_cooling = "cooling method.1"
    aging_temp = "ageing temperature (°C)"
    aging_time = "time.2 (h)"
    aging_cooling = "cooling method.2"
    second_aging_temp = "second ageing temperature (°C)"
    second_aging_time = "time.3 (h)"
    second_aging_cooling = "cooling method.3"
    annealing_temp = "annealing temperature (°C)"
    annealing_time = "time.4 (h)"
    annealing_cooling = "cooling method.4"
    second_annealing_temp = "second annealing temperature (°C)"
    second_annealing_time = "time.5 (h)"
    second_annealing_cooling = "cooling method.5"
    
    # Meta
    alloy_designation = "alloy designation"
    remarks = "remarks"


# Cooling method encoding
COOLING_METHODS = {
    "none": 0,
    "no cooling": 0,
    "null": 0,
    "oil quench": 1,
    "oq": 1,
    "oil": 1,
    "air cool": 2,
    "ac": 2,
    "air": 2,
    "furnace cool": 3,
    "fc": 3,
    "furnace": 3,
    "water quench": 4,
    "wq": 4,
    "water": 4
}

# Condition labels
CONDITION_LABELS = [
    "STA-OQ", "STA-AC", "STA-FC", "STA-WQ",
    "ST-OQ", "ST-AC", "ST-FC", "ST-WQ",
    "Direct-Aged", "Annealed-AC", "Annealed-WQ",
    "As-forged", "Unknown"
]


def parse_paper(llm: models.LlamaCpp, messages: List[Dict[str,str]], logger: Logger) -> Optional[str]:
    assert messages[0]["role"] == "system", "First prompt must be system prompt"
    assert messages[1]["role"] == "user", "Second prompt must be a user prompt"
    assert len(messages) == 2, "Expecting a system and a user prompt"

    with system():
        lm = llm + messages[0]["content"]

    with user():
        lm += (
            messages[1]["content"]
            + "\nBefore creating the JSON, tell me whether there is any relevant information to be extracted from the given paper.\n"
            + "Are there tensile strength (UTS) or elongation measurements for titanium alloys stated in the paper? (yes/no) -> "
        )

    with assistant():
        lm += select(["yes", "no"], name="has_mech_prop")

    if lm["has_mech_prop"] == "no":
        logger.log("    No mechanical properties found in the paper, skipping...")
        return None

    with user():
        lm += """Is there a chemical composition for the tested titanium alloys stated in the paper with exact numbers (for example "Ti-6Al-4V", or as a table, or described in text)? (yes/no) -> """

    with assistant():
        lm += select(["yes", "no"], name="has_chem_comp")

    if lm["has_chem_comp"] == "no":
        logger.log("    No chemical composition found in the paper, skipping...")
        return None

    with user():
        lm += "Alright, before we put the information into json, let's go through all relevant information that we need to extract from the paper step by step. Answer the following questions very concisely (do not add information that was not asked in the question!) and then wait for my next instruction."
        lm += "\n -- Collection of relevant information start --"
        lm += """Q1: First, tell me which titanium alloys occurred in the paper: write down their alloy designations (if any, otherwise write 'null') A1:"""
    
    with assistant():
        lm += gen(max_tokens=1000, name="alloy_designations")

    with user():
        lm += "Q2: For each of these alloys, tell me the chemical composition (Ti, Mo, Al, Sn, V, Zr, Cr, Nb, Ta, Fe, W, Si, O contents in wt%), if mentioned. If not mentioned, just write 'null' \nA2:"

    with assistant():
        lm += gen(max_tokens=1000, name="chemical_compositions")

    with user():
        lm += "Q3: For each of these alloys, tell me the beta transus temperature (transition temperature) if mentioned. If not stated, write 'null' \nA3:"

    with assistant():
        lm += gen(max_tokens=500, name="transition_temperatures")

    with user():
        lm += "Q4: For each of these alloys, tell me all measurements of tensile strength (UTS) and elongation that were mentioned in the paper (if not stated, write 'null') \nA4:"

    with assistant():
        lm += gen(max_tokens=1000, name="mechanical_properties")

    with user():
        lm += "Q5: For each measurement, tell me the heat treatment process details including: forging temperature, processing temperature, deformation amount, solution treatment (temperature, time, cooling method), aging treatment (temperature, time, cooling method), annealing (temperature, time, cooling method). If any of these are not stated, write 'null'. Also identify the condition label (like STA-OQ, ST-AC, As-forged, etc.) \nA5:"

    with assistant():
        lm += gen(max_tokens=1500, name="process_details")

    with user():
        lm += "-- Collection of relevant information end --\nOkay, now create a json with all of this information. Make sure to only add an alloy entry if both the chemical composition as well as **at least one of** strength or elongation were stated. Start creating the json now:"

    # Log all the summarized information
    logger.log(f"    \n-------\nAlloy designations:\n-------\n{lm['alloy_designations']}")
    logger.log(f"    \n-------\nChemical compositions:\n-------\n{lm['chemical_compositions']}")
    logger.log(f"    \n-------\nTransition temperatures:\n-------\n{lm['transition_temperatures']}")
    logger.log(f"    \n-------\nMechanical properties:\n-------\n{lm['mechanical_properties']}")
    logger.log(f"    \n-------\nProcess details:\n-------\n{lm['process_details']}")

    with assistant():
        lm += "\n```json\n["
        lm += build_alloy_json(
            verbosity=0,
            max_count_alloys=30,
            max_count_chemical=13,  # Fixed 13 elements
            max_count_mechanical=30,
            logger=logger,
        )

    max_corrections = 2
    n_corrections = 0
    while True:
        if n_corrections == 0:
            with user():
                lm += "Great! Now review the json you created (make sure there are no hallucinations - everything must be actually stated in the given paper). Is there anything you would like to add, remove or correct? (yes/no) -> "
        else:
            with user():
                lm += f"Okay, then please correct the mistake. Note that some of your outputs were restricted using controlled generation. Now please provide the new json:"

            with assistant():
                lm += "\n```json\n["
                lm += build_alloy_json(
                    verbosity=0,
                    max_count_alloys=30,
                    max_count_chemical=13,
                    max_count_mechanical=30,
                    logger=logger,
                )

            with user():
                lm += "Again, review this latest corrected json. Is there still something you would like to add, remove or correct? (yes/no) -> "

        with assistant():
            lm += select(["yes", "no"], name="erroneous_json")

        if lm["erroneous_json"] == "no":
            break

        with user():
            lm += "\nIn short, why would you like to correct it? What is wrong with it? (do NOT give me the corrected json yet, just quickly tell me what you think is wrong with it)"

        with assistant():
            lm += gen(max_tokens=1000, name="correction_reason")

        logger.log(f"    Json needs correction. Reason: {lm['correction_reason']}")

        if n_corrections >= max_corrections:
            out = str(lm)
            out = out.split("```json")[-1].strip(" `\n")
            out = re.sub(r",\s*}", "}", out)
            out = re.sub(r",\s*]", "]", out)
            out = re.sub(r"\][^\]]*$", "]", out)
            logger.error(f"Max number of corrections reached ({max_corrections}). Latest json: {out}")
            break

        n_corrections += 1

    if n_corrections >= max_corrections:
        print("Max number of corrections reached. Trying one last time with free text input.")
        with user():
            lm += "Okay, you'll get one more chance to correct the json. Here you go:"

        with assistant():
            lm += "\n```json\n[" + gen(max_tokens=8000, name="json") + "\n"

        with user():
            lm += "Review the json you created. Is there anything you would like to add, remove or correct? (yes/no) -> "

        with assistant():
            lm += select(["yes", "no"], name="erroneous_json")

        if lm["erroneous_json"] == "yes":
            with user():
                lm += "In short, why would you like to correct it?"

            with assistant():
                lm += gen(max_tokens=1000, name="correction_reason")

            logger.log(f"    Json still not correct after last attempt. Reason: {lm['correction_reason']}")

            try:
                out = str(lm)
                out = out.split("```json")[-1].strip(" `\n")
                out = re.sub(r",\s*}", "}", out)
                out = re.sub(r",\s*]", "]", out)
                out = re.sub(r"\][^\]]*$", "]", out)
            except Exception as e:
                logger.error(f"parsing error after last correction attempt: {e}")
                logger.error("..." + str(lm)[:5000:])
                out = "< parsing error >"

            raise Exception(f"Last correction attempt unsuccessful. Last json: {out}")

    logger.log(f"    JSON created and confirmed, number of corrections: {n_corrections}")

    out = str(lm)
    out = out.split("```json")[-1].strip(" `\n")
    out = re.sub(r",\s*}", "}", out)
    out = re.sub(r",\s*]", "]", out)
    out = re.sub(r"\][^\]]*$", "]", out)

    return out


def encode_cooling_method(cooling_str: str) -> int:
    """Encode cooling method string to numeric code"""
    if not cooling_str or cooling_str == "null":
        return 0
    cooling_lower = cooling_str.lower().strip()
    return COOLING_METHODS.get(cooling_lower, 0)


def build_process_text(row: Dict) -> str:
    """Build process description text from row data"""
    parts = []
    
    # Forging temperature
    if row.get(Cols.forging_temp) and row[Cols.forging_temp] != "null":
        parts.append(f"Forging temperature: {row[Cols.forging_temp]}°C.")
    
    # Processing temperature
    if row.get(Cols.processing_temp) and row[Cols.processing_temp] != "null":
        parts.append(f"Processing temperature: {row[Cols.processing_temp]}°C.")
    
    # Deformation
    if row.get(Cols.deformation) and row[Cols.deformation] != "null":
        parts.append(f"Deformation amount: {row[Cols.deformation]}%.")
    
    # Solution treatment
    if row.get(Cols.solution_temp) and row[Cols.solution_temp] != "null":
        temp = row[Cols.solution_temp]
        time = row.get(Cols.solution_time, "null")
        cooling = row.get(Cols.solution_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Solution: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Solution: {temp}°C for {time}h.")
        else:
            parts.append(f"Solution: {temp}°C.")
    
    # Second solution
    if row.get(Cols.second_solution_temp) and row[Cols.second_solution_temp] != "null":
        temp = row[Cols.second_solution_temp]
        time = row.get(Cols.second_solution_time, "null")
        cooling = row.get(Cols.second_solution_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Second Solution: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Second Solution: {temp}°C for {time}h.")
        else:
            parts.append(f"Second Solution: {temp}°C.")
    
    # Aging
    if row.get(Cols.aging_temp) and row[Cols.aging_temp] != "null":
        temp = row[Cols.aging_temp]
        time = row.get(Cols.aging_time, "null")
        cooling = row.get(Cols.aging_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Aging: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Aging: {temp}°C for {time}h.")
        else:
            parts.append(f"Aging: {temp}°C.")
    
    # Second aging
    if row.get(Cols.second_aging_temp) and row[Cols.second_aging_temp] != "null":
        temp = row[Cols.second_aging_temp]
        time = row.get(Cols.second_aging_time, "null")
        cooling = row.get(Cols.second_aging_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Second Aging: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Second Aging: {temp}°C for {time}h.")
        else:
            parts.append(f"Second Aging: {temp}°C.")
    
    # Annealing
    if row.get(Cols.annealing_temp) and row[Cols.annealing_temp] != "null":
        temp = row[Cols.annealing_temp]
        time = row.get(Cols.annealing_time, "null")
        cooling = row.get(Cols.annealing_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Annealing: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Annealing: {temp}°C for {time}h.")
        else:
            parts.append(f"Annealing: {temp}°C.")
    
    # Second annealing
    if row.get(Cols.second_annealing_temp) and row[Cols.second_annealing_temp] != "null":
        temp = row[Cols.second_annealing_temp]
        time = row.get(Cols.second_annealing_time, "null")
        cooling = row.get(Cols.second_annealing_cooling, "null")
        if time != "null" and cooling != "null":
            parts.append(f"Second Annealing: {temp}°C for {time}h, cooled by {cooling}.")
        elif time != "null":
            parts.append(f"Second Annealing: {temp}°C for {time}h.")
        else:
            parts.append(f"Second Annealing: {temp}°C.")
    
    return " ".join(parts)


def build_df(data, *_):
    df = pd.DataFrame()
    for list_of_dict in data.copy():
        for dict_ in list_of_dict:
            try:
                chem_comps = dict_["chemical composition"]
                mech_props = dict_["mechanical properties"]
                file = dict_["file"]
                params = dict_["params"]
                alloy_designation = dict_[Cols.alloy_designation]
                remarks = dict_[Cols.remarks]
                transition_temp = dict_.get(Cols.transition_temp, "null")

                if not len(chem_comps):
                    continue

                if not len(mech_props):
                    continue

                # Chemical composition (13 fixed elements)
                row_chem_comp = {}
                element_map = {
                    "Ti": Cols.ti, "Mo": Cols.mo, "Al": Cols.al, "Sn": Cols.sn,
                    "V": Cols.v, "Zr": Cols.zr, "Cr": Cols.cr, "Nb": Cols.nb,
                    "Ta": Cols.ta, "Fe": Cols.fe, "W": Cols.w, "Si": Cols.si, "O": Cols.o
                }
                
                for elem, col_name in element_map.items():
                    if elem in chem_comps and chem_comps[elem] != "null":
                        try:
                            row_chem_comp[col_name] = float(chem_comps[elem])
                        except:
                            row_chem_comp[col_name] = None
                    else:
                        row_chem_comp[col_name] = None

                rows = []
                for mprop in mech_props:
                    # Extract mechanical properties
                    strength = mprop.get(Cols.strength, "null")
                    elongation = mprop.get(Cols.elongation, "null")
                    
                    if strength == "null" and elongation == "null":
                        continue
                    
                    row = {}
                    
                    # Add chemical composition
                    row.update(row_chem_comp)
                    
                    # Add transition temperature
                    try:
                        row[Cols.transition_temp] = float(transition_temp) if transition_temp != "null" else None
                    except:
                        row[Cols.transition_temp] = None
                    
                    # Add mechanical properties
                    try:
                        row[Cols.strength] = float(strength) if strength != "null" else None
                    except:
                        row[Cols.strength] = None
                    
                    try:
                        row[Cols.elongation] = float(elongation) if elongation != "null" else None
                    except:
                        row[Cols.elongation] = None
                    
                    # Add process parameters
                    process_params = {
                        Cols.forging_temp: mprop.get(Cols.forging_temp, "null"),
                        Cols.processing_temp: mprop.get(Cols.processing_temp, "null"),
                        Cols.deformation: mprop.get(Cols.deformation, "null"),
                        Cols.solution_temp: mprop.get(Cols.solution_temp, "null"),
                        Cols.solution_time: mprop.get(Cols.solution_time, "null"),
                        Cols.solution_cooling: mprop.get(Cols.solution_cooling, "null"),
                        Cols.second_solution_temp: mprop.get(Cols.second_solution_temp, "null"),
                        Cols.second_solution_time: mprop.get(Cols.second_solution_time, "null"),
                        Cols.second_solution_cooling: mprop.get(Cols.second_solution_cooling, "null"),
                        Cols.aging_temp: mprop.get(Cols.aging_temp, "null"),
                        Cols.aging_time: mprop.get(Cols.aging_time, "null"),
                        Cols.aging_cooling: mprop.get(Cols.aging_cooling, "null"),
                        Cols.second_aging_temp: mprop.get(Cols.second_aging_temp, "null"),
                        Cols.second_aging_time: mprop.get(Cols.second_aging_time, "null"),
                        Cols.second_aging_cooling: mprop.get(Cols.second_aging_cooling, "null"),
                        Cols.annealing_temp: mprop.get(Cols.annealing_temp, "null"),
                        Cols.annealing_time: mprop.get(Cols.annealing_time, "null"),
                        Cols.annealing_cooling: mprop.get(Cols.annealing_cooling, "null"),
                        Cols.second_annealing_temp: mprop.get(Cols.second_annealing_temp, "null"),
                        Cols.second_annealing_time: mprop.get(Cols.second_annealing_time, "null"),
                        Cols.second_annealing_cooling: mprop.get(Cols.second_annealing_cooling, "null"),
                    }
                    
                    # Convert numeric process parameters
                    for key in [Cols.forging_temp, Cols.processing_temp, Cols.deformation,
                               Cols.solution_temp, Cols.solution_time, Cols.second_solution_temp,
                               Cols.second_solution_time, Cols.aging_temp, Cols.aging_time,
                               Cols.second_aging_temp, Cols.second_aging_time, Cols.annealing_temp,
                               Cols.annealing_time, Cols.second_annealing_temp, Cols.second_annealing_time]:
                        try:
                            row[key] = float(process_params[key]) if process_params[key] != "null" else None
                        except:
                            row[key] = None
                    
                    # Encode cooling methods
                    row[Cols.solution_cooling] = encode_cooling_method(process_params[Cols.solution_cooling])
                    row[Cols.second_solution_cooling] = encode_cooling_method(process_params[Cols.second_solution_cooling])
                    row[Cols.aging_cooling] = encode_cooling_method(process_params[Cols.aging_cooling])
                    row[Cols.second_aging_cooling] = encode_cooling_method(process_params[Cols.second_aging_cooling])
                    row[Cols.annealing_cooling] = encode_cooling_method(process_params[Cols.annealing_cooling])
                    row[Cols.second_annealing_cooling] = encode_cooling_method(process_params[Cols.second_annealing_cooling])
                    
                    # Add Condition label
                    row[Cols.condition] = mprop.get(Cols.condition, "Unknown")
                    
                    # Build Process text
                    row[Cols.process] = build_process_text(process_params)
                    
                    # Add meta data
                    row["file"] = file
                    row["params"] = params
                    row[Cols.alloy_designation] = alloy_designation
                    row[Cols.remarks] = remarks
                    
                    rows.append(row)

                if not len(rows):
                    continue

                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue

    return sort_columns(df)


def sort_columns(df) -> pd.DataFrame:
    """Sort columns according to specification"""
    # Chemical composition columns (13 elements)
    chem_cols = [
        Cols.ti, Cols.mo, Cols.al, Cols.sn, Cols.v, Cols.zr, Cols.cr,
        Cols.nb, Cols.ta, Cols.fe, Cols.w, Cols.si, Cols.o
    ]
    
    # Property columns
    prop_cols = [
        Cols.transition_temp,
        Cols.condition,
        Cols.process,
        Cols.strength,
        Cols.elongation
    ]
    
    # Process parameter columns
    process_cols = [
        Cols.forging_temp, Cols.processing_temp, Cols.deformation,
        Cols.solution_temp, Cols.solution_time, Cols.solution_cooling,
        Cols.second_solution_temp, Cols.second_solution_time, Cols.second_solution_cooling,
        Cols.aging_temp, Cols.aging_time, Cols.aging_cooling,
        Cols.second_aging_temp, Cols.second_aging_time, Cols.second_aging_cooling,
        Cols.annealing_temp, Cols.annealing_time, Cols.annealing_cooling,
        Cols.second_annealing_temp, Cols.second_annealing_time, Cols.second_annealing_cooling
    ]
    
    # Meta columns
    meta_cols = [Cols.alloy_designation, "params", "file", Cols.remarks]
    
    # Reorder columns
    all_cols = chem_cols + prop_cols + process_cols + meta_cols
    existing_cols = [col for col in all_cols if col in df.columns]
    df = df.reindex(columns=existing_cols)
    
    return df


@guidance
def chemical_composition_entry(lm, element_name: str, verbosity: int, logger: Logger):
    """Generate entry for a specific chemical element"""
    lm += f'''"{element_name}": "'''
    lm += gen_limited(regex=r'Bal\.|null|[+0-9\.\-<>=]+', name=f'chem_{element_name}')
    lm += '"'
    
    if verbosity > 1:
        logger.log(f"    added chemical component: {element_name}")
    
    return lm


@guidance
def mechanical_property_entry(lm, verbosity: int, logger: Logger):
    lm += select(["]", "{", ", {", ",{"], name="entry_start_mech")

    if verbosity > 1:
        logger.log(f"""    new mechanical property entry: {lm["entry_start_mech"] != "]"}""")

    if lm["entry_start_mech"] == "]":
        return lm

    # Mechanical properties
    lm += (
        f'''"{Cols.strength}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+', name='strength')}", '''
        + f'''"{Cols.elongation}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+', name='elongation')}", '''
    )
    
    # Condition label
    lm += f'''"{Cols.condition}": "{gen_limited(name='condition')}", '''
    
    # Process parameters - Forging
    lm += f'''"{Cols.forging_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    
    # Processing temperature
    lm += f'''"{Cols.processing_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    
    # Deformation
    lm += f'''"{Cols.deformation}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    
    # Solution treatment
    lm += f'''"{Cols.solution_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.solution_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.solution_cooling}": "{gen_limited()}", '''
    
    # Second solution
    lm += f'''"{Cols.second_solution_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_solution_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_solution_cooling}": "{gen_limited()}", '''
    
    # Aging
    lm += f'''"{Cols.aging_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.aging_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.aging_cooling}": "{gen_limited()}", '''
    
# Second aging
    lm += f'''"{Cols.second_aging_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_aging_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_aging_cooling}": "{gen_limited()}", '''
    
    # Annealing
    lm += f'''"{Cols.annealing_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.annealing_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.annealing_cooling}": "{gen_limited()}", '''
    
    # Second annealing
    lm += f'''"{Cols.second_annealing_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_annealing_time}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    lm += f'''"{Cols.second_annealing_cooling}": "{gen_limited()}"'''
    
    lm += "}"

    if verbosity > 1:
        logger.log("    added new mechanical property entry")

    return lm


@guidance
def alloy_entry(
    lm,
    max_count_chemical: int,
    max_count_mechanical: int,
    verbosity: int,
    logger: Logger,
):
    lm += f'''"{Cols.alloy_designation}": "{gen_limited()}", '''
    
    # Chemical composition - fixed 13 elements
    lm += '''"chemical composition": {'''
    
    elements = ["Ti", "Mo", "Al", "Sn", "V", "Zr", "Cr", "Nb", "Ta", "Fe", "W", "Si", "O"]
    
    for i, elem in enumerate(elements):
        lm += chemical_composition_entry(elem, verbosity, logger)
        if i < len(elements) - 1:
            lm += ", "
    
    lm += "}, "
    
    # Transition temperature
    lm += f'''"{Cols.transition_temp}": "{gen_limited(regex=r'null|[+0-9\.\-<>=]+')}", '''
    
    # Mechanical properties
    lm += '''"mechanical properties": ['''

    n_mech = 0
    mech_prop_last = [None, None]
    while True:
        lm += mechanical_property_entry(verbosity, logger)

        if max_count_mechanical is not None:
            max_count_mechanical -= 1
            if max_count_mechanical == 0:
                logger.log(lm)
                raise Exception(
                    "Max count for mechanical properties reached, this shouldn't happen for normal papers"
                )

        if lm["entry_start_mech"] == "]":
            break

        if verbosity == 1:
            logger.log("  added mechanical property entry")

        if lm["strength"] == "null" and lm["elongation"] == "null":
            lm += "]"
            if verbosity > 0:
                logger.log(
                    "  stopping mechanical properties generation because all values of last entry were 'null'"
                )
            break

        if mech_prop_last == [lm["strength"], lm["elongation"]]:
            lm += "]"
            if verbosity > 0:
                logger.log(
                    "  stopping mechanical properties generation because last entry was the same as the one before"
                )
            break

        mech_prop_last = [lm["strength"], lm["elongation"]]
        n_mech += 1

    global RELEVANT_INFORMATION_ADDED
    global CHEM_VALS_PREV
    
    # Collect chemical composition for duplicate detection
    chem_vals = []
    for elem in elements:
        val = lm.get(f'chem_{elem}', 'null')
        if val != 'null':
            chem_vals.append((elem, val))
    
    RELEVANT_INFORMATION_ADDED = (len(chem_vals) > 0) and (n_mech > 0) and (
        set(chem_vals) not in CHEM_VALS_PREV
    )

    if verbosity > 1 and set(chem_vals) in CHEM_VALS_PREV:
        logger.log("  chemical composition repeated, stopping entry generation")

    CHEM_VALS_PREV.append(set(chem_vals))

    if verbosity > 1:
        logger.log(
            f"  n_chem={len(chem_vals)}, n_mech={n_mech}, chem_vals in prev: {set(chem_vals) in CHEM_VALS_PREV}"
        )

    return lm


@guidance
def build_alloy_json(
    lm,
    max_count_alloys=None,
    max_count_chemical=None,
    max_count_mechanical=None,
    verbosity=0,
    logger=Logger(),
):
    global RELEVANT_INFORMATION_ADDED
    global CHEM_VALS_PREV
    RELEVANT_INFORMATION_ADDED = True
    CHEM_VALS_PREV = []

    while True:
        lm += select(["]", "]```", "{"], name="alloy_entry_start")

        if verbosity > 0:
            logger.log(f'''  new alloy to add: {"]" not in lm["alloy_entry_start"]}''')

        if max_count_alloys is not None:
            max_count_alloys -= 1
            if max_count_alloys == 0:
                logger.log(lm)
                raise Exception(
                    "Max count for alloy entries reached, this shouldn't happen for normal papers"
                )

        if "]" in lm["alloy_entry_start"]:
            return lm

        lm += (
            alloy_entry(max_count_chemical, max_count_mechanical, verbosity, logger)
            + f''', "{Cols.remarks}": "{gen(max_tokens=500, stop_regex=r'["}\]]')}"}}, '''
        )

        if not RELEVANT_INFORMATION_ADDED == True:
            lm += "]"
            if verbosity > 0:
                logger.log(
                    "  stopping alloy generation because no relevant information was added in last entry"
                )
            break

        if verbosity > 0:
            logger.log("  added new alloy entry")

    return lm


def unindent_json_examples_in_prompt(prompt: str):
    parts = prompt.split("```")

    def try_parse(part: str):
        try:
            return json.loads(part.strip("json\n ").replace("\n", ""))
        except:
            return None

    parsed = [try_parse(part) for part in parts]
    
    # Count valid JSON blocks
    valid_jsons = [el for el in parsed if el is not None]
    assert len(valid_jsons) >= 2, f"Expected at least 2 JSON examples, found {len(valid_jsons)}"
    assert len(parsed) == len(parts), f"Lengths do not match: {len(parsed)} vs {len(parts)}"

    for i, el in enumerate(parsed):
        if el is not None:
            parts[i] = "json\n" + json.dumps(el) + "\n"

    prompt = "```".join(parts)
    return prompt


SYSTEM_PROMPT = unindent_json_examples_in_prompt(f"""You are a paper information extractor specialized in titanium alloys. Given a paper about titanium alloys, extract the chemical composition and ONLY the following 2 mechanical properties: tensile strength (UTS) and elongation. Do not extract any other mechanical properties! If no relevant information about the chemical composition and at least one of the aforementioned mechanical properties is found in the paper, just make the json empty.

### Required Information to Extract:

1. **{Cols.alloy_designation}**: The designation of the alloy as stated in the paper (e.g., "Ti-6Al-4V", "TC4", etc.)

2. **chemical composition**: A dictionary containing the weight percentages of 13 elements:
   - Ti, Mo, Al, Sn, V, Zr, Cr, Nb, Ta, Fe, W, Si, O
   - Each element value should be a string (e.g., "6.0" for 6.0 wt%, or "Bal." for balance, or "null" if not stated)
   - If an element is stated as "balance" or "remaining" or "rest", use "Bal."
   - Example: {{"Ti": "Bal.", "Al": "6.0", "V": "4.0", "Mo": "null", ...}}

3. **{Cols.transition_temp}**: The beta transus temperature (β phase transition temperature) in °C. Use "null" if not stated.

4. **mechanical properties**: A list of dictionaries, where each dictionary represents one test/measurement with the following fields:
   
   **Mechanical Property Values:**
   - **{Cols.strength}**: Ultimate tensile strength (UTS) in MPa (string, e.g., "950.5" or "null")
   - **{Cols.elongation}**: Elongation (EL) in % (string, e.g., "12.5" or "null")
   
   **Condition Label:**
   - **{Cols.condition}**: One of the following condition labels:
     * "STA-OQ" (Solution + Aging with Oil Quench)
     * "STA-AC" (Solution + Aging with Air Cool)
     * "STA-FC" (Solution + Aging with Furnace Cool)
     * "STA-WQ" (Solution + Aging with Water Quench)
     * "ST-OQ" (Solution Treatment with Oil Quench only)
     * "ST-AC" (Solution Treatment with Air Cool only)
     * "ST-FC" (Solution Treatment with Furnace Cool only)
     * "ST-WQ" (Solution Treatment with Water Quench only)
     * "Direct-Aged" (Aging only, no solution treatment)
     * "Annealed-AC" (Annealing with Air Cool)
     * "Annealed-WQ" (Annealing with Water Quench)
     * "As-forged" (Forged state without heat treatment)
     * "Unknown" (if condition cannot be determined)
   
   **Process Parameters** (all are strings, use "null" if not stated):
   - **{Cols.forging_temp}**: Forging temperature in °C (e.g., "1050" or "null")
   - **{Cols.processing_temp}**: Processing temperature in °C (e.g., "900" or "null")
   - **{Cols.deformation}**: Deformation amount in % (e.g., "70" or "null")
   
   **Solution Treatment:**
   - **{Cols.solution_temp}**: Solution temperature in °C (e.g., "950" or "null")
   - **{Cols.solution_time}**: Solution time in hours (e.g., "2" or "null")
   - **{Cols.solution_cooling}**: Cooling method (e.g., "oil quench", "air cool", "water quench", "furnace cool", or "null")
   
   **Second Solution Treatment** (if applicable):
   - **{Cols.second_solution_temp}**: Temperature in °C
   - **{Cols.second_solution_time}**: Time in hours
   - **{Cols.second_solution_cooling}**: Cooling method
   
   **Aging Treatment:**
   - **{Cols.aging_temp}**: Aging temperature in °C (e.g., "550" or "null")
   - **{Cols.aging_time}**: Aging time in hours (e.g., "4" or "null")
   - **{Cols.aging_cooling}**: Cooling method
   
   **Second Aging Treatment** (if applicable):
   - **{Cols.second_aging_temp}**: Temperature in °C
   - **{Cols.second_aging_time}**: Time in hours
   - **{Cols.second_aging_cooling}**: Cooling method
   
   **Annealing Treatment:**
   - **{Cols.annealing_temp}**: Annealing temperature in °C
   - **{Cols.annealing_time}**: Annealing time in hours
   - **{Cols.annealing_cooling}**: Cooling method
   
   **Second Annealing Treatment** (if applicable):
   - **{Cols.second_annealing_temp}**: Temperature in °C
   - **{Cols.second_annealing_time}**: Time in hours
   - **{Cols.second_annealing_cooling}**: Cooling method

5. **{Cols.remarks}**: Any important remarks, caveats, or assumptions (string, can be empty "")

### JSON Format Examples:

**Example 1: Standard STA treatment**
```json
[
    {{
        "{Cols.alloy_designation}": "Ti-6Al-4V",
        "chemical composition": {{
            "Ti": "Bal.", "Al": "6.0", "V": "4.0", "Mo": "null", "Sn": "null",
            "Zr": "null", "Cr": "null", "Nb": "null", "Ta": "null", "Fe": "0.3",
            "W": "null", "Si": "null", "O": "0.2"
        }},
        "{Cols.transition_temp}": "995",
        "mechanical properties": [
            {{
                "{Cols.strength}": "950.0",
                "{Cols.elongation}": "12.5",
                "{Cols.condition}": "STA-WQ",
                "{Cols.forging_temp}": "null",
                "{Cols.processing_temp}": "null",
                "{Cols.deformation}": "null",
                "{Cols.solution_temp}": "920",
                "{Cols.solution_time}": "1",
                "{Cols.solution_cooling}": "water quench",
                "{Cols.second_solution_temp}": "null",
                "{Cols.second_solution_time}": "null",
                "{Cols.second_solution_cooling}": "null",
                "{Cols.aging_temp}": "540",
                "{Cols.aging_time}": "4",
                "{Cols.aging_cooling}": "air cool",
                "{Cols.second_aging_temp}": "null",
                "{Cols.second_aging_time}": "null",
                "{Cols.second_aging_cooling}": "null",
                "{Cols.annealing_temp}": "null",
                "{Cols.annealing_time}": "null",
                "{Cols.annealing_cooling}": "null",
                "{Cols.second_annealing_temp}": "null",
                "{Cols.second_annealing_time}": "null",
                "{Cols.second_annealing_cooling}": "null"
            }}
        ],
        "{Cols.remarks}": ""
    }}
]
```

**Example 2: Multiple heat treatments, as-forged condition**
```json
[
    {{
        "{Cols.alloy_designation}": "Ti-5Al-2.5Sn",
        "chemical composition": {{
            "Ti": "Bal.", "Al": "5.0", "Sn": "2.5", "Mo": "null", "V": "null",
            "Zr": "null", "Cr": "null", "Nb": "null", "Ta": "null", "Fe": "null",
            "W": "null", "Si": "null", "O": "null"
        }},
        "{Cols.transition_temp}": "null",
        "mechanical properties": [
            {{
                "{Cols.strength}": "850.0",
                "{Cols.elongation}": "null",
                "{Cols.condition}": "As-forged",
                "{Cols.forging_temp}": "1050",
                "{Cols.processing_temp}": "null",
                "{Cols.deformation}": "null",
                "{Cols.solution_temp}": "null",
                "{Cols.solution_time}": "null",
                "{Cols.solution_cooling}": "null",
                "{Cols.second_solution_temp}": "null",
                "{Cols.second_solution_time}": "null",
                "{Cols.second_solution_cooling}": "null",
                "{Cols.aging_temp}": "null",
                "{Cols.aging_time}": "null",
                "{Cols.aging_cooling}": "null",
                "{Cols.second_aging_temp}": "null",
                "{Cols.second_aging_time}": "null",
                "{Cols.second_aging_cooling}": "null",
                "{Cols.annealing_temp}": "null",
                "{Cols.annealing_time}": "null",
                "{Cols.annealing_cooling}": "null",
                "{Cols.second_annealing_temp}": "null",
                "{Cols.second_annealing_time}": "null",
                "{Cols.second_annealing_cooling}": "null"
            }},
            {{
                "{Cols.strength}": "1050.0",
                "{Cols.elongation}": "8.2",
                "{Cols.condition}": "STA-AC",
                "{Cols.forging_temp}": "null",
                "{Cols.processing_temp}": "null",
                "{Cols.deformation}": "null",
                "{Cols.solution_temp}": "900",
                "{Cols.solution_time}": "2",
                "{Cols.solution_cooling}": "air cool",
                "{Cols.second_solution_temp}": "null",
                "{Cols.second_solution_time}": "null",
                "{Cols.second_solution_cooling}": "null",
                "{Cols.aging_temp}": "600",
                "{Cols.aging_time}": "6",
                "{Cols.aging_cooling}": "air cool",
                "{Cols.second_aging_temp}": "null",
                "{Cols.second_aging_time}": "null",
                "{Cols.second_aging_cooling}": "null",
                "{Cols.annealing_temp}": "null",
                "{Cols.annealing_time}": "null",
                "{Cols.annealing_cooling}": "null",
                "{Cols.second_annealing_temp}": "null",
                "{Cols.second_annealing_time}": "null",
                "{Cols.second_annealing_cooling}": "null"
            }}
        ],
        "{Cols.remarks}": ""
    }}
]
```

**Example 3: Complex multi-step treatment**
```json
[
    {{
        "{Cols.alloy_designation}": "Ti-10V-2Fe-3Al",
        "chemical composition": {{
            "Ti": "Bal.", "V": "10.0", "Fe": "2.0", "Al": "3.0", "Mo": "null",
            "Sn": "null", "Zr": "null", "Cr": "null", "Nb": "null", "Ta": "null",
            "W": "null", "Si": "null", "O": "null"
        }},
        "{Cols.transition_temp}": "805",
        "mechanical properties": [
            {{
                "{Cols.strength}": "1200.0",
                "{Cols.elongation}": "10.0",
                "{Cols.condition}": "STA-OQ",
                "{Cols.forging_temp}": "null",
                "{Cols.processing_temp}": "900",
                "{Cols.deformation}": "70",
                "{Cols.solution_temp}": "850",
                "{Cols.solution_time}": "1.5",
                "{Cols.solution_cooling}": "oil quench",
                "{Cols.second_solution_temp}": "null",
                "{Cols.second_solution_time}": "null",
                "{Cols.second_solution_cooling}": "null",
                "{Cols.aging_temp}": "520",
                "{Cols.aging_time}": "8",
                "{Cols.aging_cooling}": "air cool",
                "{Cols.second_aging_temp}": "480",
                "{Cols.second_aging_time}": "4",
                "{Cols.second_aging_cooling}": "air cool",
                "{Cols.annealing_temp}": "null",
                "{Cols.annealing_time}": "null",
                "{Cols.annealing_cooling}": "null",
                "{Cols.second_annealing_temp}": "null",
                "{Cols.second_annealing_time}": "null",
                "{Cols.second_annealing_cooling}": "null"
            }}
        ],
        "{Cols.remarks}": "Two-step aging process applied"
    }}
]
```

### Important Rules:

1. **Only extract data that is explicitly stated in the paper** - do not make up information!
2. **Use "null" for missing data** - never leave fields empty or omit them
3. **All numeric values must be strings** (e.g., "950.5" not 950.5)
4. **Chemical composition must include all 13 elements** - use "null" for elements not mentioned
5. **Only create entries where BOTH chemical composition AND at least one mechanical property (strength or elongation) are stated**
6. **Cooling methods** should be one of: "oil quench", "air cool", "water quench", "furnace cool", or "null"
7. **Condition labels** must be one of the 13 predefined labels listed above
8. **For "balance" elements**, always use "Bal." (not "balance", "rest", "remaining", etc.)
9. **Values must match their field names** - don't put hardness in strength field, etc.
10. **Extract ALL measurements** for each alloy - if the same alloy was tested under different conditions, create separate entries in the mechanical properties list

If the paper contains no relevant information about titanium alloys, chemical composition, or mechanical properties, return an empty JSON array: []
""")

