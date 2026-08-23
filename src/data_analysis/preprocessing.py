import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re

ALL_RELEVANT_COLS = None
from .elements import ELEMENT_ABBREVIATIONS

# disable warnings
import warnings
warnings.filterwarnings("ignore")

def _extract_numerical_values(val, verbose=False, range_list=[], range_method="mean"):
    """
    extract numerical values from the given string output of the LLM by first removing whitespaces from the string and then applying regex rules

    in particular:
    - if e.g. "2.0-3.0" is given -> return the mean (2.5)

    """

    assert range_method in ["mean", "min", "max"], "handle_ranges should be either mean, min or max!"

    if pd.isna(val) or (isinstance(val, str) and (val.strip() == "-" or val.strip() == "")):
        return np.nan

    val = str(val)
    val = val.replace(" ", "")
    range_pattern1 = re.compile(r"(\d+\.*\d*)[-–](\d+\.*\d*)") # example: 2.0-3.0
    range_pattern2 = re.compile(r"(\d+\.*\d*)to(\d+\.*\d*)") # example: 2.0 to 3.0 
    range_pattern3 = re.compile(r"(\d+\.*\d*)/(\d+\.*\d*)") # example: 2.0 / 3.0

    smaller_pattern1 = re.compile(r"<(\d+\.*\d*)") # example: <2.0
    smaller_pattern2 = re.compile(r"<=(\d+\.*\d*)") # example: <=2.0
    smaller_pattern3 = re.compile(r"(\d+\.*\d*)(max|maximum)\.*", re.IGNORECASE) # example: 2.0 max
    smaller_pattern4 = re.compile(r"(max|maximum)\.*(\d+\.*\d*)", re.IGNORECASE) # example: max 2.0

    pattern6 = re.compile(r">(\d+\.*\d*)") # example: >2.0
    pattern7 = re.compile(r">=(\d+\.*\d*)") # example: >=2.0
    pattern10 = re.compile(r"(\d+\.*\d*)[a-zA-Z.%]+$") # example: 2.0MPa
    pattern11 = re.compile(r"&lt;(\d+\.*\d*)") # example: &lt;2.0 (html encoded - &lt; is <)
    pattern12 = re.compile(r"~(\d+\.*\d*)") # example: ~2.0
    if range_pattern1.match(val):
        if verbose: print("matched range_pattern1")
        num1, num2 = map(float, range_pattern1.search(val).groups())
        range_list.append((num1, num2))

        if range_method == "mean":
            return str(np.mean([num1, num2]))
        elif range_method == "min":
            return str(np.min([num1, num2]))
        elif range_method == "max":
            return str(np.max([num1, num2]))
    elif range_pattern2.match(val):
        if verbose: print("matched range_pattern2")
        num1, num2 = map(float, range_pattern2.search(val).groups())
        range_list.append((num1, num2))

        if range_method == "mean":
            return str(np.mean([num1, num2]))
        elif range_method == "min":
            return str(np.min([num1, num2]))
        elif range_method == "max":
            return str(np.max([num1, num2]))
    elif range_pattern3.match(val):
        if verbose: print("matched range_pattern3")
        num1, num2 = map(float, range_pattern3.search(val).groups())
        range_list.append((num1, num2))

        if range_method == "mean":
            return str(np.mean([num1, num2]))
        elif range_method == "min":
            return str(np.min([num1, num2]))
        elif range_method == "max":
            return str(np.max([num1, num2]))
    elif smaller_pattern1.match(val):
        if verbose: print("matched smaller_pattern1")
        num = smaller_pattern1.search(val).group(1)
    elif smaller_pattern2.match(val):
        if verbose: print("matched smaller_pattern2")
        num = smaller_pattern2.search(val).group(1)
    elif pattern6.match(val):
        if verbose: print("matched pattern6")
        num = pattern6.search(val).group(1)
    elif pattern7.match(val):
        if verbose: print("matched pattern7")
        num = pattern7.search(val).group(1)
    elif smaller_pattern3.match(val):
        if verbose: print("matched smaller_pattern3")
        num = smaller_pattern3.search(val).group(1)
    elif smaller_pattern4.match(val):
        if verbose: print("matched smaller_pattern4")
        num = smaller_pattern4.search(val).group(2)
    elif pattern10.match(val):
        if verbose: print("matched pattern10")
        num = pattern10.search(val).group(1)
    elif pattern11.match(val):
        if verbose: print("matched pattern11")
        num = pattern11.search(val).group(1)
    elif pattern12.match(val):
        if verbose: print("matched pattern12")
        num = pattern12.search(val).group(1)
    else:
        if verbose: print("no pattern matched")
        num = val

    return num

def _can_be_cast_to_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def get_df(fname):
    return pd.read_csv(fname, dtype=str)

def filter_duplicates_and_invalid_ys_units(df):
    """
    remove duplicates
    remove rows where the yield strength has an invalid unit
    extract numerical values from the yield strength, ultimate tensile strength and elongation columns
    """

    df = df[~df.duplicated()]

    # only keep rows where the yield strength has a valid unit
    valid_units_yield_strength = ["mpa", "gpa", "pa", "kpsi", "ksi", "n/mm2", "n/mm^2", "n/mm^2", "n/m^2", "kg/mm^2"]
    units = df["YS (unit)"].apply(lambda x: re.sub(r'^[*\d^-]*', '', str(x).strip(" \n").lower()))
    valid_units = units.isin(valid_units_yield_strength + ["nan"])
    idx_valid = valid_units[valid_units].index
    df = df.loc[idx_valid].drop_duplicates().dropna(subset=["YS"]).reset_index(drop=True)

    # extract numerical values from ys, uts and elongation columns
    uts_processed = df["UTS"].apply(_extract_numerical_values).astype(float)
    elong_processed = df["elong"].apply(_extract_numerical_values).astype(float)
    ys_processed = df["YS"].apply(_extract_numerical_values).astype(float)

    non_processed = ys_processed[~ys_processed.apply(_can_be_cast_to_number)].values
    non_processed = np.concatenate([non_processed, uts_processed[~uts_processed.apply(_can_be_cast_to_number)].values])
    non_processed = np.concatenate([non_processed, elong_processed[~elong_processed.apply(_can_be_cast_to_number)].values])

    if len(non_processed) > 0:
        print("The following values could not be parsed:")
        for val in np.unique(non_processed):
            print(val)

    # remove rows where the yield strength is smaller or equal to the ultimate tensile strength
    df["UTS"] = df["UTS"].apply(_extract_numerical_values).astype(float)
    df["elong"] = df["elong"].apply(_extract_numerical_values).astype(float)
    df["YS"] = df["YS"].apply(_extract_numerical_values).astype(float)

    return df.reset_index(drop=True)

def get_ys_df(df):
    remove_row = (df["UTS"].astype(float) <= df["YS"].astype(float))
    print(f"dropped {remove_row.sum()} rows because the UTS was smaller or equal to the YS!")
    df = df[~remove_row]
    df = df.drop(columns=["UTS", "UTS (unit)", "elong", "elong (unit)"])
    df = df[df["YS"].apply(_can_be_cast_to_number)]

    return df.reset_index(drop=True)

def parse_non_standard_element_cols(df):
    """
    in some cases the LLM used the full name of the element instead of the abbreviation -> merge those columns their respective abbreviation columns
    """
    n_al_nonnan = df["Al"].notna().sum()
    print(f"\nnumber of non-nan entries for Al: {n_al_nonnan}")

    n_rows_before = len(df)
    print(f"n_rows_before: {n_rows_before}")
    n_cols_before = len(df.columns)
    print(f"merging columns with the same name! n_cols_before: {n_cols_before}")
    n_al_cols = len([el for el in df.columns if el == "Al"])


    for key, val in ELEMENT_ABBREVIATIONS.items():
        if not key in df.columns:
            continue
        n_renamed = df[key].notna().sum()
        print(f"replacing {key} with {val} ({n_renamed} entries)")
        to_transfer = ~df[key].isna()
        df.loc[to_transfer, val] = df.loc[to_transfer, key]


    drop_keys = [el for el in ELEMENT_ABBREVIATIONS.keys() if el in df.columns]
    drop_keys += [f"{el} (unit)" for el in drop_keys]
    df = df.drop(columns=drop_keys)

    n_al_cols = len([el for el in df.columns if el == "Al"])
    assert n_al_cols == 1, "there should only be one column for Al!"
    n_cols_after = len(df.columns)
    assert n_cols_after == (n_cols_before - len(drop_keys)), "number of columns should be reduced by the number of elements!"

    print(f"merging columns with the same name! n_cols_after: {n_cols_after}")
    n_rows_after = len(df)
    print(f"n_rows_after: {n_rows_after}")

    return df.reset_index(drop=True)

def get_all_relevant_cols(df=None):
    global ALL_RELEVANT_COLS
    if ALL_RELEVANT_COLS is None:
        assert df is not None, "df should be given if ALL_RELEVANT_COLS is empty!"
        measurement_cols = [
            "YS",
            "YS (unit)",
            "UTS",
            "UTS (unit)",
            "elong",
            "elong (unit)",
        ]

        chem_val_cols = [col for col in df.columns if col not in measurement_cols and f"{col} (unit)" in df.columns and " " not in col]
        time_cols = [c for c in df.columns if re.match(r"time \d", c) and f"{c} (unit)" in df.columns]
        temperature_cols = [c for c in df.columns if re.match(r"temperature \d", c) and f"{c} (unit)" in df.columns]
        cooling_method_col = [c for c in df.columns if "cooling method" in c]
        testing_temp_col = "testing temperature"
        assert testing_temp_col in df.columns, "testing temperature column should be present!"
        alloy_type_col = "alloy type"
        assert alloy_type_col in df.columns, "alloy type column should be present!"
        temper_designation_col = "temper designation"
        assert temper_designation_col in df.columns, "temper designation column should be present!"

        ALL_RELEVANT_COLS = {
            "chem_val_cols": chem_val_cols,
            "time_cols": time_cols,
            "temperature_cols": temperature_cols,
            "cooling_method_col": cooling_method_col,
            "testing_temp_col": testing_temp_col,
            "alloy_type_col": alloy_type_col,
            "temper_designation_col": temper_designation_col,
            "target_val_col": "YS",
            "measurement_cols": measurement_cols,
            "file_col": "file"
        }

    return ALL_RELEVANT_COLS

def parse_chemical_composition_values(df, range_method="mean"):
    """
    parse the chemical composition values from the given dataframe and return a dataframe where all values are numerical
    """
    all_relevant_cols = get_all_relevant_cols(df)
    numeric_cols = (
            all_relevant_cols["chem_val_cols"] +
            all_relevant_cols["time_cols"] +
            all_relevant_cols["temperature_cols"] +
            [all_relevant_cols["testing_temp_col"]] +
            [col for col in all_relevant_cols["measurement_cols"] if "unit" not in col and col in df.columns]
    )
    is_in_val_cols = np.array([el in numeric_cols for el in df.columns])
    to_cast_mask = (is_in_val_cols & (~df.isna()))

    range_list = []
    df[to_cast_mask] = df[to_cast_mask].map(_extract_numerical_values, range_list=range_list, range_method=range_method)
    if len(range_list) > 0:
        range_diffs = pd.Series([np.abs(r[1]-r[0]) for r in range_list])
        print("description of the range differences:")
        print(range_diffs.describe())

    def replace_invalid(val):
        if not re.match(r"\d", str(val)):
            valid_terms = ["bal", "rest", "rem"]
            if any(term in str(val).lower() for term in valid_terms):
                return -np.inf
            else:
                return np.nan

        return val

    df[to_cast_mask] = df[to_cast_mask].map(replace_invalid)

    n_drop = df[numeric_cols].isna().all(1).sum()
    print(f"dropped {n_drop} rows because all chemical comp. values were nan!")
    df = df[~df[numeric_cols].isna().all(1)]
    n_drop = ((df == -np.inf).sum(1)>1).sum()
    print(f"dropped {n_drop} rows because Bal. was present in more than one column!")
    df = df[(df == -np.inf).sum(1)<=1] # remove those rows where Bal. is present in more than one column
    df = df.reset_index(drop=True)

    df[numeric_cols] = df[numeric_cols].astype(float)

    return df.reset_index(drop=True)

def convert_valid_element_units_and_replace_bal(df):
    """
    convert the units of the chemical composition values to wt% and replace "Bal." in the chemical composition with the actually remaining value
    """
    chem_val_cols = get_all_relevant_cols(df)["chem_val_cols"]

    def calc_bal_val(row, log_dict):
        vals = row[chem_val_cols]
        units = row[[col + " (unit)" for col in chem_val_cols]]

        # convert vals to wt%
        for i, unit in enumerate(units):
            if "ppm" in str(unit):
                vals.iloc[i] = vals.iloc[i] / 10_000 # * 1/1e6 * 100
                log_dict["N_ppm"] += 1
            elif str(unit)!="nan" and "wt" not in str(unit):
                log_dict["N_invalid_units"] += 1
                return np.nan

        if any(val == -np.inf for val in vals):
            bal_val = 100 - sum([val for val in vals if val != -np.inf and not np.isnan(val)])
            return bal_val
        
        log_dict["N_no_bal"] += 1
        
        return np.nan

    log_dict = {"N_ppm": 0, "N_invalid_units": 0, "N_no_bal": 0}
    bal_replacements = df.apply(lambda x: calc_bal_val(x, log_dict), axis=1)
    N_ppm, N_invalid_units, N_no_bal = log_dict["N_ppm"], log_dict["N_invalid_units"], log_dict["N_no_bal"]

    print(f"calculated {N_ppm} values in ppm!")
    print(f"replaced {N_no_bal} Bal. values with the remaining value!")
    print(f"dropped {N_invalid_units} rows because they contained unknown units!")

    bal_replacements.name = "bal"
    df = pd.concat([df, bal_replacements], axis=1)

    for i, row in df.iterrows():
        if not np.isnan(row["bal"]):
            vals = row[chem_val_cols]
            bal_idx = np.where(np.array(vals) == -np.inf)[0][0]
            df.loc[i, chem_val_cols[bal_idx]] = row["bal"]

    df = df.drop(columns="bal")

    # drop rows which still contain -np.inf
    drop = df[df.apply(lambda x: -np.inf in x.values, axis=1)]
    print(f"dropped {len(drop)} rows because they still contained -np.inf!")
    df = df[~df.apply(lambda x: -np.inf in x.values, axis=1)].reset_index(drop=True)

    return df.reset_index(drop=True)

def remove_outliers(df, thr_max=1000, thr_min=10):
    thr_max = 1000
    drop = df[df["YS"].astype(float) > thr_max]
    print(f"dropped {len(drop)} rows because the yield strength was above {thr_max} MPa!")
    df = df[df["YS"].astype(float) <= thr_max].reset_index(drop=True)

    thr_min = 10
    drop = df[df["YS"].astype(float) < thr_min]
    print(f"dropped {len(drop)} rows because the yield strength was below {thr_min} MPa!")
    df = df[df["YS"].astype(float) >= thr_min].reset_index(drop=True)

    return df.reset_index(drop=True)

def drop_if_elements_not_sum_to_100(df):
    chem_val_cols = get_all_relevant_cols(df)["chem_val_cols"]

    log = {"n_dropped": 0}
    def drop_if_not_sum1(x, log):
        close = np.isclose(x[chem_val_cols].sum(), 100, atol=0.1)
        if not close:
            log["n_dropped"] += 1
            #print(f"dropping row because sum is not 100%: {x[chem_val_cols].sum()}")

        return close

    df = df[df.apply(drop_if_not_sum1, args=(log,), axis=1)].reset_index(drop=True)
    print(f"dropped {log['n_dropped']} rows because the chemical components did not sum up to 100%!")

    return df.reset_index(drop=True)

def update_relevant_cols(key, val, verbose=False):
    global ALL_RELEVANT_COLS
    assert ALL_RELEVANT_COLS is not None, "ALL_RELEVANT_COLS should be set first by calling get_all_relevant_cols and passing the dataframe!"

    if verbose:
        vals_prev = ALL_RELEVANT_COLS[key]
        vals_dropped = [el for el in vals_prev if el not in val]
        vals_added = [el for el in val if el not in vals_prev]
        if len(vals_dropped) > 0:
            print(f"dropped the following columns from {key}: {vals_dropped}")
        if len(vals_added) > 0:
            print(f"added the following columns to {key}: {vals_added}")

    ALL_RELEVANT_COLS[key] = val

def remove_duplicated_chem_comps_and_ys(df):
    all_relevant_cols = get_all_relevant_cols(df)
    chem_val_cols = [col for col in all_relevant_cols["chem_val_cols"] if col in df.columns]
    update_relevant_cols("chem_val_cols", chem_val_cols)

    n_before = len(df)
    df.drop_duplicates(subset=chem_val_cols + ["YS"], inplace=True)
    n_drop = n_before - len(df)

    print(f"dropped {n_drop} rows because they were duplicates of other rows with the same chemical composition and yield strength!")

    return df.reset_index(drop=True)

def filter_non_al_alloys(df):
    df = df[df["Al"] > 80] # only consider aluminum alloys
    df = df[df["Al"] < 100] # remove aluminum-only rows

    return df.reset_index(drop=True)

def validate_values_in_text(df, path_to_data):
    drop_row_idx = []
    for i, row in df.iterrows():
        file = row["file"].split("/")[-1]
        path = os.path.join(path_to_data, file)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"file {file} not found!")

        text = data["body"]
        text = text.lower()
        tables = data.get("tables", [])
        tables = "".join(tables.values())

        ys = row["YS"]

        if not np.isnan(ys):
            ys = str(int(ys))
            if ys not in text and ys not in tables:
                #print(f"YS not found in text for file {file}!")
                drop_row_idx.append(i)
                continue

    df = df.drop(drop_row_idx).reset_index(drop=True)
    print(f"dropped {len(drop_row_idx)} rows because the YS was not found in the text!")

    return df.reset_index(drop=True)

def select_common_elements(df, n_min=50):
    n_drop = 1
    while n_drop > 0:
        chem_val_cols = get_all_relevant_cols(df)["chem_val_cols"]

        # replace chem_val_cols where value is 0 with nan
        df.loc[:,chem_val_cols] = df.loc[:,chem_val_cols].replace(0.0, np.nan)

        # constrain the dataset s.t. each element has to be presented atleast n times in the dataset
        n_min = 50
        top_cols = df[chem_val_cols].notna().sum(0).sort_values(ascending=False) 
        keep_cols = top_cols[top_cols >= n_min].index

        print(f"keeping the top {len(keep_cols)} chemical components that are present in at least {n_min} entries!")

        non_zero_cols = df[chem_val_cols].apply(lambda x: x[~x.isna() & x != 0].index, axis=1)
        drop_row = non_zero_cols.apply(lambda x: any([idx not in keep_cols for idx in x]))

        n_drop = drop_row.sum()
        print(f"dropped {n_drop} rows because they contained chemical components that were not in the 20 most common ones!")
        df = df[~drop_row].reset_index(drop=True)

        chem_vals_to_drop = [col for col in df.columns if col not in keep_cols and col in chem_val_cols]
        df = df.drop(columns=chem_vals_to_drop)

        chem_val_cols = [col for col in df.columns if col in keep_cols]

        update_relevant_cols("chem_val_cols", chem_val_cols)

        print(f"number of val-cols left: {len(chem_val_cols)}")

    return df.reset_index(drop=True)

def remove_invalid_and_irrelevant_cols(df):
    n_cols_before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    print(f"dropped {n_cols_before - len(df.columns)} columns because they were all nan!")

    chem_val_cols = get_all_relevant_cols(df)["chem_val_cols"]
    chem_val_cols = [col for col in chem_val_cols if col in df.columns]
    chem_val_cols_without_unit = [col for col in chem_val_cols if f"{col} (unit)" not in df.columns]
    chem_val_cols_with_unit = [col for col in chem_val_cols if f"{col} (unit)" in df.columns]
    update_relevant_cols("chem_val_cols", chem_val_cols_with_unit)
    df = df.drop(columns=chem_val_cols_without_unit)

    unit_cols = [col for col in df.columns if "unit" in col]
    drop_cols = [col for col in unit_cols if col.replace(" (unit)", "") not in df.columns]

    df = df.drop(columns=drop_cols)

    all_relevant_cols = get_all_relevant_cols()
    invalid_cols = []
    for col in df.columns:
        found = False
        for vals in all_relevant_cols.values():
            if isinstance(vals, list) and col in vals or isinstance(vals, str) and col == vals:
                found = True
                break

        if not found:
            invalid_cols.append(col)

    df = df.drop(columns=invalid_cols)

    unit_cols = [col for col in df.columns if "unit" in col]
    df = df.drop(columns=unit_cols)

    return df.reset_index(drop=True)

def save_df_for_visualization_program(df, save_dir):
    if not os.path.exists(save_dir):
        print(f"Direcory {save_dir} does not exist! Exiting...")
        return

    df = df.copy()
    all_relevant_cols = get_all_relevant_cols() 
    df = df[all_relevant_cols["chem_val_cols"] + ["YS", "file"]]
    df["file"] = df["file"].str.split("/").str[-1]
    save_path = os.path.join(save_dir, "df_valid_ys.csv")
    df.to_csv(save_path, index=False)
    print(f"saved df to {save_path}")

def remove_correlated_features(df, threshold=0.9, enable_plots=False, image_dir="./publication/images"):
    df_corr = df.copy()

    # drop columns where all values are the same or nan
    to_drop = df_corr.columns[df_corr.nunique() <= 1]
    print(f"dropping {len(to_drop)} columns because they only contain one unique value!")
    df_corr = df_corr.drop(to_drop, axis=1)

    # create feature correlation plot/heatmap
    corr = df_corr.fillna(0).corr()
    # create the plot, and mark the dropped features in red
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=False)#, fmt=".2f")
    plt.title("Feature correlation heatmap")
    # disable axis (too many features)
    plt.xticks(())
    plt.yticks(())
    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(image_dir, "feature_correlation_heatmap.png")
        plt.savefig(img_path)
        plt.close()
        print("feature correlation heatmap saved to", img_path)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    to_drop = []
    for col in upper.columns:
        # do not consider correlation with columns which are already marked for dropping
        cvals = upper[col]
        cols_already_dropped = cvals.index.isin(to_drop)
        cvals[cols_already_dropped] = np.nan
        if any(cvals.abs() > threshold):
            to_drop.append(col)

    to_drop_non_matcalc = [el for el in to_drop if "scheil" not in el and "equ" not in el]

    print(f"dropping {len(to_drop)} features")
    print(f"dropping {len(to_drop_non_matcalc)} features which are not matcalc features")
    df_corr = df_corr.drop(to_drop, axis=1)

    # new correlation plot
    corr = df_corr.fillna(0).corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=False)#, fmt=".2f")
    plt.title(f"Feature correlation heatmap after removing highly correlated features (removed {len(to_drop)} features)")
    plt.xticks(())
    plt.yticks(())
    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(image_dir, "feature_correlation_heatmap_after.png")
        plt.savefig(img_path)
        plt.close()
        print("feature correlation heatmap after removing highly correlated features saved to", img_path)

    return df_corr.reset_index(drop=True)

