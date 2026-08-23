def get_element_df_and_targets(df, relevant_cols):
    df_el = df.copy()[relevant_cols["chem_val_cols"]]
    targets = df.copy()["YS"].reset_index(drop=True).astype(float)
    return df_el, targets

def process_temper_designation(df_feat, df_all, relevant_cols):
# temper designation
    col = relevant_cols["temper_designation_col"]

    df_all[col].value_counts()
    min_samples_from_different_papers = 10 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain the temper designation
    grouped = df_all.groupby("file")[col].apply(lambda x: x.str.lower().value_counts())
    unstacked = grouped.unstack()
    unstacked[unstacked > 1] = 1
    unstacked = unstacked.loc[:, unstacked.sum(0) >= min_samples_from_different_papers]
    selected_temper_designations = unstacked.columns # these occur atleast <min_samples_from_different_papers> times in seperate papers

    for temper_designation in selected_temper_designations:
        df_feat[f"OH__temper_designation__{temper_designation}"] = df_all[col].apply(lambda x: 1 if temper_designation == str(x).lower().strip() else 0)

    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def process_cooling_method(df_feat, df_all, relevant_cols, max_stages=3):
    cols = "cooling_method_col"
    all_vals = df_all[relevant_cols[cols][:max_stages]].stack().dropna()
    #print(all_vals.str.lower().value_counts())

    str_keywords_per_category = {"water": "water", "air": "air|room|20"}
    num = []
    for category, keywords in str_keywords_per_category.items():
        num.append(all_vals.str.lower().str.contains(keywords).sum())
        print(f"{category}: {num[-1]}")
    print(f"other: {all_vals.shape[0] - sum(num)}")


    min_samples_from_different_papers = 5 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain the cooling method at the given stage
    for stage in range(max_stages):
        col = relevant_cols[cols][stage]
        s = df_all[col].str.lower()
        for category, keywords in str_keywords_per_category.items():
            s[s.str.contains(keywords) == True] = category
        s[s.notna() & (~s.isin(str_keywords_per_category.keys()))] = "other"

        assert len(s.value_counts()) == len(str_keywords_per_category) + 1, "there should be one category for each keyword and one for 'other'"

        grouped = s.groupby(df_all["file"]).apply(lambda x: x.str.lower().value_counts())
        unstacked = grouped.unstack()
        unstacked[unstacked > 1] = 1

        unstacked_filtered = unstacked.loc[:, unstacked.sum(0) >= min_samples_from_different_papers]
        selected_cooling_methods = unstacked_filtered.columns # these occur atleast <min_samples_from_different_papers> times in seperate papers

        removed = unstacked.columns.difference(unstacked_filtered.columns)
        if len(removed):
            print(f"not including columns {removed} of stage {stage} because they only occur {unstacked[removed].sum(0).values} times in seperate papers")

        for cooling_method in selected_cooling_methods:
            df_feat[f"OH__cooling_method__{stage}__{cooling_method}"] = s.apply(lambda x: 1 if cooling_method in str(x).lower() else 0)


    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def process_times(df_feat, df_all, relevant_cols, max_stages=3):
    cols = "time_cols" # numeric, multiple columns

    df_all[relevant_cols[cols][:max_stages]].notna().sum(0)
    min_samples_from_different_papers = 5 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain heat treatment times at the given stage

    rel_cols = relevant_cols[cols][:max_stages]
    notnan_per_paper = df_all.groupby("file")[rel_cols].apply(lambda x: x.notna().sum(0))
    notnan_per_paper[notnan_per_paper >= 1] = 1
    above_thr = (notnan_per_paper.sum(0) >= min_samples_from_different_papers)
    add_cols = above_thr[above_thr].index

    df_feat[add_cols] = df_all[add_cols]

    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def process_temperatures(df_feat, df_all, relevant_cols, max_stages=3):
    cols = "temperature_cols" # numeric, multiple columns

    df_all[relevant_cols[cols][:max_stages]].notna().sum(0)
    min_samples_from_different_papers = 5 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain heat treatment temperatures at the given stage

    rel_cols = relevant_cols[cols][:max_stages]
    notnan_per_paper = df_all.groupby("file")[rel_cols].apply(lambda x: x.notna().sum(0))
    notnan_per_paper[notnan_per_paper >= 1] = 1
    above_thr = (notnan_per_paper.sum(0) >= min_samples_from_different_papers)
    add_cols = above_thr[above_thr].index

    df_feat[add_cols] = df_all[add_cols]

    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def process_testing_temp(df_feat, df_all, relevant_cols):
    col = "testing_temp_col" # numeric, single column
    col = relevant_cols[col]

    min_samples_from_different_papers = 5 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain the testing temperature
    grouped = df_all.groupby("file")[col].apply(lambda x: x.notna().sum())
    grouped[grouped >= 1] = 1
    if not (grouped.sum() >= min_samples_from_different_papers):
        print(f"not enough samples for column {col} -> skipping")
    else:
        df_feat[col] = df_all[col]

    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def process_alloy_type(df_feat, df_all, relevant_cols):
    col = "alloy_type_col" # categorical, single column
    col = relevant_cols[col]

    all_vals = df_all[col].dropna()

    str_keywords_per_category = {"cast": "cast", "wrought": "wrought"}
    num = []
    for category, keywords in str_keywords_per_category.items():
        num.append(all_vals.str.lower().str.contains(keywords).sum())
        print(f"{category}: {num[-1]}")
    print(f"other: {all_vals.shape[0] - sum(num)}")

    min_samples_from_different_papers = 5 # i.e. there has to be atleast <min_samples_from_different_papers> papers which contain the alloy type
    s = df_all[col].str.lower()
    for category, keywords in str_keywords_per_category.items():
        s[s.str.contains(keywords) == True] = category
    s[s.notna() & (~s.isin(str_keywords_per_category.keys()))] = "other"

    assert len(s.value_counts()) == len(str_keywords_per_category) + 1, "there should be one category for each keyword and one for 'other'"
    grouped = s.groupby(df_all["file"]).apply(lambda x: x.str.lower().value_counts())
    unstacked = grouped.unstack()
    unstacked[unstacked > 1] = 1
    unstacked_filtered = unstacked.loc[:, unstacked.sum(0) >= min_samples_from_different_papers]
    selected_alloy_types = unstacked_filtered.columns # these occur atleast <min_samples_from_different_papers> times in seperate papers

    removed = unstacked.columns.difference(unstacked_filtered.columns)
    if len(removed):
        print(f"not including columns {removed} because they only occur {unstacked[removed].sum(0).values} times in seperate papers")

    for alloy_type in selected_alloy_types:
        df_feat[f"OH__alloy_type__{alloy_type}"] = s.apply(lambda x: 1 if alloy_type in str(x).lower() else 0)

    print(f"new feature shape: {df_feat.shape}")

    return df_feat

def check_features(df_feat, df_all):
    oh_cols = [col for col in df_feat.columns if col.startswith("OH__")]
    other_cols = [col for col in df_feat.columns if not col.startswith("OH__")]
    files = df_all["file"]
    warn_at = 5

    print(f"\n----------\nNumber of samples: {len(df_feat)}\n----------")

    # pretty print an overview of the new features
    print(f"\n----------\nOH columns\n----------")
    for col in oh_cols:
        n_papers = df_feat[col].groupby(files).sum()
        n_papers[n_papers > 0] = 1
        n_papers = n_papers.sum()

        if n_papers < warn_at:
            print(f"WARNING (low paper diversity): {col} - {df_feat[col].sum()} samples from {n_papers} different papers")
        else:
            print(f"{col} - {df_feat[col].sum()} samples from {n_papers} different papers")

    print(f"\n----------\nother columns / num non-nan\n----------")
    for col in other_cols:
        n_papers = df_feat[col].groupby(files).apply(lambda x: x.notna().sum())
        n_papers[n_papers > 0] = 1
        n_papers = n_papers.sum()
        if n_papers < warn_at:
            print(f"WARNING (low paper diversity): {col} - {df_feat[col].notna().sum()} samples from {n_papers} different papers")
        else:
            print(f"{col} - {df_feat[col].notna().sum()} samples from {n_papers} different papers")

def add_YS_and_file(df_feat, df_all):
    df_feat = df_feat.copy()
    df_feat[["YS", "file"]] = df_all.loc[df_feat.index, ["YS", "file"]]
    return df_feat
