import re
import os 
import pandas as pd
import numpy as np
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


def get_features(df, col, only_solidus=False):
    features = []
    n_features = 8 if not only_solidus else 2

    temp_col = [el for el in df.columns if "__Temp" in el]
    assert len(temp_col) == 1, "there should be exactly one temperature column"
    temp_col = temp_col[0]
    x = df[temp_col].values.reshape(-1, 1)
    y = df[col].values

    idx_max = np.argmax(y)
    temp_at_max_val = x[idx_max].squeeze()
    max_val = y[idx_max]

    if not only_solidus:
        features.extend([temp_at_max_val.item(), max_val.item()]) # maximum value and the corresponding temperature

    if not len(y[y!=0]) and not only_solidus:
        return np.array(features + [np.nan]*(n_features-len(features)))
    elif not len(y[y!=0]):
        return np.array([0])

    idx_min_no_zero = np.argmin(y[y!=0])

    temp_min_no_zero = x[y!=0][idx_min_no_zero]
    val_min_no_zero = y[y!=0][idx_min_no_zero].squeeze()

    idx_min_temp = np.argmin(x)
    if not only_solidus:
        features.extend([temp_min_no_zero.item(), val_min_no_zero.item()]) # minimum value that is not zero and the corresponding temperature
        # add last value (at solidification) with corresponding temperature
        features.extend([x[idx_min_temp].item(), y[idx_min_temp].item()])
    else:
        #features.append(y[idx_min_temp].item())
        #assert len(features) == 1, f"number of features should be 1, but is {len(features)}"
        #return np.array(features)
        #raise NotImplementedError("take value at specific temperature (400C?)!")
        idx_400C = np.argwhere(x == 400).flatten()
        if len(idx_400C) > 0:
            features.append(y[idx_400C[0]].item())
        else:
            features.append(np.nan)

        assert len(features) == 1, f"number of features should be 1, but is {len(features)}"
        return np.array(features)


    # Linear fit
    # remove leading and trailing zeros in y
    first_non_zero = np.where(y != 0)[0].min()
    last_non_zero = np.where(y != 0)[0].max()
    y_no_dup = y[first_non_zero:last_non_zero+1]
    x_no_dup = x[first_non_zero:last_non_zero+1]

    if not len(x_no_dup) > 0:
        return np.array(features + [np.nan]*(n_features-len(features)))

    if y[0] != y_no_dup[0]:
        y = np.concatenate([[y[0]], y_no_dup])
        x = np.concatenate([[x[0]], x_no_dup])

    if not len(x) > 1:
        return np.array(features + [np.nan]*(n_features-len(features)))

    model = LinearRegression()
    model.fit(x, y)

    coeffs = model.coef_
    intercept = model.intercept_
    features.extend([coeffs[0].item(), intercept.item()])

    assert len(features) == n_features, f"number of features should be {n_features}, but is {len(features)}"

    return np.array(features)


def get_phase_features(
        recalc,
        only_solidus,
        targets,
        df_feat=None,
        df_all=None,
        relevant_cols=None,
        paths={
            "featpath": "./untracked/df_feat_matcalc.csv",
            "featpath_solidus": "./untracked/df_feat_matcalc_solidus.csv",
            "respath_scheil": "./untracked/results_scheil",
            "respath_equ": "./untracked/results_equ",
        }
    ):
    featpath = paths["featpath"]
    featpath_solidus = paths["featpath_solidus"]
    respath_scheil = paths["respath_scheil"]
    respath_equ = paths["respath_equ"]
    featpath_not_found = not os.path.exists(featpath) and not only_solidus
    featpath_solidus_not_found = not os.path.exists(featpath_solidus) and only_solidus
    non_matched_idx = set()
    if recalc or (featpath_not_found or featpath_solidus_not_found):
        err_string = (
                "df_feat, df_all, targets, relevant_cols must be provided if recalc is True" if recalc
                else f"feature file not found at {featpath}" if featpath_not_found
                else f"feature file not found at {featpath_solidus}" if featpath_solidus_not_found
                else "unknown error"
        )
        assert (
                df_feat is not None
                and df_all is not None
                and relevant_cols is not None
            ), err_string

        types = ["scheil", "equ"]
        folders = [respath_scheil, respath_equ]
        if respath_scheil is not None:
            assert os.path.isdir(respath_scheil), f"results_scheil folder not found at {respath_scheil}"
            types = ["scheil"]
            folders = [respath_scheil]
        elif respath_equ is not None:
            assert os.path.isdir(respath_equ), f"results_equ folder not found at {respath_equ}"
            types = ["equ"]
            folders = [respath_equ]
        else:
            raise ValueError("either respath_scheil or respath_equ must be provided")


        chem_val_cols = relevant_cols["chem_val_cols"]

        df_feat_matcalc = df_feat.copy()
        for type_, folder in zip(types, folders):
            comps_dfs = {}
            comps_parsed = {}
            cols_no_al = [col for col in chem_val_cols if col.lower() != "al"]
            for fname in os.listdir(folder):
                if "errors" in fname:
                    continue

                test_file = os.path.join(folder, fname)
                with open(test_file, "r") as f:
                    lines = f.readlines()

                first_header_idx = np.where(list(map(lambda x: "Temp / C" in x, lines)))[0].min()
                lines = [el.strip("% ") for el in lines[first_header_idx-1:]]
                header_indices = np.where(list(map(lambda x: "Temp / C" in x, lines)))[0]

                dfs = []
                for i in range(len(header_indices)):
                    part = lines[header_indices[i]:header_indices[i+1]] if i < len(header_indices) - 1 else lines[header_indices[i]:]
                    columns = re.split(r"\s\s+", part[0])
                    columns = [c.strip() for c in columns]
                    part = [el for el in part if len(el.strip("\n")) > 10]
                    table = np.loadtxt(part, skiprows=1)
                    if len(table.shape) == 1:
                        table = table.reshape(1, -1)
                    df = pd.DataFrame(table, columns=columns)
                    dfs.append(df)

                if len(dfs) > 1:
                    df_concat = pd.concat(dfs, axis=1)
                else:
                    df_concat = dfs[0]

                # check if there are any columns with exactly the same values in each row and the same column name
                # if so, drop them
                drop_cols = df_concat.columns[df_concat.T.duplicated().values & (pd.Series(df_concat.columns).duplicated()).values]
                df_concat = df_concat.drop(drop_cols, axis=1)

                temp_cols_idx = np.where([re.match(r"Temp / C", c) for c in df_concat.columns])[0]
                df_concat = pd.concat([df_concat.iloc[:,temp_cols_idx[0]], df_concat.drop(df_concat.columns[temp_cols_idx], axis=1)], axis=1)


                assert len(df_concat.columns) == len(set(df_concat.columns)), "the following column names are duplicated: " + str(set([c for c in df_concat.columns if list(df_concat.columns).count(c) > 1]))

                chem_comp = re.findall(r"([a-zA-Z]*)(\d+\.\d+)", fname)
                chem_comp = {el[0]: float(el[1]) for el in chem_comp}
                chem_comp.update({el: 0 for el in cols_no_al if el not in chem_comp})
                chem_comp = {k: chem_comp[k] for k in cols_no_al}

                df_concat.columns = [f"{type_}__{c}" for c in df_concat.columns]
                comps_dfs[fname] = df_concat
                comps_parsed[fname] = chem_comp


            comp_df = pd.DataFrame(comps_parsed.values(), index=comps_parsed.keys())


            # for each row in the comp_df, get the closest row in comp_df using the euclidean distance
            max_acceptable_distance = 0. # must match exactly
            compare_df = df_all[chem_val_cols].fillna(0)
            if "Al" in compare_df.columns:
                compare_df.drop("Al", axis=1, inplace=True)
            assert list(compare_df.columns) == list(comp_df.columns), "columns should match"
            distances = cdist(compare_df, comp_df, metric="euclidean")

            #print(pd.Series(distances[~np.isinf(distances)].flatten()).describe())

            pd.Series(distances[~np.isinf(distances)]).describe()
            distances[distances > max_acceptable_distance] = np.inf
            np.fill_diagonal(distances, np.inf)
            closest_idx = np.argmin(distances, axis=1)
            closest_idx[np.isinf(distances).all(1)] = -1

            closest = comp_df.iloc[closest_idx]
            closest[closest_idx == -1] = np.nan
            idx = pd.Series(closest.index)
            idx[closest_idx == -1] = np.nan
            closest.index = idx

            match_idx_in_comp_df = pd.Series(closest.index)

            assert len(match_idx_in_comp_df) == len(df_feat_matcalc), "lengths should match"

            non_matched_idx = set(match_idx_in_comp_df[match_idx_in_comp_df.isna()].index) | non_matched_idx # indices where we do not have a corresponding scheil/equilibrium calculation
            matched_results = match_idx_in_comp_df[match_idx_in_comp_df.notna().values] # indices where we have a matching scheil/equilibrium calculation

            for row, fname in matched_results.items():
                #final_state = comps_dfs[fname].iloc[0]
                #df_feat_matcalc.loc[row, final_state.index] = final_state.values
                features = [get_features(comps_dfs[fname], col, only_solidus) for col in comps_dfs[fname].columns[1:]]
                features = np.concatenate(features)
                if not only_solidus:
                    feature_names = ["temp_at_max_val", "max_val", "temp_at_nonzero_min_val", "nonzero_min_val", "solidus_temp", "solidus_val", "slope", "intercept"]
                else:
                    feature_names = ["solidus_val"]

                col_names = [f"{col}__{fn}" for col in comps_dfs[fname].columns[1:] for fn in feature_names]
                df_feat_matcalc.loc[row, col_names] = features

        # drop all rows where we do not have a corresponding scheil/equilibrium calculation
        print(f"dropping {len(non_matched_idx)} rows where we do not have a corresponding scheil/equilibrium calculation")
        df_feat_matcalc = df_feat_matcalc.drop(non_matched_idx)

        targets_matcalc = targets.loc[df_feat_matcalc.index]
        df_feat_matcalc = df_feat_matcalc.loc[:,~df_feat_matcalc.isna().all(0)]
        if only_solidus:
            df_feat_matcalc.to_csv(featpath_solidus)
        else:
            df_feat_matcalc.to_csv(featpath)
    else:
        if not only_solidus:
            assert os.path.exists(featpath), f"feature file not found at {featpath}"
            df_feat_matcalc = pd.read_csv(featpath, index_col=0)
        else:
            assert os.path.exists(featpath_solidus), f"feature file not found at {featpath_solidus}"
            df_feat_matcalc = pd.read_csv(featpath_solidus, index_col=0)

        targets_matcalc = targets.loc[df_feat_matcalc.index]

    return df_feat_matcalc, targets_matcalc
