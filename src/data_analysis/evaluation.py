import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

from .models import MODELS, MODEL_PARAMS


def get_cv_indices(files_series, n_folds=5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    files = files_series.unique()
    files_shuffled = np.random.permutation(files)
    n_files = len(files_shuffled)
    n_per_group = n_files // n_folds
    rest = n_files % n_folds
    n_files_per_group = [n_per_group] * n_folds

    i = rest
    while i>0:
        for j in range(n_folds):
            n_files_per_group[j] += 1
            i -= 1
            if i == 0:
                break

    n_files_per_group = [0] + n_files_per_group

    files_per_group = [files_shuffled[sum(n_files_per_group[:i]):sum(n_files_per_group[:i+1])] for i in range(1, n_folds+1)]
    assert [len(el) for el in files_per_group] == n_files_per_group[1:], "number of files per group should match the expected number!"

    for i in range(n_folds):
        assert not any([
            pd.Series(files_per_group[i]).isin(files_per_group[j]).any()
            for j in range(n_folds) if j != i
            ]), "files should not be in multiple groups!"

    train_idx_per_fold = []
    test_idx_per_fold = []
    for files in files_per_group:
        test_idx = files_series.isin(files)
        train_idx = ~test_idx
        train_idx_per_fold.append(train_idx)
        test_idx_per_fold.append(test_idx)

        assert len(files_series[train_idx].unique()) + len(files_series[test_idx].unique()) == files_series.nunique(), "all files should be in either train or test set"

    assert len(train_idx_per_fold) == len(test_idx_per_fold) == n_folds, "number of folds should match the expected number!"
    assert files_series[train_idx_per_fold[0]].nunique() + files_series[test_idx_per_fold[0]].nunique() == files_series.nunique(), "all files should be in either train or test set"

    return train_idx_per_fold, test_idx_per_fold


def get_cv_indices_files_and_chem(chem_comps, files_series, n_folds=5, seed=None, return_groups=False):
    """
    same as above, but also make sure that the same chemical composition is not in train and test set
    chem_comps: dataframe with same length as files_series, containing the chemical composition
    """
    chem_comps_series = chem_comps.astype(str).apply(lambda x: ";".join(x), axis=1)

    comp_maps = {}
    file_maps = {}
    current_group = -1
    groups = np.array([])
    for comp, file in zip(chem_comps_series.values, files_series.values):
        if (comp not in comp_maps) and (file not in file_maps):
            current_group += 1
            comp_maps[comp] = current_group
            file_maps[file] = current_group
            groups = np.append(groups, current_group)
        elif (comp in comp_maps) and (file in file_maps):
            if comp_maps[comp] != file_maps[file]:
                g1, g2 = comp_maps[comp], file_maps[file]
                min_group = min(g1, g2)
                groups[groups == g1] = min_group
                groups[groups == g2] = min_group
                comp_maps[comp] = min_group
                file_maps[file] = min_group
                for k,v in comp_maps.items():
                    if v in [g1, g2]:
                        comp_maps[k] = min_group

                for k,v in file_maps.items():
                    if v in [g1, g2]:
                        file_maps[k] = min_group

            groups = np.append(groups, comp_maps[comp])
        elif comp in comp_maps:
            file_maps[file] = comp_maps[comp]
            groups = np.append(groups, comp_maps[comp])
        elif file in file_maps:
            comp_maps[comp] = file_maps[file]
            groups = np.append(groups, file_maps[file])
        else:
            raise ValueError("there's a glitch in the matrix")

    assert len(groups) == len(files_series) == len(chem_comps_series), "number of samples should match!"
    train_idx, test_idx = get_cv_indices(pd.Series(groups), n_folds=n_folds, seed=seed)

    # sanity check
    for i in range(n_folds):
        train_group = chem_comps_series.values[train_idx[i]]
        test_group = chem_comps_series.values[test_idx[i]]
        test_samples_in_train = [el in train_group for el in test_group]
        assert not any(test_samples_in_train), f"same chemical composition should not be in train and test set -> {test_group[test_samples_in_train]}"
        
    if return_groups:
        return train_idx, test_idx, groups
    else:
        return train_idx, test_idx


def cv_with_file_and_element_grouping(
    feat,
    y,
    p,
    model_name,
    chem_val,
    files,
    chem_cols,
    scaling_method,
    enable_plots=False,
    seed=2,
    img_dir="./publication/images",
    n_folds=3,
    suffix="",
):
    if seed is not None:
        np.random.seed(seed)

    train_idx_per_fold, test_idx_per_fold, groups = get_cv_indices_files_and_chem(chem_val, pd.Series(files), n_folds=n_folds, return_groups=True)

    r2s = []
    maes = []
    plot_data = []
    for i, (train_idx, test_idx) in enumerate(zip(train_idx_per_fold, test_idx_per_fold)):
        x_train = feat[train_idx]
        y_train = y[train_idx]
        x_test = feat[test_idx]
        y_test = y[test_idx]

        files_train = files[train_idx.values]
        files_test = files[test_idx.values]

        chem_train_str_concat = chem_val.loc[train_idx.values][chem_cols].astype(str).apply(lambda x: ";".join(x), axis=1)
        chem_test_str_concat = chem_val.loc[test_idx.values][chem_cols].astype(str).apply(lambda x: ";".join(x), axis=1)
        chem_intersections = set(chem_train_str_concat).intersection(set(chem_test_str_concat))
        file_intersections = set(files_train).intersection(set(files_test))
        assert not len([el for el in groups[train_idx.values] if el in groups[test_idx.values]])
        assert len(file_intersections) == 0, f"train and test files should not overlap! -> {file_intersections}"
        assert len(chem_intersections) == 0, f"train and test chemical compositions should not overlap! -> {chem_intersections}"


        if scaling_method == "minmax":
            min_ = x_train.min(0)
            max_ = x_train.max(0)
            x_train = (x_train - min_) / (max_ - min_ + 1e-7)
            x_test = (x_test - min_) / (max_ - min_ + 1e-7)
        elif scaling_method == "standard":
            mean = x_train.mean(0)
            std = x_train.std(0)
            x_train = (x_train - mean) / (std + 1e-7)
            x_test = (x_test - mean) / (std + 1e-7)
        elif scaling_method == "robust":
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaling_method is not None:
            raise ValueError("scaling method not recognized!")

        model = MODELS[model_name](**p)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

        # do tsne of train and test data together, then plot them in different colors (train is blue, test is red)
        # do 2 plots side by side: in the first, plot them as stated above, in the second, color train in blue and test based on the AE
        tsne = TSNE(n_components=2, random_state=42)
        x_all = tsne.fit_transform(feat)
        x_train = x_all[train_idx]
        x_test = x_all[test_idx]

        aes = np.abs(y_test - y_pred)
        vmax = max(aes)
        vmin = 10
        plot_data.append({"aes": aes, "x_train": x_train, "x_test": x_test, "vmin": vmin, "vmax": vmax, "r2": r2, "mae": mae, "y_test": y_test, "y_pred": y_pred})

    print(f"mean r2: {np.mean(r2s)}")
    print(f"mean mae: {np.mean(maes)}")

    if n_folds % 2 != 0:
        fig, ax = plt.subplots(n_folds, 1, figsize=(15, 25))
    else:
        fig, ax = plt.subplots(n_folds // 2, 2, figsize=(20, 15))
        ax = ax.flatten()

    for i, data in enumerate(plot_data):
        aes = data["aes"]
        x_train = data["x_train"]
        x_test = data["x_test"]
        r2 = data["r2"]
        mae = data["mae"]
        vmin = data["vmin"]
        vmax = data["vmax"]

        ax[i].scatter(x_train[:, 0], x_train[:, 1], c="green", alpha=0.1, label="training data", s=100)
        ax[i].set_xlabel("TSNE 1", fontsize=15)
        ax[i].set_ylabel("TSNE 2", fontsize=15)
        sc = ax[i].scatter(x_test[:, 0], x_test[:, 1], c=aes, cmap="seismic", norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sc, ax=ax[i])
        cbar.set_label("Absolute Error in MPa (Test Set)")

        ax[i].legend()
        ax[i].set_title(f"TSNE of train and test data, R^2: {r2:.2f}, MAE: {mae:.2f} MPa", fontsize=17)

    plt.tight_layout()
    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, f"tsne_{model_name}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved plot to {img_path}")


    fig, ax = plt.subplots(n_folds, 1, figsize=(15, 25))
    for i, data in enumerate(plot_data):
        aes = data["aes"]
        x_train = data["x_train"]
        x_test = data["x_test"]
        r2 = data["r2"]
        mae = data["mae"]
        y_test = data["y_test"]
        y_pred = data["y_pred"]
        ax[i].scatter(y_test, y_pred, c="blue", alpha=0.5)
        ax[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="red")
        ax[i].set_xlabel("True MPa", fontsize=15)
        ax[i].set_ylabel("Predicted MPa", fontsize=15)
        ax[i].set_title(f"R^2: {r2:.2f}, MAE: {mae:.2f} MPa", fontsize=17)
    
    plt.tight_layout()
    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, f"r2_{model_name}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved plot to {img_path}")

    return r2s, maes
