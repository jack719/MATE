from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np


def pretty_print_cv(scores, print_means = True):
    n_spaces = 20
    sample_metrics = next(iter(scores.values()))
    if isinstance(sample_metrics, list):
        sample_metrics = sample_metrics[0]

    if isinstance(sample_metrics, dict):
        metrics = sample_metrics.keys()
        for key, val in scores.items():
            for metric in metrics:
                n_spaces_needed = n_spaces - len(key) - len(metric) - 1
                spacing = " " * n_spaces_needed
                if isinstance(val, dict):
                    print(f"{key}/{metric}:{spacing}{val[metric]:.3f}")
                elif isinstance(val, list):
                    print(f"{key}/{metric}:{spacing}{[f'{el[metric]:.3f}' for el in val]}")
    else:
        for key, val in scores.items():
            n_spaces_needed = n_spaces - len(key)
            spacing = " " * n_spaces_needed
            if isinstance(val, float):
                print(f"{key}:{spacing}{val:.3f}")
            elif isinstance(val, list):
                print(f"{key}:{spacing}{[f'{el:.3f}' for el in val]}")

    if print_means:
        metrics = next(iter(scores.values()))[0].keys()
        for metric in metrics:
            means = {f"{key}/{metric}": np.mean([el[metric] for el in val]) for key, val in scores.items()}
            means = dict(sorted(means.items(), key=lambda x: -x[1] if metric == "r^2" else x[1]))
            print("\nmeans:")
            pretty_print_cv(means, print_means=False)


def run_cv(x, y, files_series=None, n_folds=5, seed=42, col_idx_scaling = None, scaling_method = "minmax", best_model_metric="r^2", balance=False, params = {}):
    assert not np.isnan(x).any(), "x should not contain nan values!"
    assert not np.isnan(y).any(), "y should not contain nan values!"

    np.random.seed(seed)

    if files_series is not None:
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
    else:
        # just do n-fold cv (not by file)
        train_idx_per_fold = []
        test_idx_per_fold = []
        ss = ShuffleSplit(n_splits=n_folds, test_size=0.2)
        for train_idx, test_idx in ss.split(x, y):
            train_idx_per_fold.append(train_idx)
            test_idx_per_fold.append(test_idx)

    scores = {}
    for i in range(n_folds):
        if files_series is not None:
            test_files = files_per_group[i]
            test_idx = files_series.isin(test_files)
            train_idx = ~test_idx
        else:
            test_idx = test_idx_per_fold[i]
            train_idx = train_idx_per_fold[i]

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if col_idx_scaling is not None:
            # only center non-oh-features
            print("centering features...")

            if scaling_method == "standard":
                mean_train = np.mean(x_train[:, col_idx_scaling], axis=0)
                std_train = np.std(x_train[:, col_idx_scaling], axis=0)

                try:
                    x_train[:, col_idx_scaling] = (x_train[:, col_idx_scaling] - mean_train) / std_train
                    x_test[:, col_idx_scaling] = (x_test[:, col_idx_scaling] - mean_train) / std_train
                except Exception as e:
                    critical_std = std_train[std_train < 1e-3]
                    raise Exception(f"ERROR - possible cause - std close to 0 - stds < 1e-3: {critical_std}") from e
            elif scaling_method == "minmax":
                min_train = np.min(x_train[:, col_idx_scaling], axis=0)
                max_train = np.max(x_train[:, col_idx_scaling], axis=0)

                invalid = (min_train == 0) & (max_train == 0)
                if invalid.any():
                    print(f"WARNING: min and max values are both 0 for the following column idx: {np.where(invalid)[0]} -> skipping")
                    idx = col_idx_scaling & ~invalid
                    min_train = np.min(x_train[:, idx], axis=0)
                    max_train = np.max(x_train[:, idx], axis=0)
                else:
                    idx = col_idx_scaling.copy()

                x_train[:, idx] = (x_train[:, idx] - min_train) / (max_train - min_train)
                x_test[:, idx] = (x_test[:, idx] - min_train) / (max_train - min_train)
            elif scaling_method == "robust":
                scaler = RobustScaler()
                x_train[:, col_idx_scaling] = scaler.fit_transform(x_train[:, col_idx_scaling])
                x_test[:, col_idx_scaling] = scaler.transform(x_test[:, col_idx_scaling])
            elif scaling_method is None:
                pass
            else:
                raise ValueError(f"scaling method {scaling_method} not recognized!")

            assert not np.isnan(x_train).any(), "feature scaling might've introduced nan values in training samples"
            assert not np.isnan(x_test).any(), "feature scaling might've introduced nan values in test samples"

        if balance:
            # segment ys from min to max in 100mpa steps and make it so that each segment has roughly the same number of samples
            maxys = np.max(y_train)
            min_ = 0
            max_ = int(np.ceil(maxys / 100) * 100)

            bins = np.arange(min_, max_ + 100 + 1e-7, 100)
            y_train_binned = np.digitize(y_train, bins)

            n_per_bin = np.bincount(y_train_binned)
            n_per_bin = n_per_bin[n_per_bin > 0]
            n_per_bin = np.max(n_per_bin)

            _get_arr = lambda i: np.random.choice(np.where(y_train_binned == i)[0], n_per_bin, replace=True) if len(np.where(y_train_binned == i)[0]) > 0 else np.array([])
            idx = np.concatenate([_get_arr(i) for i in range(1, len(bins))]).astype(int)

            x_train = np.vstack([x_train[idx], x_train])
            y_train = np.concatenate([y_train[idx], y_train])

            print(f"balanced dataset: {len(y_train)} samples")


        par = params.get("lgbm", {})
        model = LGBMRegressor(**par)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        print(f"num test samples: {len(y_test)}")
        print(f"num train samples: {len(y_train)}")

        if "lgbm" not in scores:
            scores["lgbm"] = []
        scores["lgbm"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("gradient-boosting", {})
        model = GradientBoostingRegressor(**par)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if "gradient-boosting" not in scores:
            scores["gradient-boosting"] = []
        scores["gradient-boosting"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("knn", {})
        model = KNeighborsRegressor(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "knn" not in scores:
            scores["knn"] = []
        scores["knn"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("svr", {})
        model = SVR(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "svr" not in scores:
            scores["svr"] = []
        scores["svr"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("sklearn-mlp", {})
        model = MLPRegressor(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "sklearn-mlp" not in scores:
            scores["sklearn-mlp"] = []
        scores["sklearn-mlp"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("linear-regression", {})
        model = LinearRegression(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "linear-regression" not in scores:
            scores["linear-regression"] = []
        scores["linear-regression"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("ridge", {})
        model = Ridge(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "ridge" not in scores:
            scores["ridge"] = []
        scores["ridge"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        par = params.get("lasso", {})
        model = Lasso(**par)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        if "lasso" not in scores:
            scores["lasso"] = []
        scores["lasso"].append({
            "mae": mean_absolute_error(y_test, y_pred),
            "r^2": r2_score(y_test, y_pred)
        })

        y_pred_naive = np.mean(y_train)
        if "mean" not in scores:
            scores["mean"] = []
        scores["mean"].append({
            "mae": mean_absolute_error(y_test, np.full_like(y_test, y_pred_naive)),
            "r^2": r2_score(y_test, np.full_like(y_test, y_pred_naive))
        })


    # best model has the lowest/highest mean metric across all folds
    high_is_better = best_model_metric in ["r^2"]
    means = {key: np.mean([el[best_model_metric] for el in val]) for key, val in scores.items()}
    best_model = max(means, key=lambda x: means[x] if high_is_better else -means[x])
    print(f"best model: {best_model} - {best_model_metric}: {means[best_model]:.3f}")

    return scores, best_model

