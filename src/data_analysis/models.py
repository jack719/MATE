from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

MODEL_PARAMS = {
    "lgbm": {
        "n_estimators": {"min": 50, "max": 300, "current": 200, "dt": int},
        "max_depth": {"min": 3, "max": 30, "current": 7, "dt": int},
        "learning_rate": {"min": 0.01, "max": 0.5, "current": 0.2, "dt": float},
        "num_leaves": {"min": 2, "max": 50, "current": 31, "dt": int},
        "min_child_samples": {"min": 5, "max": 100, "current": 3, "dt": int},
        "reg_alpha": {"min": 0.0, "max": 1.0, "current": 0.0, "dt": float},
        "reg_lambda": {"min": 0.0, "max": 1.0, "current": 0.0, "dt": float},
        "min_split_gain": {"min": 0.0, "max": 1.0, "current": 0.0, "dt": float},
        "verbose": -1,
        "n_jobs": -1
    },
    "svr": {
        "kernel": {"values": ["linear", "poly", "rbf", "sigmoid"], "current": "rbf"},
        "C": {"min": 0.1, "max": 1e4, "current": 500, "dt": float},
        "gamma": {"min": 1e-3, "max": 1e3, "current": 0.1, "dt": float},
        "epsilon": {"min": 1e-3, "max": 1e3, "current": 0.1, "dt": float},
        "tol": {"min": 1e-5, "max": 1e-1, "current": 1e-3, "dt": float},
        "degree": {"min": 2, "max": 20, "current": 3, "dt": int},
        "max_iter": 1000,
        "verbose": False
    },
    "elasticnet": {
        "alpha": {"min": 1e-3, "max": 1e3, "current": 1, "dt": float},
        "l1_ratio": {"min": 0.0, "max": 1.0, "current": 0.5, "dt": float},
        "max_iter": {"min": 100, "max": 1e5, "current": int(1e4), "dt": int},
        "tol": {"min": 1e-5, "max": 1e-1, "current": 1e-3, "dt": float},
    },
    "randomforest": {
        "n_estimators": {"min": 100, "max": 1000, "current": 100, "dt": int},
        "max_depth": {"min": 3, "max": 30, "current": 7, "dt": int},
        "min_samples_split": {"min": 2, "max": 20, "current": 2, "dt": int},
        "min_samples_leaf": {"min": 1, "max": 20, "current": 1, "dt": int},
        "max_features": {"min": 0.1, "max": 1.0, "current": 0.5, "dt": float},
        "bootstrap": True,
        "verbose": 0,
        "n_jobs": -1
    },
    "knn": {
        "n_neighbors": {"min": 1, "max": 100, "current": 5, "dt": int},
        "weights": {"values": ["uniform", "distance"], "current": "uniform"},
        "algorithm": {"values": ["auto", "ball_tree", "kd_tree", "brute"], "current": "auto"},
        "leaf_size": {"min": 10, "max": 100, "current": 30, "dt": int},
        "p": 2
    },
    "mlp": {
        "hidden_layer_sizes": {"values": [(8,8),(16,16),(8,),(16,),(8,16),(16,8),(4,8),(8,4), (4,8,4), (8,16,8)], "current": (16,16)},
        "activation": {"values": ["identity", "logistic", "tanh", "relu"], "current": "relu"},
        "solver": {"values": ["lbfgs", "sgd", "adam"], "current": "adam"},
        "alpha": {"min": 1e-5, "max": 1e-1, "current": 0.0001, "dt": float},
        "learning_rate": {"values": ["constant", "invscaling", "adaptive"], "current": "constant"},
    },
    "rf": {
        "n_estimators": {"min": 50, "max": 400, "current": 100, "dt": int},
        "max_depth": {"min": 3, "max": 30, "current": 7, "dt": int},
        "min_samples_split": {"min": 2, "max": 20, "current": 2, "dt": int},
        "min_samples_leaf": {"min": 1, "max": 20, "current": 1, "dt": int},
        "max_features": {"min": 0.1, "max": 1.0, "current": 1.0, "dt": float},
        "max_samples": {"min": 0.1, "max": 1.0, "current": 1.0, "dt": float},
        "bootstrap": True,
        "verbose": 0,
        "n_jobs": -1
    }
}

MODELS = {
    "lgbm": LGBMRegressor,
    "svr": SVR,
    "elasticnet": ElasticNet,
    "randomforest": RandomForestRegressor,
    "knn": KNeighborsRegressor,
    "mlp": MLPRegressor,
    "rf": RandomForestRegressor
}
