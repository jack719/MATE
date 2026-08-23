import random
import time
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from deap import base, creator, tools, algorithms
from warnings import simplefilter

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)


from .evaluation import get_cv_indices, get_cv_indices_files_and_chem


def eval_func(individual, x, y, files, chem_val, model_name, models, n_folds=5, col_idx_scaling=None, scaling_method=None, seed=None, subsample_size=None, penalty_per_active_feature=None, print_errors=False):
    feat_ind = individual.genes[:x.shape[1]]
    hyperparams = individual.genes[x.shape[1]:]

    if penalty_per_active_feature is not None:
        penalty = penalty_per_active_feature * sum(feat_ind)
    else:
        penalty = 0

    #print(f"penalty: {penalty}, penalty_per_active_feature: {penalty_per_active_feature}")
    params = {el["name"]: el["current"] for el in hyperparams}

    # get the indices of the selected features
    col_idx = np.array(feat_ind, dtype=bool)
    x = x[:, col_idx]
    if not x.shape[1] > 0:
        return 0, 

    #train_idx_per_fold, test_idx_per_fold = get_cv_indices(pd.Series(files), n_folds=n_folds, seed=seed)
    train_idx_per_fold, test_idx_per_fold, groups = get_cv_indices_files_and_chem(chem_val, pd.Series(files), n_folds=n_folds, return_groups=True, seed=seed)

    r2s = []
    for train_idx, test_idx in zip(train_idx_per_fold, test_idx_per_fold):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        #files_train = files[train_idx.values]
        #files_test = files[test_idx.values]

        #chem_train_str_concat = chem_val.loc[train_idx.values][chem_cols].astype(str).apply(lambda x: ";".join(x), axis=1)
        #chem_test_str_concat = chem_val.loc[test_idx.values][chem_cols].astype(str).apply(lambda x: ";".join(x), axis=1)
        #chem_intersections = set(chem_train_str_concat).intersection(set(chem_test_str_concat))
        #file_intersections = set(files_train).intersection(set(files_test))
        #assert not len([el for el in groups[train_idx.values] if el in groups[test_idx.values]])
        #assert len(file_intersections) == 0, f"train and test files should not overlap! -> {file_intersections}"
        #assert len(chem_intersections) == 0, f"train and test chemical compositions should not overlap! -> {chem_intersections}"


        if subsample_size is not None:
            if seed is not None:
                np.random.seed(seed)

            subsample_idx = np.random.choice(len(x_train), subsample_size, replace=False)
            x_train = x_train[subsample_idx]
            y_train = y_train[subsample_idx]

        if scaling_method == "minmax":
            col_idx = np.array(col_idx_scaling, dtype=bool) if col_idx_scaling is not None else np.array([True]*x_train.shape[1])
            min_ = x_train[:, col_idx].min(0)
            max_ = x_train[:, col_idx].max(0)
            x_train[:, col_idx] = (x_train[:, col_idx] - min_) / (max_ - min_ + 1e-7)
            x_test[:, col_idx] = (x_test[:, col_idx] - min_) / (max_ - min_ + 1e-7)
        elif scaling_method == "standard":
            col_idx = np.array(col_idx_scaling, dtype=bool) if col_idx_scaling is not None else np.array([True]*x_train.shape[1])
            mean = x_train[:, col_idx].mean(0)
            std = x_train[:, col_idx].std(0)
            x_train[:, col_idx] = (x_train[:, col_idx] - mean) / (std + 1e-7)
            x_test[:, col_idx] = (x_test[:, col_idx] - mean) / (std + 1e-7)
        elif scaling_method == "robust":
            col_idx = np.array(col_idx_scaling, dtype=bool) if col_idx_scaling is not None else np.array([True]*x_train.shape[1])
            scaler = RobustScaler()
            x_train[:, col_idx] = scaler.fit_transform(x_train[:, col_idx])
            x_test[:, col_idx] = scaler.transform(x_test[:, col_idx])
        elif scaling_method is not None:
            raise ValueError(f"scaling_method {scaling_method} not recognized")


        model = models[model_name](**params)
        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        except Exception as e:
            if print_errors:
                print(f"error: {e}, num_features: {sum(feat_ind)}, params: {params}")
            y_pred = np.zeros(len(y_test)) + 1e5

        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)

    return np.mean(r2s) - penalty,
    

class FitnessMax(base.Fitness):
    weights = (1.0,)


def generation_ea_simple(population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=True, constant_seed=None, re_eval_all=False):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    if not constant_seed:
        seed = random.randint(0, int(1e6))
    else:
        seed = constant_seed

    evaluate_with_seed = lambda ind: toolbox.evaluate(ind, seed=seed)

    fitnesses = toolbox.map(evaluate_with_seed, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    genes = {0: [(ind.genes, ind.fitness) for ind in population]}
    try:
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            #lambda_ = len(population) # number of individuals to produce
            #offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            if re_eval_all:
                # invalidate all individuals
                for ind in offspring:
                    del ind.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                if isinstance(fit, tuple):
                    ind.fitness.values = fit
                else:
                    ind.fitness.values = (fit,)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # track genes
            genes[gen] = [(ind.genes, ind.fitness) for ind in population]

            if verbose:
                print(logbook.stream)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, stopping the genetic algorithm...")

    return population, logbook, genes


class Individual:
    def __init__(self, n_features, params, num_features_to_select, init_random_params=True, n_feat_init=None):
        self.n_features = n_features
        self.params = params
        self.num_features_to_select = num_features_to_select
        self.n_feat_init = n_feat_init
        self.fitness = FitnessMax() # max fitness -> maximize r2

        self.genes = self.init_genes(init_random_params)

    def init_genes(self, init_random_params):
        genes = []
        for _ in range(self.n_features):
            if self.n_feat_init is not None:
                select_prob = min(self.n_feat_init/self.n_features,1)
            elif self.num_features_to_select is not None:
                select_prob = min(self.num_features_to_select/self.n_features,1)
            else:
                select_prob = 0.5

            genes.append(bool(np.random.choice([True, False], p=[select_prob, 1 - select_prob])))

        if not init_random_params:
            for k, v in self.params.items():
                if isinstance(v, dict) and "values" in v:
                    genes.append({"name": k, **v})
                elif isinstance(v, dict) and "min" in v:
                    genes.append({"name": k, **v})
                else:
                    genes.append({"name": k, "current": v})
        else:
            for k, v in self.params.items():
                if isinstance(v, dict) and "values" in v:
                    genes.append({"name": k, "current": random.choice(v["values"]), "values": v["values"]})
                elif isinstance(v, dict) and "min" in v:
                    val = int(random.uniform(v["min"], v["max"])) if v["dt"] == int else random.uniform(v["min"], v["max"])
                    genes.append({"name": k, "current": val, "min": v["min"], "max": v["max"], "dt": v["dt"]})
                else:
                    genes.append({"name": k, "current": v})

        return genes

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, key):
        return self.genes[key]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __iter__(self):
        return iter(self.genes)

    def __repr__(self):
        return f"Individual({self.genes})"

    def __str__(self):
        return f"Individual({self.genes})"


def mutate(individual, pb, to_mutate, mutate_probs_per_idx_range=None, mutate_method_params="uniform"):
    assert to_mutate in ["all", "features", "params", "custom_prob", "one_off_or_one_on"], "to_mutate should be either 'all', 'features', 'params', 'custom_prob', 'one_off_or_one_on'"

    if mutate_probs_per_idx_range is not None and to_mutate != "custom_prob":
        raise ValueError("mutate_probs_per_idx_range should be None if to_mutate is not 'custom_prob'")

    if to_mutate == "all":
        n_mutated_max = max(1, int(len(individual) * pb))
        n_mutated = random.randint(1, n_mutated_max)
        mutated_indices = random.sample(
            [index for index in range(len(individual.genes))], n_mutated)
    elif to_mutate == "features":
        n_mutated_max = max(1, int(individual.n_features * pb))
        n_mutated = random.randint(1, n_mutated_max)
        mutated_indices = random.sample(
            [index for index in range(individual.n_features)], n_mutated)
    elif to_mutate == "params":
        n_mutated_max = max(1, int((len(individual) - individual.n_features) * pb))
        n_mutated = random.randint(1, n_mutated_max)
        mutated_indices = random.sample(
            [index for index in range(individual.n_features, len(individual))], n_mutated)
    elif to_mutate == "custom_prob":
        assert mutate_probs_per_idx_range is not None, "mutate_probs_per_idx_range should not be None if to_mutate is 'custom_prob'"
        all_indices = [el.item() for indices, _ in mutate_probs_per_idx_range for el in indices]
        assert sorted(all_indices) == list(range(len(individual))), f"all indices should be covered by the mutate_probs_per_idx_range dictionary! n indices: {len(all_indices)}, n genes: {len(individual)}"
        mutated_indices = []
        for indices, prob in mutate_probs_per_idx_range:
            chosen = [i for i in indices if random.random() < prob]
            mutated_indices.extend(chosen)
    elif to_mutate == "one_off_or_one_on":
        turn_on = random.random() > 0.5
        # if turn on, find all the indices which are False and randomly turn one of them on
        # if turn off, find all the indices which are True and randomly turn one of them off
        if turn_on:
            mutated_indices = [i for i in range(len(individual.genes)) if isinstance(individual.genes[i], bool) and not individual.genes[i]]
        else:
            mutated_indices = [i for i in range(len(individual.genes)) if isinstance(individual.genes[i], bool) and individual.genes[i]]

        if len(mutated_indices) == 0:
            # if there are no genes to mutate, randomly select one
            mutated_index = random.randint(0, len(individual) - 1)
        else:
            mutated_index = random.choice(mutated_indices)

        individual.genes[mutated_index] = not individual.genes[mutated_index]

        vary_param = random.random() > 0.25 # with 25% probability, also vary a parameter
        if vary_param:
            mutated_indices = [i for i in range(len(individual.genes)) if isinstance(individual.genes[i], dict) and ("values" in individual.genes[i] or "min" in individual.genes[i])]
            mutated_indices = [random.choice(mutated_indices)]
        else:
            mutated_indices = []

    else:
        raise ValueError(f"to_mutate {to_mutate} not recognized")

    # for each mutated index which is a binary gene, flip the bit
    for index in mutated_indices:
        if isinstance(individual.genes[index], bool):
            #if not individual.genes[index]:
            #    if individual.genes.count(True) < individual.num_features_to_select:
            #        individual.genes[index] = True
            #    else: # if there are already enough features selected, flip the bit with a lower probability
            #        individual.genes[index] = bool(np.random.choice([True, False], p=[0.25, 0.75]))
            #else:
            #    if individual.genes.count(True) > individual.num_features_to_select:
            #        individual.genes[index] = False
            #    else: # if we do not have enough features selected, only turn off the feature with a lower probability
            #        individual.genes[index] = bool(np.random.choice([True, False], p=[0.75, 0.25]))
            if individual.num_features_to_select is None:
                individual.genes[index] = not individual.genes[index]
            elif not individual.genes[index]:
                if individual.genes.count(True) < individual.num_features_to_select:
                    individual.genes[index] = True
                else: # turn another feature off
                    individual.genes[index] = True
                    other_features = [i for i in range(len(individual.genes)) if i != index and individual.genes[i] is True]
                    if other_features:
                        other_idx = random.choice(other_features)
                        individual.genes[other_idx] = False
            else:
                if individual.genes.count(True) > individual.num_features_to_select:
                    individual.genes[index] = False
                else: # turn another feature on
                    individual.genes[index] = False
                    other_features = [i for i in range(len(individual.genes)) if i != index and individual.genes[i] is False]
                    if other_features:
                        other_idx = random.choice(other_features)
                        individual.genes[other_idx] = True
        elif isinstance(individual.genes[index], dict) and "values" in individual.genes[index]:
            values = individual.genes[index]["values"]
            current_idx = values.index(individual.genes[index]["current"])
            new_idx = random.choice([i for i in range(len(values)) if i != current_idx])
            individual.genes[index]["current"] = values[new_idx]
        elif isinstance(individual.genes[index], dict) and "min" in individual.genes[index]:
            min_ = individual.genes[index]["min"]
            max_ = individual.genes[index]["max"]

            if mutate_method_params == "uniform":
                new_val = random.uniform(min_, max_)
            elif mutate_method_params == "shift":
                shift = random.uniform(-0.33, 0.33)
                current_val = individual.genes[index]["current"]
                new_val = current_val * (1 + shift)
            else:
                raise ValueError(f"mutate_method_params {mutate_method_params} not recognized")

            new_val = int(new_val) if individual.genes[index]["dt"] == int else new_val
            if (mutate_method_params == "shift") and (individual.genes[index]["dt"] == int) and (new_val == individual.genes[index]["current"]):
                # if the shift did not change the value because it was rounded to the same integer, add 1 or subtract 1
                new_val = int(new_val) + 1 if shift > 0 else int(new_val) - 1

            individual.genes[index]["current"] = new_val
        elif isinstance(individual.genes[index], dict):
            pass # static parameters
        else:
            raise ValueError(f"gene type not recognized: {individual.genes[index]} - {type(individual.genes[index])}")

    return individual,


def mate(ind1, ind2, method="1point", to_mutate="all"):
    assert to_mutate in ["all", "features", "params", "custom_prob", "one_off_or_one_on"], "to_mutate should be either 'all', 'features', 'params', or 'custom_prob', 'one_off_or_one_on'"

    n_features = ind1.n_features
    params = ind1.params
    num_features_to_select = ind1.num_features_to_select
    n_feat_init = ind1.n_feat_init
    child1 = Individual(n_features, params, num_features_to_select, n_feat_init=n_feat_init)
    child2 = Individual(n_features, params, num_features_to_select, n_feat_init=n_feat_init)

    if method.lower() == "uniform":
        # method: uniform crossover (50% chance of inheriting from each parent)

        nrange = (
            range(len(ind1)) if to_mutate in ["all", "custom_prob"]
            else range(n_features) if to_mutate == "features"
            else range(n_features, len(ind1))
        )
        for i in nrange:
            if random.random() > 0.5:
                child1[i] = ind1[i]
                child2[i] = ind2[i]
            else:
                child1[i] = ind2[i]
                child2[i] = ind1[i]

    elif re.match(r"\d+point", method, re.IGNORECASE):
        # method: n-point crossover
        n = int(re.match(r"(\d+)point", method, re.IGNORECASE).group(1))

        crossover_points = (
                random.sample(range(len(ind1)), n) if to_mutate == ["all", "custom_prob"]
                else random.sample(range(n_features), n) if to_mutate == "features"
                else random.sample(range(n_features, len(ind1)), n)
        )
        nrange = (
            range(len(ind1)) if to_mutate == ["all", "custom_prob", "one_off_or_one_on"]
            else range(n_features) if to_mutate == "features"
            else range(n_features, len(ind1))
        )

        crossover_points.sort()
        switch = 0
        for i in nrange:
            if i in crossover_points:
                switch = 1 - switch
            
            if switch == 0:
                child1.genes[i] = ind1.genes[i]
                child2.genes[i] = ind2.genes[i]
            else:
                child1.genes[i] = ind2.genes[i]
                child2.genes[i] = ind1.genes[i]
    else:
        raise ValueError(f"method {method} not recognized")

    if not len(child1) == len(child2) == len(ind1) == len(ind2):
        breakpoint()

    return child1, child2


def generate_individual(n_features, params, init_random_params, num_features_to_select, n_feat_init):
    ind = Individual(n_features, params, init_random_params=init_random_params, num_features_to_select=num_features_to_select, n_feat_init=n_feat_init)
    return ind

def run_genetic_algorithm(x, y, files, chem_val, model_name, params, models, n_folds, n_generations=100, col_idx_scaling=None, scaling_method=None, constant_seed=None, mutate_probs_per_idx_range=None, mutate_method_params="uniform", feature_names = None, **gen_algo_params):

    n_feat = x.shape[1]

    prob_mutate_individual = gen_algo_params.get("prob_mutate_individual", 0.25)
    mating_prob = gen_algo_params.get("mating_prob", 0.25)
    prob_mutate_gene = gen_algo_params.get("prob_mutate_gene", 0.05)
    n_pop = gen_algo_params.get("n_pop", 50)
    crossover_method = gen_algo_params.get("crossover_method", "uniform")
    selection_algo = gen_algo_params.get("selection_algo", tools.selTournament)
    selection_algo_args = gen_algo_params.get("selection_algo_args", {"tournsize": 5})
    init_random_params = gen_algo_params.get("init_random_params", False)
    num_features_to_select = gen_algo_params.get("num_features_to_select", None) # 30
    penalty_per_active_feature = gen_algo_params.get("penalty_per_active_feature", None)
    n_feat_init = gen_algo_params.get("n_feat_init", None)
    re_eval_all = gen_algo_params.get("re_eval_all", False)
    assert (
        (num_features_to_select is None and penalty_per_active_feature is None)
        or (num_features_to_select is not None and penalty_per_active_feature is None)
        or (num_features_to_select is None and penalty_per_active_feature is not None)
    ), "num_features_to_select and penalty_per_active_feature cannot be set at the same time"

    to_mutate = gen_algo_params.get("to_mutate", "all")
    subsample_size = gen_algo_params.get("subsample_size", None) # for models like SVR which take a long time to fit on large datasets


    if to_mutate == "params" and num_features_to_select < x.shape[1]:
        import warnings
        print("---------------")
        warnings.warn("num_features_to_select < x.shape[1] and to_mutate == 'params' -> the same features will be selected in each created individual! Are you sure this is what you want?")
        print("---------------")


    creator.create("Individual", Individual)
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, n_features=x.shape[1], params=params[model_name], init_random_params=init_random_params, num_features_to_select=num_features_to_select, n_feat_init=n_feat_init)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func, x=x, y=y, files=files, chem_val=chem_val, model_name=model_name, models=models, n_folds=n_folds, col_idx_scaling=col_idx_scaling, scaling_method=scaling_method, subsample_size=subsample_size, penalty_per_active_feature=penalty_per_active_feature)

    toolbox.register("mate", mate, method=crossover_method, to_mutate=to_mutate)
    toolbox.register("mutate", mutate, pb=prob_mutate_gene, to_mutate=to_mutate, mutate_probs_per_idx_range=mutate_probs_per_idx_range, mutate_method_params=mutate_method_params)
    toolbox.register("select", selection_algo, **selection_algo_args)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind)
    n_dec = 3
    
    get_fitnesses = lambda x: [el.fitness.values[0] for el in x]
    stats.register("min", lambda x: max(0.0, np.round(np.min(get_fitnesses(x)), n_dec)))
    stats.register("25%", lambda x: max(0.0, np.round(np.percentile(get_fitnesses(x), 25), n_dec)))
    stats.register("50%", lambda x: max(0.0, np.round(np.percentile(get_fitnesses(x), 50), n_dec)))
    stats.register("75%", lambda x: max(0.0, np.round(np.percentile(get_fitnesses(x), 75), n_dec)))
    stats.register("max", lambda x: max(0.0, np.round(np.max(get_fitnesses(x)), n_dec)))
    stats.register("avg", lambda x: max(0.0, np.round(np.mean(get_fitnesses(x)), n_dec)))
    stats.register("std", lambda x: np.round(np.std(get_fitnesses(x)), n_dec))

    NITER = 0
    START_TIME = pd.Timestamp.now()
    def calc_time_left(_):
        nonlocal NITER
        nonlocal START_TIME
        NITER += 1

        time_passed = pd.Timestamp.now() - START_TIME
        time_per_iter_seconds = time_passed.total_seconds() / NITER
        time_left = time_per_iter_seconds * (n_generations - NITER)

        time_left_hours = time_left // 3600
        time_left_minutes = (time_left % 3600) // 60
        time_left_seconds = time_left % 60
        return f"{int(time_left_hours)}h:{int(time_left_minutes)}m:{int(time_left_seconds)}s"

    stats.register("time_left", lambda x: calc_time_left(x))

    #def get_feature_counts(all_inds):
    #    #feat_names = [f"f{i}" for i in range(x.shape[1])]
    #    feat_names = [i for i in range(x.shape[1])]
    #    val_counts = pd.Series([feat_names[i] for el in all_inds for i in range(len(el.genes)) if el.genes[i] is True]).value_counts()
    #    #return ", ".join([f"{k} ({v})" for k, v in val_counts.head(15).items()])
    #    if feature_names is None:
    #        return ";".join([str(k) for k, _ in val_counts.head(10).items()])
    #    
    #    return ";".join([feature_names[k]
    #                     .replace("equ__", "")
    #                     .replace("__solidus_val", "")
    #                     .replace("OH__temper_designation", "temper")
    #                     .replace("OH__", "")
    #                     for k, _ in val_counts.head(10).items()])

    #stats.register("features", lambda x: get_feature_counts(x))

    def get_features_of_best_ind(all_inds):
        best_ind = max(all_inds, key=lambda x: x.fitness.values[0])
        if feature_names is None:
            feat_names = [str(i) for i in range(x.shape[1]) if isinstance(best_ind.genes[i], bool) and best_ind.genes[i]]
        else:
            feat_names = [el for i, el in enumerate(feature_names) if isinstance(best_ind.genes[i], bool) and best_ind.genes[i]]
       
        return ";".join([name
                         .replace("equ__", "")
                         .replace("__solidus_val", "")
                         .replace("OH__temper_designation", "temper")
                         .replace("OH__", "")
                         for name in feat_names])

    stats.register("mean_n_feat", lambda x: np.mean([sum(el.genes[:n_feat]) for el in x]))
    stats.register("feat. of best ind.", lambda x: get_features_of_best_ind(x))

    pop, log, tracked_genes = generation_ea_simple(pop, toolbox, cxpb=mating_prob, mutpb=prob_mutate_individual, ngen=n_generations, stats=stats, halloffame=hof, verbose=True, constant_seed=constant_seed, re_eval_all=re_eval_all)

    return hof[0], log, tracked_genes


def visualize_tracked_genes(tracked_genes, model_name, selected_cols, all_feat_columns, enable_plots=False, img_dir="./publication/images", suffix=""):
    # visualize tracked genes
    n_feat = len([el for el in tracked_genes[0][0][0] if isinstance(el, bool)])
    tracked_features = [[np.arange(n_feat)[[f for f in ind[0] if isinstance(f, bool)]] for ind in el] for el in tracked_genes.values()]
    tracked_fitnesses = [[ind[1].values[0].item() for ind in el] for el in tracked_genes.values()]

    df_fitnesses = pd.DataFrame(tracked_fitnesses)
    df_feat = pd.DataFrame(tracked_features)

    fig, ax = plt.subplots(1, 1, figsize=(15, 20))
    fig.subplots_adjust(right=0.87, left=0.25, top=0.95, bottom=0.05)
    cmap = plt.get_cmap("viridis")
    max_fitness = df_fitnesses.values.flatten().max()
    min_fitness = np.quantile(df_fitnesses.values.flatten(), 0.05)
    norm = matplotlib.colors.Normalize(vmin=min_fitness, vmax=max_fitness)
    for gen, ((_, fitnesses), (_, features)) in enumerate(zip(df_fitnesses.iterrows(), df_feat.iterrows())):
        for feat, fit in zip(features, fitnesses):
            color = cmap(norm(fit))
            plt.scatter([gen]*len(feat), feat, c=color, cmap="viridis", s=12, alpha=0.05)
    idx_selected_cols = [i for i, col in enumerate(all_feat_columns) if col in selected_cols]
    plt.yticks(idx_selected_cols, selected_cols)
    for tick in ax.get_yticklabels():
        tick.set_rotation(30)

    plt.xlabel("Generation")
    # don't show transperancy in the colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Fitness")
    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, f"tracked_genes_{model_name}{suffix}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved image to {img_path}")

def __deprecated__visualize_tracked_genes_with_params(tracked_genes, model_name, selected_cols, all_feat_columns, selected_params, enable_plots=False, img_dir="./publication/images", height_ratio_subpl = 0.3, height_ratio_feat=5, fitness_thr_plot=None, suffix=""):
    n_feat = len([el for el in tracked_genes[0][0][0] if isinstance(el, bool)])
    tracked_features = [[np.arange(n_feat)[[f for f in ind[0] if isinstance(f, bool)]] for ind in el] for el in tracked_genes.values()]
    tracked_fitnesses = [[ind[1].values[0].item() for ind in el] for el in tracked_genes.values()]

    df_fitnesses = pd.DataFrame(tracked_fitnesses)
    df_feat = pd.DataFrame(tracked_features)


    # create a plot just for model parameters
    cmap = plt.get_cmap("viridis")
    max_fitness = df_fitnesses.values.flatten().max()
    min_fitness = np.quantile(df_fitnesses.values.flatten(), 0.05)
    norm = matplotlib.colors.Normalize(vmin=min_fitness, vmax=max_fitness)
    model_params = [d for d in tracked_genes[0][0][0] if isinstance(d, dict)]
    model_params_changeable = [d for d in model_params if "values" in d or "dt" in d]

    # get the generation where the mean fitness first passes the 0.95 quantile of all mean fitnesses
    if fitness_thr_plot is not None:
        mean_fitnesses = df_fitnesses.mean(1)
        gen_thr = mean_fitnesses[mean_fitnesses > fitness_thr_plot].index[0]

    # number of subplots in Plot 1
    N = len(model_params_changeable)
    # figure with a custom grid specification to accommodate both plots
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(N + 1, 1, figure=fig, height_ratios=[height_ratio_subpl]*N + [height_ratio_feat], hspace=0.)

    # Create a scalar mappable instance for the colorbar
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7]) # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, label='Fitness')
    cbar_ax.yaxis.label.set_size(15)

    fig.subplots_adjust(right=0.87, left=0.25, top=0.95, bottom=0.05)

    # Plot 1: Model Parameters over Generations
    for i, param in enumerate(model_params_changeable):
        ax = fig.add_subplot(gs[i])
        
        param_name = param["name"]
        print("plotting param:", param_name)
        param_idx = [i for i, gene in enumerate(tracked_genes[0][0][0]) if isinstance(gene, dict) and gene["name"] == param_name][0]
        assert tracked_genes[0][0][0][param_idx]["name"] == param_name
        current_per_ind = np.array([[ind[0][param_idx]["current"] for ind in el] for el in tracked_genes.values()])
        fitness_per_ind = np.array([[ind[1].values[0].item() for ind in el] for el in tracked_genes.values()])
        colors = np.array([cmap(norm(f)) for f in fitness_per_ind.flatten()]).reshape(fitness_per_ind.shape[0], fitness_per_ind.shape[1], 4)
        is_categorical = "values" in param

        ax.set_xticks([])

        if not is_categorical:
            ax.set_yscale("log")
            for j in range(current_per_ind.shape[1]):
                c = colors[:,j]
                ax.scatter(np.arange(current_per_ind.shape[0]), current_per_ind[:,j], alpha=0.1, s=8, c=c, cmap="viridis")

            #ax.set_ylabel(param_name)
            #ax.yaxis.set_label_coords(-0.12, 0.5)

            selected_val = selected_params[param_name]
            ax.set_yticks([np.round(selected_val, 3)])
            #text_obj = ax.set_yticklabels([np.round(selected_val, 3)], fontsize=10, color="red", fontweight="bold")[0]
            #ax.get_yticklabels()[0].set_color("red")
            #ax.get_yticklabels()[0].set_fontweight("bold")
            # instead of the above, display <param_name> = <selected_val>, but only make the selected value red and bold
            # to do this, we'll have to create a text object for each part
            y_coord_ytick = ax.get_yticks()[0]

            range_ = ax.get_xlim()[1] - ax.get_xlim()[0]
            x_coord_ytick = - 0.15 * range_
            ax.text(x_coord_ytick, y_coord_ytick, f"{param_name} = ", va="center", ha="right", fontsize=10)
            #ax.text(x_coord_ytick, y_coord_ytick, f"{np.round(selected_val,3)}", va="center", ha="right", fontsize=10, color="red", fontweight="bold")

            ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5) # this messes up the yticks again... not sure how to fix it
        else:
            classes = param["values"]
            le = LabelEncoder()
            le.fit(classes)
            for j in range(current_per_ind.shape[1]):
                c = colors[:,j]
                current_per_ind[:,j] = le.transform(current_per_ind[:,j])
                ax.scatter(np.arange(current_per_ind.shape[0]), current_per_ind[:,j], alpha=0.1, s=8, c=c, cmap="viridis")

            #ax.set_ylabel(param_name)
            ax.set_yticks(np.arange(len(classes)))
            ax.set_yticklabels(classes)
            #ax.yaxis.set_label_coords(-0.12, 0.5)
            ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5)

            # set yticklabel of selected value to red and bold
            selected_val = selected_params[param_name]
            selected_idx = le.transform([selected_val])[0]
            ax.get_yticklabels()[selected_idx].set_color("red")
            ax.get_yticklabels()[selected_idx].set_fontweight("bold")
            # set font size to 10 only for the selected value, the others to 7
            for tick in ax.get_yticklabels():
                if tick.get_text() == selected_val:
                    tick.set_fontsize(10)
                else:
                    tick.set_fontsize(7)

            y_coord_ytick = ax.get_yticks()[selected_idx]
            range_ = ax.get_xlim()[1] - ax.get_xlim()[0]
            x_coord_ytick = - 0.15 * range_
            ax.text(x_coord_ytick, y_coord_ytick, f"{param_name} = ", va="center", ha="right", fontsize=10)

        if fitness_thr_plot is not None:
            ax.axvline(gen_thr, color="red", linestyle="--", alpha=0.5)


    #fig.text(0.1, 0.75, "Model Parameters", va='center', rotation='vertical', fontsize=15)
    fig.text(0.08, 0.85, "Model Parameters", va='center', rotation='vertical', fontsize=15)

    # Plot 2: Features over Generations
    ax_features = fig.add_subplot(gs[N])
    cmap = plt.get_cmap("viridis")
    max_fitness = df_fitnesses.values.flatten().max()
    min_fitness = np.quantile(df_fitnesses.values.flatten(), 0.05)
    norm = mcolors.Normalize(vmin=min_fitness, vmax=max_fitness)
    n_gen = len(tracked_genes)
    print_every = n_gen // 20
    t = time.time()
    for gen, ((_, fitnesses), (_, features)) in enumerate(zip(df_fitnesses.iterrows(), df_feat.iterrows())):
        if (gen+1) % print_every == 0:
            print(f"gen: {gen+1}/{n_gen} (time for last {print_every} generations: {time.time()-t:.2f}s)")
            t = time.time()
        for feat, fit in zip(features, fitnesses):
            color = cmap(norm(fit))
            ax_features.scatter([gen]*len(feat), feat, c=color, cmap="viridis", s=8, alpha=0.05)

    print("scatter plot done - setting ticks and labels...")

    #idx_selected_cols = [i for i, col in enumerate(all_feat_columns) if col in selected_cols]
    #ax_features.set_yticks(idx_selected_cols)
    #ax_features.set_yticklabels(selected_cols)
    #for tick in ax_features.get_yticklabels():
    #    tick.set_rotation(30)

    # instead of the above, show all cols/yticklabels, but make the selected ones red and bold
    ax_features.set_yticks(range(len(all_feat_columns)))
    ax_features.set_yticklabels(all_feat_columns)
    for tick in ax_features.get_yticklabels():
        tick.set_rotation(30)
        if tick.get_text() in selected_cols:
            tick.set_color("red")
            tick.set_fontweight("bold")
        else:
            tick.set_fontsize(7)
    
    ax_features.set_xlabel("Generation")
    ax_features.set_ylabel("Features", fontsize=15)
    if fitness_thr_plot is not None:
        ax_features.axvline(gen_thr, color="red", linestyle="--", alpha=0.5)

    print("done! saving/showing plot...")

    if enable_plots:
        plt.show()
    else:
        img_path = os.path.join(img_dir, f"tracked_genes_{model_name}{suffix}_with_params.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved image to {img_path}")


def visualize_tracked_genes_with_params(tracked_genes, model_name, selected_cols, all_feat_columns, selected_params, enable_plots=False, img_dir="./publication/images", height_ratio_subpl = 0.3, height_ratio_feat=5, fitness_thr_plot=None, suffix="", heatmap_parition_size=None):

    general_font_size = 20
    cbar_font_size = 30
    yticklabels_fontsize = 22
    yticklabels_fontsize_small = 14
    yaxis_overview_font_size = 27
    
    plt.rcParams.update({'font.size': general_font_size})  # Adjust the number to your preferred size

    n_feat = len([el for el in tracked_genes[0][0][0] if isinstance(el, bool)])
    tracked_features = [[np.arange(n_feat)[[f for f in ind[0] if isinstance(f, bool)]] for ind in el] for el in tracked_genes.values()]
    tracked_fitnesses = [[float(ind[1].values[0]) for ind in el] for el in tracked_genes.values()]

    df_fitnesses = pd.DataFrame(tracked_fitnesses)
    df_feat = pd.DataFrame(tracked_features)


    # create a plot just for model parameters
    cmap = plt.get_cmap("viridis")
    #max_fitness = df_fitnesses.values.flatten().max()
    #max_fitness = np.quantile(df_fitnesses.values.flatten(),0.95)

    #fitnesses_min0 = df_fitnesses.values.flatten()
    #fitnesses_min0[fitnesses_min0 < 0] = 0
    #min_fitness = np.quantile(fitnesses_min0, 0.05) # for colorbar

    max_fitness = 0.8
    min_fitness = 0.7

    print(f"max_fitness (colorbar): {max_fitness}, min_fitness (colorbar): {min_fitness}")

    norm = matplotlib.colors.Normalize(vmin=min_fitness, vmax=max_fitness)
    model_params = [d for d in tracked_genes[0][0][0] if isinstance(d, dict)]
    model_params_changeable = [d for d in model_params if "values" in d or "dt" in d]

    param_dfs = {}
    for _, param in enumerate(model_params_changeable):
        tracked_temp = [[np.array([
                ind[0][i]["current"] for i, gene in enumerate(ind[0]) if type(gene) is dict and gene["name"] == param["name"]
            ]) for ind in el] for el in tracked_genes.values()]
        df_temp = pd.DataFrame(tracked_temp)
        param_dfs[param["name"]] = df_temp.copy()

    # get the generation where the mean fitness first passes the 0.95 quantile of all mean fitnesses
    if fitness_thr_plot is not None:
        mean_fitnesses = df_fitnesses.mean(1)
        gen_thr = mean_fitnesses[mean_fitnesses > fitness_thr_plot].index[0]

    # number of subplots in Plot 1
    N = len(model_params_changeable) if selected_params["kernel"] == "poly" else len(model_params_changeable) - 1
    # figure with a custom grid specification to accommodate both plots
    #fig = plt.figure(figsize=(15, 20))
    fig = plt.figure(figsize=(30, 40))
    gs = gridspec.GridSpec(N + 1, 1, figure=fig, height_ratios=[height_ratio_subpl]*N + [height_ratio_feat], hspace=0.)

    # Create a scalar mappable instance for the colorbar
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.9, 0.15, 0.04, 0.7]) # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, label='Fitness')
    cbar_ax.yaxis.label.set_size(cbar_font_size)

    fig.subplots_adjust(right=0.87, left=0.25, top=0.95, bottom=0.05)

    n_gen = len(tracked_genes)

    # Plot 1: Model Parameters over Generations
    i = 0
    for param in model_params_changeable:
        print(i)
        
        param_name = param["name"]
        print("plotting param:", param_name)

        if param_name=="degree":
            not_poly = selected_params["kernel"] != "poly"
            print(f"parameter degree, not_poly: {not_poly}")
            if not_poly:
                print("skipping degree plot")
                continue

        ax = fig.add_subplot(gs[i])

        param_idx = [i for i, gene in enumerate(tracked_genes[0][0][0]) if isinstance(gene, dict) and gene["name"] == param_name][0]
        assert tracked_genes[0][0][0][param_idx]["name"] == param_name
        current_per_ind = np.array([[ind[0][param_idx]["current"] for ind in el] for el in tracked_genes.values()])
        fitness_per_ind = np.array([[float(ind[1].values[0]) for ind in el] for el in tracked_genes.values()])
        colors = np.array([cmap(norm(f)) for f in fitness_per_ind.flatten()]).reshape(fitness_per_ind.shape[0], fitness_per_ind.shape[1], 4)
        is_categorical = "values" in param

        ax.set_xticks([])

        prefix_offset_factor = 0.13
        if not is_categorical:
            #ax.set_yscale("log")
            for j in range(current_per_ind.shape[1]):
                c = colors[:,j]
                ax.scatter(np.arange(current_per_ind.shape[0]), current_per_ind[:,j], alpha=0.1, s=8, c=c, cmap="viridis", norm=norm)

            #ax.set_ylabel(param_name)
            #ax.yaxis.set_label_coords(-0.12, 0.5)

            selected_val = selected_params[param_name]
            label_text = f"{np.round(selected_val, 2)}" if param_name != "tol" else f"{selected_val:.1e}"
            ax.set_yticks([selected_val])
            ax.set_yticklabels([label_text])
            ax.get_yticklabels()[0].set_color("red")
            ax.get_yticklabels()[0].set_fontweight("bold")
            ax.get_yticklabels()[0].set_fontsize(yticklabels_fontsize)
            #text_obj = ax.set_yticklabels([np.round(selected_val, 3)], fontsize=10, color="red", fontweight="bold")[0]
            #ax.get_yticklabels()[0].set_color("red")
            #ax.get_yticklabels()[0].set_fontweight("bold")
            # instead of the above, display <param_name> = <selected_val>, but only make the selected value red and bold
            # to do this, we'll have to create a text object for each part
            y_coord_ytick = ax.get_yticks()[0]

            range_ = ax.get_xlim()[1] - ax.get_xlim()[0]
            x_coord_ytick = - prefix_offset_factor * range_
            ax.text(x_coord_ytick, y_coord_ytick, f"{param_name} = ", va="center", ha="right", fontsize=yticklabels_fontsize)
            #ax.text(x_coord_ytick, y_coord_ytick, f"{np.round(selected_val,3)}", va="center", ha="right", fontsize=10, color="red", fontweight="bold")

            ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5) # this messes up the yticks again... not sure how to fix it
        else:
            classes = param["values"]
            le = LabelEncoder()
            le.fit(classes)
            for j in range(current_per_ind.shape[1]):
                c = colors[:,j]
                current_per_ind[:,j] = le.transform(current_per_ind[:,j])
                ax.scatter(np.arange(current_per_ind.shape[0]), current_per_ind[:,j].astype(int), alpha=0.1, s=8, c=c, cmap="viridis", norm=norm)

            ax.set_yticks(le.transform(le.classes_))
            ax.set_yticklabels(le.classes_)

            ax.yaxis.set_label_coords(-0.12, 0.5)
            ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5)

            # set yticklabel of selected value to red and bold
            selected_val = selected_params[param_name]
            selected_idx = le.transform([selected_val])[0]
            ax.get_yticklabels()[selected_idx].set_color("red")
            ax.get_yticklabels()[selected_idx].set_fontweight("bold")
            ax.get_yticklabels()[selected_idx].set_fontsize(yticklabels_fontsize)
            # set font size to 10 only for the selected value, the others to 7
            for tick in ax.get_yticklabels():
                if tick.get_text() == selected_val:
                    tick.set_fontsize(yticklabels_fontsize)
                else:
                    tick.set_fontsize(yticklabels_fontsize_small)

            y_coord_ytick = ax.get_yticks()[selected_idx]
            range_ = ax.get_xlim()[1] - ax.get_xlim()[0]
            x_coord_ytick = - prefix_offset_factor * range_
            ax.text(x_coord_ytick, y_coord_ytick, f"{param_name} = ", va="center", ha="right", fontsize=yticklabels_fontsize)

        if fitness_thr_plot is not None:
            ax.axvline(gen_thr, color="red", linestyle="--", alpha=0.5)

        i += 1


    #fig.text(0.1, 0.85, "Model Parameters", va='center', rotation='vertical', fontsize=yaxis_overview_font_size)
    fig.text(0.12, 0.85, "Model Parameters", va='center', rotation='vertical', fontsize=yaxis_overview_font_size)

    # Plot 2: Features over Generations
    ax_features = fig.add_subplot(gs[N])
    # instead of scatter plot, make a heatmap, color is mean fitness with an alpha weighted by the number of times the feature was selected
    feat_counts = np.zeros((n_gen, n_feat))
    fitnesses_sum = np.zeros((n_gen, n_feat))
    for gen, ((_, fitnesses), (_, features)) in enumerate(zip(df_fitnesses.iterrows(), df_feat.iterrows())):
        for feat, fit in zip(features, fitnesses):
            for f in feat:
                feat_counts[gen, f] += 1 ###############
                fitnesses_sum[gen, f] += max(0,fit) ################
                #fitnesses_sum[gen, f] = max(0, max(fitnesses_sum[gen, f], fit)) ################

    fitnesses_sum /= feat_counts ###############

    if heatmap_parition_size is None:
        fitnesses_sum[np.isnan(fitnesses_sum)] = 0
    alphas = feat_counts / feat_counts.max() ################
    #alphas = np.ones((n_gen, n_feat)) ################

    if heatmap_parition_size is not None:
        # partition into groups of 100 generations
        n_partitions = n_gen // heatmap_parition_size + (1 if n_gen % heatmap_parition_size > 0 else 0)
        # for each partition, average the fitnesses
        fitnesses_sum_partitioned = np.zeros((n_partitions, n_feat))
        alphas_partitioned = np.zeros((n_partitions, n_feat))

        for i in range(n_partitions):
            start = i * heatmap_parition_size
            end = (i+1) * heatmap_parition_size
            fitnesses_sum_partitioned[i] = np.nanmean(fitnesses_sum[start:end],axis=0)
            alphas_partitioned[i] = alphas[start:end].mean(0)

        ax_features.imshow(fitnesses_sum_partitioned.T, cmap="viridis", norm=norm, aspect="auto", alpha=alphas_partitioned.T, extent=[0, n_gen, 0, n_feat], origin="lower")
    else:
        ax_features.imshow(fitnesses_sum.T, cmap="viridis", norm=norm, aspect="auto", alpha=alphas.T, extent=[0, n_gen, 0, n_feat], origin="lower")

    print("scatter plot done - setting ticks and labels...")

    #idx_selected_cols = [i for i, col in enumerate(all_feat_columns) if col in selected_cols]
    #ax_features.set_yticks(idx_selected_cols)
    #ax_features.set_yticklabels(selected_cols)
    #for tick in ax_features.get_yticklabels():
    #    tick.set_rotation(30)

    # instead of the above, show all cols/yticklabels, but make the selected ones red and bold
    ax_features.set_yticks(np.arange(len(all_feat_columns))+0.5)
    #ax_features.set_yticklabels(all_feat_columns)

    def transform_col_label(label):
        equ_regex = re.compile(r"equ__(\w+)__.*")
        scheil_regex = re.compile(r"scheil__(\w+)__.*")

        if equ_regex.match(label):
            phase = equ_regex.match(label).group(1)
            return f"{phase} (Equ.)".replace("_", " ")
        elif scheil_regex.match(label):
            phase = scheil_regex.match(label).group(1)
            return f"{phase} (Sch.)".replace("_", " ")

        return label.replace("_", " ")

    labels_transformed = [transform_col_label(label) for label in all_feat_columns]
    ax_features.set_yticklabels(labels_transformed)
    selected_cols_transformed = [transform_col_label(label) for label in selected_cols]

    for tick in ax_features.get_yticklabels():
        tick.set_rotation(30)
        if tick.get_text() in selected_cols_transformed:
            tick.set_color("red")
            tick.set_fontweight("bold")
        else:
            tick.set_fontsize(yticklabels_fontsize_small)
    
    ax_features.set_xlabel("Generation")
    ax_features.set_ylabel("Features", fontsize=yaxis_overview_font_size)
    if fitness_thr_plot is not None:
        ax_features.axvline(gen_thr, color="red", linestyle="--", alpha=0.5)

    print("done! saving/showing plot...")

    if enable_plots:
        plt.show()
    else:
        #img_path = os.path.join(img_dir, f"tracked_genes_{model_name}{suffix}_with_params.pdf")
        img_path = os.path.join(img_dir, f"tracked_genes_{model_name}{suffix}_with_params.png")
        plt.savefig(img_path)
        plt.close()
        print(f"saved image to {img_path}")
