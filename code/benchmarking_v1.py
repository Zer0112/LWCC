from dataclasses import dataclass
from functools import lru_cache
from os import makedirs
import train_chain as tc
import pyglove as pg
import pandas as pd
import pickle
import csv
import numpy as np
from tqdm import tqdm
import random
import argparse
import warnings
from sklearnex import patch_sklearn

patch_sklearn(verbose=False)

classifiers_list_reduced = [cl for cl in tc._dict_of_classifiers.keys()]
classifiers_list_reduced2 = [cl for cl in tc._dict_of_classifiers.keys() if cl!="knn"]
# classifiers_list_reduced=classifiers_list_reduced[:3]+["random_forest"]
_dataset_names_nr_labels = {
    "birds": 19,
    "yeast": 14,
    "emotions": 6,
    "scene": 6,
    "reuters": 7,
    # "enron": 53,
    "image": 5,
    "tmc2007_500": 22,
    # "medical":45,
    "rcv1subset1": 101,
}

_algo_evo_dict = {
    "neat": pg.evolution.neat,
    "regularized_evolution": pg.evolution.regularized_evolution,
    "hill": pg.evolution.hill_climb,
    "random": pg.geno.Random,
}
_algo_evo_dict2 = {
    "regularized_evolution": pg.evolution.regularized_evolution,
    "hill": pg.evolution.hill_climb,
    "random": pg.geno.Random,
}
parser = argparse.ArgumentParser(description="Args for the training loop")
parser.add_argument("seed", type=int, help="an integer for the seed")
parser.add_argument("dataset_name", type=str, help="a string for the dataset name")
parser.add_argument(
    "clear", type=bool, help="a boolean for clearing the results", default=False
)
parser.add_argument(
    "--prog_bar",
    type=bool,
    help="a boolean for showing the progress bar",
    default=False,
)
parser.add_argument(
    "--neat",
    type=bool,
    help="a boolean for including running neat in the loop",
    default=False,
)


def benchmark(
    dataset_name="yeast",
    eval_metric="f1",
    algo_evo="neat",
    seed_alg=42,
    order=None,
    num_examples=10,
    number_of_labels=14,
    save=False,
    early_stopping=0,
):
    search_space = get_search_space(number_of_labels)
    history = []
    best_history = []
    best_chain, min_value = None, None
    algo = _algo_evo_dict[algo_evo](seed=seed_alg)
    for lst, feedback in pg.sample(search_space, algo, num_examples=num_examples):
        l = tc.feedback_ga(
            dataset_name=dataset_name,
            classifier_list=lst,
            eval_metric=eval_metric,
            order=order,
        )
        history.append(l)
        if min_value is None or min_value > l:
            best_chain, min_value = lst, l
            best_history.append((len(history), l))
        feedback(-l)
        if (
            early_stopping
            and len(history) >= 200
            and len(history) - best_history[-1][0] > early_stopping
        ):
            break

    order = [i for i in range(number_of_labels)] if order is None else tuple(order)
    order_str = "_".join([str(i) for i in order])
    result = (
        history,
        best_history,
        best_chain,
        min_value,
        algo_evo,
        seed_alg,
        eval_metric,
        order,
        num_examples,
        dataset_name,
    )
    save_bench(
        dataset_name,
        eval_metric,
        algo_evo,
        seed_alg,
        order,
        num_examples,
        number_of_labels,
        save,
        best_chain,
        min_value,
        order_str,
    )

    return result
def peak_benchmark(
    dataset_name="yeast",
    eval_metric="f1",
    algo_evo="neat",
    seed_alg=42,
    order=None,
    num_examples=10,
    number_of_labels=14,
    save=False,
    early_stopping=0,
):
    search_space = get_search_space2(number_of_labels)
    history = []
    best_history = []
    best_chain, min_value = None, None
    algo = _algo_evo_dict[algo_evo](seed=seed_alg)
    for lst, feedback in pg.sample(search_space, algo, num_examples=num_examples):
        order = lst.order
        lst=lst.cl_list
        l = tc.feedback_ga(
            dataset_name=dataset_name,
            classifier_list=lst,
            eval_metric=eval_metric,
            order=order,
        )
        history.append(l)
        if min_value is None or min_value > l:
            best_chain, min_value = lst, l
            best_history.append((len(history), l))
        feedback(-l)
        if (
            early_stopping
            and len(history) >= 200
            and len(history) - best_history[-1][0] > early_stopping
        ):
            break

    order = [i for i in range(number_of_labels)] if order is None else tuple(order)
    order_str = "_".join([str(i) for i in order])
    result = (
        history,
        best_history,
        best_chain,
        min_value,
        algo_evo,
        seed_alg,
        eval_metric,
        order,
        num_examples,
        dataset_name,
    )
    save_bench(
        dataset_name,
        eval_metric,
        algo_evo,
        seed_alg,
        order,
        num_examples,
        number_of_labels,
        save,
        best_chain,
        min_value,
        order_str,
        not_peak=False,
    )

    return result


def save_bench(
    dataset_name,
    eval_metric,
    algo_evo,
    seed_alg,
    order,
    num_examples,
    number_of_labels,
    save,
    best_chain,
    min_value,
    order_str,
    not_peak=True,
):
    if save:
        # save result with pickle
        # does not save history or best_history
        
        if not_peak:
            makedirs(f"results/{dataset_name}/{order_str}", exist_ok=True)
            name = f"results/{dataset_name}/{order_str}/{algo_evo}_S{seed_alg}_eval_{eval_metric}_O{order_str}_N{num_examples}_L{number_of_labels}"
        else:
            makedirs(f"presults/{dataset_name}/{order_str}", exist_ok=True)
            name = f"presults/{dataset_name}/{order_str}/{algo_evo}_S{seed_alg}_eval_{eval_metric}_O{order_str}_N{num_examples}_L{number_of_labels}"

        # evaluate the best chain and save it in the results
        names_e, scores_e = tc.evaluate_chain(
            dataset_name=dataset_name,
            classifier_list=best_chain,
            order=order,
            name_of_results_file=f"{name}_results.pkl",
        )
        n_score = np.array(scores_e)
        temp_arr = n_score[:, -8:]
        temp_arr = np.array(temp_arr, dtype=float)
        temp_arr = np.mean(temp_arr, axis=0)

        # make a dataframe out of the result
        temp_result = [
            dataset_name,
            float(min_value),
            algo_evo,
            seed_alg,
            eval_metric,
            order_str,
            num_examples,
            number_of_labels,
            *[i for i in best_chain],
            *list(temp_arr),
        ]
        names = [
            "dataset",
            "best_value",
            "algo_name",
            "seed_alg",
            "eval_metric",
            "order",
            "num_examples",
            "number_of_labels",
            *[f"classifiers_{i}" for i in range(number_of_labels)],
            *names_e[-8:],
        ]
        dict_temp_result = dict(zip(names, temp_result))

        # save the bench results as csv
        with open(f"{name}_bench.csv", "w") as f:
            w = csv.DictWriter(f, dict_temp_result.keys())
            w.writeheader()
            w.writerow(dict_temp_result)


@lru_cache(1)
def get_search_space(number_of_labels):
    search_space = pg.List([pg.oneof(classifiers_list_reduced)] * number_of_labels)
    return search_space


@pg.symbolize
@dataclass()
class cl_conf:
    cl_list: list
    order: list


@lru_cache(1)
def get_search_space2(number_of_labels):
    search_space = cl_conf(
        cl_list=pg.List([pg.oneof(classifiers_list_reduced2)] * number_of_labels),
        order=pg.permutate(list(range(number_of_labels))),
    )
    return search_space


def benchmark_greedy1(
    dataset_name="yeast", eval_metric="f1", order=None, number_of_labels=14, noise=0
):
    full_clasifier_list = [None for _ in range(number_of_labels)]
    order = [i for i in range(number_of_labels)] if order is None else tuple(order)
    order_str = "_".join([str(i) for i in order])
    algo_evo = "greedy1" if noise == 0 else "greedy1_noise"

    best1 = [None for _ in range(number_of_labels)]
    for pos in range(0, number_of_labels):
        for cl in classifiers_list_reduced:
            greedy_feedback = tc.feedback_greedy(
                dataset_name=dataset_name,
                classifier=cl,
                pos=pos,
                total_len=number_of_labels,
                order=order,
                n_splits=3,
                random_state=42,
                train_if_not_exist=True,
                eval_metric=eval_metric,
                noise=noise,
            )

            if best1[pos] is None or greedy_feedback < best1[pos]:
                best1[pos] = greedy_feedback
                full_clasifier_list[pos] = cl
    save_bench(
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        algo_evo=algo_evo,
        seed_alg=0,
        order=order,
        num_examples=0,
        number_of_labels=number_of_labels,
        save=True,
        best_chain=full_clasifier_list,
        min_value=best1[-1],
        order_str=order_str,
    )
    n1, s1 = tc.evaluate_chain(
        dataset_name=dataset_name,
        classifier_list=full_clasifier_list,
        order=order,
        n_splits=3,
        random_state=42,
        train_if_not_exist=True,
    )
    df = pd.DataFrame(s1, columns=n1)
    agg_df = df.groupby("list_classifiers", as_index=False).agg(
        {
            "score_acc": ["mean", "std"],
            "score_f1_macro": ["mean", "std"],
            "score_f1_micro": ["mean", "std"],
            "score_hamming": ["mean", "std"],
            "score_val_acc": ["mean", "std"],
            "score_val_f1_macro": ["mean", "std"],
            "score_val_f1_micro": ["mean", "std"],
            "score_val_hamming": ["mean", "std"],
        }
    )
    agg_df.columns = agg_df.columns.map("_".join)
    agg_df

    return full_clasifier_list, df, agg_df


def benchmark_greedy2(
    dataset_name="yeast", eval_metric="f1", order=None, number_of_labels=14
):
    full_clasifier_list = [None for _ in range(number_of_labels)]
    # order=order if order is not None else [i for i in range(number_of_labels)]
    order = [i for i in range(number_of_labels)] if order is None else tuple(order)
    order_str = "_".join([str(i) for i in order])
    best1 = [None for _ in range(number_of_labels)]
    for pos in range(0, number_of_labels):
        for cl in classifiers_list_reduced:
            cl_list = full_clasifier_list[:pos] + [cl]
            greedy_feedback = tc.feedback_greedy2(
                cl_list,
                dataset_name=dataset_name,
                pos=pos,
                order=order,
                n_splits=3,
                random_state=42,
                train_if_not_exist=True,
                eval_metric=eval_metric,
            )

            if best1[pos] is None or greedy_feedback < best1[pos]:
                best1[pos] = greedy_feedback
                full_clasifier_list[pos] = cl

    save_bench(
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        algo_evo="greedy2",
        seed_alg=0,
        order=order,
        num_examples=0,
        number_of_labels=number_of_labels,
        save=True,
        best_chain=full_clasifier_list,
        min_value=best1[-1],
        order_str=order_str,
    )
    n1, s1 = tc.evaluate_chain(
        dataset_name=dataset_name,
        classifier_list=full_clasifier_list,
        order=order,
        n_splits=3,
        random_state=42,
        train_if_not_exist=True,
    )
    df = pd.DataFrame(s1, columns=n1)
    agg_df = df.groupby("list_classifiers", as_index=False).agg(
        {
            "score_acc": ["mean", "std"],
            "score_f1_macro": ["mean", "std"],
            "score_f1_micro": ["mean", "std"],
            "score_hamming": ["mean", "std"],
            "score_val_acc": ["mean", "std"],
            "score_val_f1_macro": ["mean", "std"],
            "score_val_f1_micro": ["mean", "std"],
            "score_val_hamming": ["mean", "std"],
        }
    )
    agg_df.columns = agg_df.columns.map("_".join)
    agg_df

    return full_clasifier_list, df, agg_df


def benchmark_base(dataset_name, eval_metric, order, number_of_labels):
    order = [i for i in range(number_of_labels)] if order is None else tuple(order)
    order_str = "_".join([str(i) for i in order])
    for i, cl in enumerate(classifiers_list_reduced):
        best_chain = [cl] * number_of_labels
        save_bench(
            dataset_name=dataset_name,
            eval_metric=eval_metric,
            algo_evo=f"base",
            seed_alg=i,
            order=order,
            num_examples=0,
            number_of_labels=number_of_labels,
            save=True,
            best_chain=best_chain,
            min_value=0,
            order_str="_".join([str(i) for i in order]),
        )


def random_order(number_of_labels, seed=1):
    """
    The function generates a random order list of numbers from 0 to (number_of_labels - 1) using a given
    seed value.

    :param number_of_labels: The number of labels is the total number of items or categories that need
    to be randomly ordered. For example, if you have a list of 10 items and you want to randomize their
    order, the number of labels would be 10
    :param seed: The seed parameter is an optional input that sets the random seed for the random number
    generator. This ensures that the same sequence of random numbers is generated every time the
    function is called with the same seed value. If no seed value is provided, the default value of 1 is
    used, defaults to 1 (optional)
    :return: a list of integers from 0 to `number_of_labels` shuffled randomly using the `shuffle()`
    method of the `random` module. The order of the list is determined by the `seed` parameter, which is
    set to 1 by default.
    """
    rand = random.Random(seed)
    random_order_list = list(range(number_of_labels))
    rand.shuffle(random_order_list)
    return random_order_list


def train_yeast(order=None):
    """Example trainings loop for testing"""
    for alg in tqdm(_algo_evo_dict.keys(), desc="outer loop", position=0):
        for i in tqdm(range(0, 20), desc=f"inner loop {alg}", position=1, leave=False):
            bench1 = benchmark(
                dataset_name="yeast",
                eval_metric="f1",
                algo_evo=alg,
                order=order,
                num_examples=1000,
                number_of_labels=14,
                save=True,
                seed_alg=i,
                early_stopping=150,
            )


def train_full_loop_reduced(dataset: str, order=None, clean=False):
    """Example trainings loop for testing - without neat algorithm"""
    nr_labels = _dataset_names_nr_labels.get(dataset)
    for alg in tqdm(_algo_evo_dict2.keys(), desc="outer loop", position=0):
        for i in tqdm(range(0, 20), desc=f"inner loop {alg}", position=1, leave=False):
            bench1 = benchmark(
                dataset_name=dataset,
                eval_metric="f1",
                algo_evo=alg,
                order=order,
                num_examples=1000,
                number_of_labels=nr_labels,
                save=True,
                seed_alg=i,
                early_stopping=150,
            )
            # del bench1
    if clean:
        tc.clear_cache_for_dataset_order(dataset_name=dataset, order=order)
        
def train_full_loop_peak(dataset: str, clean=False):
    """Example trainings loop for testing - without neat algorithm"""
    nr_labels = _dataset_names_nr_labels.get(dataset)
    algs=["hill","regularized_evolution"]
    algs=["regularized_evolution"]
    for alg in tqdm(algs, desc="outer loop", position=0):
        for i in tqdm(range(10, 20), desc=f"inner loop {alg}", position=1, leave=False):
            bench1 = peak_benchmark(
                dataset_name=dataset,
                eval_metric="f1",
                algo_evo=alg,
                order=None,
                num_examples=1000,
                number_of_labels=nr_labels,
                save=True,
                seed_alg=i,
                early_stopping=150,
            )
            # del bench1



def train_full_loop_reduced_no_progbar(dataset: str, order=None, clean=False):
    """Example trainings loop for testing - without neat algorithm"""
    nr_labels = _dataset_names_nr_labels.get(dataset)
    for alg in _algo_evo_dict2.keys():
        for i in range(0, 20):
            bench1 = benchmark(
                dataset_name=dataset,
                eval_metric="f1",
                algo_evo=alg,
                order=order,
                num_examples=1000,
                number_of_labels=nr_labels,
                save=True,
                seed_alg=i,
                early_stopping=150,
            )
            # del bench1
    if clean:
        tc.clear_cache_for_dataset_order(dataset_name=dataset, order=order)


def train_loop_greedy1(dataset: str):
    nr_labels = _dataset_names_nr_labels.get(dataset)
    benchmark_greedy1(
        dataset_name=dataset,
        eval_metric="f1",
        order=None,
        number_of_labels=nr_labels,
        noise=0,
    )
    for i in tqdm(range(20)):
        order = random_order(nr_labels, seed=i)
        benchmark_greedy1(
            dataset_name=dataset,
            eval_metric="f1",
            order=order,
            number_of_labels=nr_labels,
            noise=0,
        )


def train_loop_greedy1_noise(dataset: str):
    nr_labels = _dataset_names_nr_labels.get(dataset)
    benchmark_greedy1(
        dataset_name=dataset,
        eval_metric="f1",
        order=None,
        number_of_labels=nr_labels,
        noise=0.10,
    )
    for i in tqdm(range(20)):
        order = random_order(nr_labels, seed=i)
        benchmark_greedy1(
            dataset_name=dataset,
            eval_metric="f1",
            order=order,
            number_of_labels=nr_labels,
            noise=0.10,
        )


def train_loop_greedy2(dataset: str):
    nr_labels = _dataset_names_nr_labels.get(dataset)
    benchmark_greedy2(
        dataset_name=dataset, eval_metric="f1", order=None, number_of_labels=nr_labels
    )
    for i in tqdm(range(20)):
        order = random_order(nr_labels, seed=i)
        benchmark_greedy2(
            dataset_name=dataset,
            eval_metric="f1",
            order=order,
            number_of_labels=nr_labels,
        )


def training_loop_base(dataset: str):
    nr_labels = _dataset_names_nr_labels.get(dataset)
    benchmark_base(
        dataset_name=dataset, eval_metric="f1", order=None, number_of_labels=nr_labels
    )
    for i in tqdm(range(20)):
        order = random_order(nr_labels, seed=i)
        benchmark_base(
            dataset_name=dataset,
            eval_metric="f1",
            order=order,
            number_of_labels=nr_labels,
        )


if __name__ == "__main__":
    # ord_ran=random_order(14,seed=2)
    # print(ord_ran)
    # train_yeast(order=ord_ran)
    warnings.simplefilter("ignore")
    args = parser.parse_args()
    nr_labels=_dataset_names_nr_labels.get(args.dataset_name)
    ord_ran=random_order(nr_labels,seed=args.seed)
    # # tc.train_all_cl_loop(6,dataset_name="scene", order=ord_ran)
    if args.prog_bar:
        train_full_loop_reduced(dataset=args.dataset_name, order=ord_ran,clean=args.clear)
    else:
        train_full_loop_reduced_no_progbar(dataset=args.dataset_name, order=ord_ran,clean=args.clear)

    # print(bench1)
    # bench2 = benchmark_greedy1(dataset_name="yeast", eval_metric="f1", order=None,number_of_labels=14,noise=0)
    # print(bench2)
    
    # lst1 = ["birds", "yeast", "emotions", "scene", "reuters", "image"]
    # for d in lst1:
    #     print(d)
    #     training_loop_base(dataset=d)
    
    # for d in lst1:
    #     print(d)
    #     train_loop_greedy1_noise(dataset=d)
    
    # train_full_loop_peak(dataset="emotions", clean=False)
