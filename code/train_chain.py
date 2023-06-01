import csv
import os
import pickle
import shutil
import time
from functools import lru_cache, wraps,partial
from typing import OrderedDict

import numpy as np

import chain_classifier as cc
import dataset_utils as ds
from chain_classifier import chain_configuration_single_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearnex import patch_sklearn

# from knn_faiss import KNeighbors_Faiss

patch_sklearn(verbose=False)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        #
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{total_time:.4f} seconds")
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


_dict_of_classifiers = OrderedDict(
    {
        "naive_bayes": GaussianNB,
        # "MultinomialNB": MultinomialNB,
        # "ComplementNB": ComplementNB,
        "ridge": RidgeClassifier,
        "perceptron": Perceptron,
        "knn": KNeighborsClassifier,
        "svm": SVC,
        "random_forest": partial(RandomForestClassifier,max_features="sqrt"), # max_features="sqrt" to remove the warning of future deprecation
        "decision_tree": DecisionTreeClassifier,
        "logistic_regression": LogisticRegression, # can cause problems with convergence
        # "knn_faiss": KNeighbors_Faiss,
        # "ada_boost": AdaBoostClassifier,
        # "gradient_boosting": GradientBoostingClassifier,
        # "linear_discriminant_analysis": LinearDiscriminantAnalysis,
        # "quadratic_discriminant_analysis": QuadraticDiscriminantAnalysis,
        # "extra_trees": ExtraTreesClassifier,
        # "gaussian_process": GaussianProcessClassifier, # takes too much memory
        # "mlp": MLPClassifier,
    }
)
_dict_of_classifiers_short = {k[:3]:k for k in _dict_of_classifiers.keys()}


                
def train_chain(
    training_classifier_nr,
    dataset_name="yeast",
    classifier_name="logistic_regression",
    random_state=42,
    order=None,
):
    """
    It takes a dataset name, a classifier name, a random state, and an order (which is a list of
    integers) and then it trains a chain of classifiers on the dataset, using the classifier name, and
    the order.

    The order is a list of integers, where each integer represents the position of the classifier in the
    chain.

    For example, if the order is [1,2,3], then the first classifier in the chain will be trained on the
    first feature, the second classifier in the chain will be trained on the second feature, and the
    third classifier in the chain will be trained on the third feature.

    The function then saves the trained classifiers in a folder.

    The folder is named after the dataset name, the order, the position of the classifier in the chain,
    and the classifier name.

    The function also saves the order in the folder name

    :param training_classifier_nr: the position of the classifier in the chain that is trained
    :param dataset_name: the name of the dataset to use, defaults to yeast (optional)
    :param classifier_name: the name of the classifier to use, defaults to logistic_regression
    (optional)
    :param random_state: the random state for the outer and inner folds, defaults to 42 (optional)
    :param order: the order of the classifiers in the chain. If None, then the order is random
    """
    X, Y = ds.fetch_openml_dataset(dataset_name)
    n_splits = 3
    training_classifier_nr = training_classifier_nr
    random_state = random_state

    outer_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )
    inner_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )

    str_order = ""

    for i_outer, (train_index, test_index) in enumerate(outer_fold.split(X)):
        X_train, _ = X.values[train_index], X.values[test_index]
        Y_train, _ = Y.values[train_index], Y.values[test_index]
        for i_inner, (train_index, test_index) in enumerate(inner_fold.split(X_train)):
            X_train_inner, _ = X_train[train_index], X_train[test_index]
            Y_train_inner, _ = Y_train[train_index], Y_train[test_index]

            cl = _dict_of_classifiers[classifier_name]()
            single_chain = chain_configuration_single_fit(
                single_classifier=cl,
                training_classifier_nr=training_classifier_nr,
                order=order,
            )
            cl, order, _ = single_chain.fit(X_train_inner, Y_train_inner)

            str_order = "_".join([str(x) for x in order])
            os.makedirs(
                f"eval/{dataset_name}/{str_order}/pos_{training_classifier_nr}/{classifier_name}",
                exist_ok=True,
            )
            name = f"eval/{dataset_name}/{str_order}/pos_{training_classifier_nr}/{classifier_name}/outer_{i_outer}_inner_{i_inner}_rs_{random_state}.pkl"
            
            with open(name, "wb") as f:
                pickle.dump(cl, f)
                


    # with open(f"eval/{dataset_name}/{str_order}/outer_fold.pkl", "wb") as f:
    #     pickle.dump(outer_fold, f)
    # with open(f"eval/{dataset_name}/{str_order}/inner_fold.pkl", "wb") as f:
    #     pickle.dump(inner_fold, f)


def evaluate_chain(
    dataset_name="yeast",
    classifier_list=["naive_bayes" for _ in range(14)],
    order=None,
    n_splits=3,
    random_state=42,
    train_if_not_exist=True,
    name_of_results_file=None,
):
    """
    It takes a dataset name, a list of classifiers, an order of the classifiers, the number of outer and
    inner folds, the random state and a boolean value that determines if the classifiers should be
    trained if they don't exist.

    It then returns a list of the names of the columns and a list of lists with the results.

    The results are the dataset name, the list of classifiers, the classifiers themselves, the order of
    the classifiers, the outer fold, the inner fold, the accuracy score, the f1 score and the hamming
    loss.

    The function is used to evaluate the classifier chain.

    The function is used in the following way:

    :param dataset_name: The name of the dataset to use, defaults to yeast (optional)
    :param classifier_list: list of classifiers to use in the chain
    :param order: the order in which the classifiers are chained together
    :param n_splits: number of outer and inner folds, defaults to 3 (optional)
    :param random_state: the random seed used for the outer and inner cross-validation, defaults to 42
    (optional)
    :param train_if_not_exist: If the classifier is not trained, then train it, defaults to True
    (optional)
    :return: The name_table is a list of strings that are the names of the columns in the results
    dataframe. The results is a list of lists, where each list is a row in the dataframe.
    """
    X, Y = ds.fetch_openml_dataset(dataset_name)

    results = []

    assert len(classifier_list) == Y.shape[1]

    order = order if order else list(range(len(classifier_list)))

    outer_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )
    inner_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )

    temp_lst_cl = [classifier_list[nr][:3] for nr, _ in enumerate(classifier_list)]
    name_table = [
        "dataset_name",
        "list_classifiers",
        *[f"classifier_nr_{nr}" for nr, _ in enumerate(classifier_list)],
        "str_order",
        "fold_outer",
        "fold_inner",
        "score_acc",
        "score_f1_macro",
        "score_f1_micro",
        "score_hamming",
        "score_val_acc",
        "score_val_f1_macro",
        "score_val_f1_micro",
        "score_val_hamming",
    ]
    str_order: str = "_".join([str(x) for x in order])
    trained_classifiers = [None for _ in classifier_list]
    for cl_name in classifier_list:
        assert cl_name in _dict_of_classifiers.keys()

    for i_outer, (train_index, test_index) in enumerate(outer_fold.split(X)):
        X_train, X_test_outer = X.values[train_index], X.values[test_index]
        Y_train, Y_test_outer = Y.values[train_index], Y.values[test_index]
        for i_inner, (train_index, test_index) in enumerate(inner_fold.split(X_train)):
            _, X_test_inner = X_train[train_index], X_train[test_index]
            _, Y_test_inner = Y_train[train_index], Y_train[test_index]

            for i, cl_name in enumerate(classifier_list):
                cl_file_name = inner_pickle_name(dataset_name, random_state, str_order, cl_name, i_outer, i_inner, i)
                if not os.path.exists(cl_file_name):
                    if train_if_not_exist:
                        train_chain(
                            training_classifier_nr=i,
                            dataset_name=dataset_name,
                            classifier_name=cl_name,
                            order=order,
                            random_state=random_state,
                        )
                    else:
                        raise FileNotFoundError(cl_file_name)

                with open(cl_file_name, "rb") as f:
                    trained_classifiers[i] = pickle.load(f)
            classifier_chain = cc.chaining_together(
                config_classifier=trained_classifiers, order=order
            )

            val_temp = scoring_table(dataset_name, classifier_list, temp_lst_cl, str_order, i_outer, i_inner, X_test_inner, Y_test_inner,X_test_outer,Y_test_outer, classifier_chain)
            results.append(val_temp)
    if name_of_results_file is not None:
        with open(name_of_results_file, "wb") as f:
            pickle.dump((name_table,results), f)
    return name_table, results

def scoring_table(dataset_name, classifier_list, temp_lst_cl, str_order, i_outer, i_inner, X_test_inner, Y_test_inner,X_test_outer, Y_test_outer, classifier_chain):
    pred = classifier_chain.predict(X_test_inner)
    score_acc = accuracy_score(Y_test_inner, pred)
    score_f1_macro = f1_score(Y_test_inner, pred, average="macro")
    score_f1_micro = f1_score(Y_test_inner, pred, average="micro")
    score_hamming = hamming_loss(Y_test_inner, pred)
    
    pred_outer = classifier_chain.predict(X_test_outer)
    score_val_acc = accuracy_score(Y_test_outer, pred_outer)
    score_val_f1_macro = f1_score(Y_test_outer, pred_outer, average="macro")
    score_val_f1_micro = f1_score(Y_test_outer, pred_outer, average="micro")
    score_val_hamming = hamming_loss(Y_test_outer, pred_outer)

    val_temp = [
                dataset_name,
                "_".join(temp_lst_cl),
                *[classifier_list[nr] for nr, _ in enumerate(classifier_list)],
                str_order,
                i_outer,
                i_inner,
                score_acc,
                score_f1_macro,
                score_f1_micro,
                score_hamming,
                score_val_acc,
                score_val_f1_macro,
                score_val_f1_micro,
                score_val_hamming,
            ]
    
    return val_temp

def inner_pickle_name(dataset_name, random_state, str_order, cl_name, i_outer, i_inner, i):
    cl_file_name = f"eval/{dataset_name}/{str_order}/pos_{i}/{cl_name}/outer_{i_outer}_inner_{i_inner}_rs_{random_state}.pkl"
    return cl_file_name



def outer_pickle_name(dataset_name, random_state, str_order, cl_name, i_outer, i):
    cl_file_name = f"eval/{dataset_name}/{str_order}/pos_{i}/{cl_name}/outer_{i_outer}_val_rs_{random_state}.pkl"
    return cl_file_name


def generate_splits_test_inner(X, Y, outer_fold, inner_fold):
    """
    For each outer fold, for each inner fold, return the index of the inner fold, the index of the outer
    fold, the training data for the inner fold, and the labels for the inner fold.

    :param X: The dataframe of features
    :param Y: The target variable
    :param outer_fold: The outer fold is the outer loop of the nested cross-validation. It is used to
    split the data into training and testing sets
    :param inner_fold: The inner fold is the fold that will be used to train the model
    :return: A generator object (inner, outer, X_test_inner, Y_test_inner)
    """
    gen_splits = (
        (
            (i_outer, i_inner),
            X_train_outer[test_index_inner],
            Y_train_outer[test_index_inner],
        )
        for i_outer, (train_index_outer, test_index_outer) in enumerate(
            outer_fold.split(X)
        )
        for i_inner, (train_index_inner, test_index_inner) in enumerate(
            inner_fold.split(X.values[train_index_outer])
        )
        for X_train_outer, Y_train_outer in [
            (X.values[train_index_outer], Y.values[train_index_outer])
        ]
    )
    return gen_splits


def calculate_metric(metric_name, Y_true, Y_pred):
    """
    It takes in a metric name, and the true and predicted labels, and returns the value of the metric.

    The metric name can be one of the following:

    - f1
    - f1_micro
    - acc
    - hamming

    If the metric name is not one of the above, it will raise a ValueError.

    If the metric name is one of the above, it will return the value of the metric.

    The value of the metric is calculated using the true and predicted labels.

    The value of the metric is returned as a negative number.

    The reason for returning a negative number is that the objective function of the optimizer is to
    minimize the value of the metric.

    If the metric is returned as a positive number, the optimizer will

    :param metric_name: The name of the metric to use
    :param Y_true: the true labels
    :param Y_pred: The predicted labels
    :return: the value of the metric.
    """

    if metric_name == "f1" or metric_name == "f1_macro":
        return -f1_score(Y_true, Y_pred, average="macro")
    if metric_name == "f1_micro":
        return -f1_score(Y_true, Y_pred, average="micro")
    elif metric_name == "acc":
        return -accuracy_score(Y_true, Y_pred)
    elif metric_name == "hamming":
        return hamming_loss(Y_true, Y_pred)
    else:
        raise ValueError(f"Metric {metric_name} not supported")


def calculate_metric_for_generator(
    eval_metric,
    dataset_name,
    classifier_list,
    order,
    random_state,
    train_if_not_exist,
    str_order,
    trained_classifiers,
    i_outer,
    i_inner,
    X_test_inner,
    Y_test_inner,
):
    return calculate_metric(
        eval_metric,
        Y_test_inner,
        make_chain(
            dataset_name,
            classifier_list,
            order,
            random_state,
            train_if_not_exist,
            str_order,
            trained_classifiers,
            i_outer,
            i_inner,
        ).predict(X_test_inner),
    )


# @timeit
def feedback_ga(
    dataset_name="yeast",
    classifier_list=["naive_bayes" for _ in range(14)],
    order=None,
    n_splits=3,
    random_state=42,
    train_if_not_exist=True,
    eval_metric="f1",
):
    """
    It takes a dataset name, a list of classifiers, an order, a number of splits, a random state, a
    boolean to train if not exist, and an evaluation metric.

    It then returns the average of the evaluation metric over the splits.


    :param dataset_name: The name of the dataset to be used, defaults to yeast (optional)
    :param classifier_list: list of classifiers to be used in the chain
    :param order: The order in which the classifiers are chained together
    :param n_splits: number of splits for the outer and inner cross-validation, defaults to 3 (optional)
    :param random_state: The random state used for the outer and inner folds, defaults to 42 (optional)
    :param train_if_not_exist: If the classifier is already trained, then it will not be trained again,
    defaults to True (optional)
    :param eval_metric: the metric to be used for evaluation. Can be "f1", "acc" or "hamming", defaults
    to f1 (optional)
    :return: The average of the metric over all the evaluations.
    """
    X, Y = ds.fetch_openml_dataset(dataset_name)

    metric = 0
    total_number_evaluations = n_splits**2

    assert len(classifier_list) == Y.shape[1]

    order = order if order else list(range(len(classifier_list)))

    outer_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )
    inner_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )

    str_order: str = "_".join([str(x) for x in order])
    trained_classifiers = [None for _ in classifier_list]
    # for cl_name in classifier_list:
    #     assert cl_name in _dict_of_classifiers.keys()

    # temp1 = [calculate_metric(eval_metric, Y_test_inner, make_chain(dataset_name, classifier_list,
    #                           order, random_state, train_if_not_exist, str_order, trained_classifiers, i_outer, i_inner).predict(X_test_inner)) for (i_outer, i_inner),X_test_inner,Y_test_inner in generate_splits_test_inner(X, Y, outer_fold, inner_fold)]

    for i_outer, (train_index, test_index) in enumerate(outer_fold.split(X)):
        X_train, _ = X.values[train_index], X.values[test_index]
        Y_train, _ = Y.values[train_index], Y.values[test_index]
        for i_inner, (train_index, test_index) in enumerate(inner_fold.split(X_train)):
            _, X_test_inner = X_train[train_index], X_train[test_index]
            _, Y_test_inner = Y_train[train_index], Y_train[test_index]
            classifier_chain = make_chain(
                dataset_name,
                classifier_list,
                order,
                random_state,
                train_if_not_exist,
                str_order,
                trained_classifiers,
                i_outer,
                i_inner,
            )

            # prediction = classifier_chain.predict_parallel(X_test_inner,cores=8)
            prediction = classifier_chain.predict(X_test_inner)
            metric += calculate_metric(eval_metric, Y_test_inner, prediction)

    return metric / total_number_evaluations
    # return np.mean(temp1)


def make_chain(
    dataset_name,
    classifier_list,
    order,
    random_state,
    train_if_not_exist,
    str_order,
    trained_classifiers,
    i_outer,
    i_inner,
):
    """
    > It takes a list of classifiers, trains them if they don't exist, and then chains them together

    :param dataset_name: the name of the dataset
    :param classifier_list: a list of classifiers to use in the chain
    :param order: the order of the classifiers in the chain
    :param random_state: the random state used for the classifier
    :param train_if_not_exist: if True, the classifier will be trained if it doesn't exist. If False,
    the classifier will be loaded from the file
    :param str_order: the order of the classifiers in the chain
    :param trained_classifiers: a list of classifiers, each of which is a tuple of (classifier,
    classifier_name, classifier_file_name)
    :param i_outer: the outer fold number
    :param i_inner: the inner fold number
    :return: A classifier chain object.
    """
    for i, cl_name in enumerate(classifier_list):
        if cl_name is None:
            continue
            
        cl_file_name = f"eval/{dataset_name}/{str_order}/pos_{i}/{cl_name}/outer_{i_outer}_inner_{i_inner}_rs_{random_state}.pkl"
        trained_classifiers[i] = get_classifier(
            dataset_name,
            tuple(order),
            random_state,
            train_if_not_exist,
            cl_name,
            i,
            cl_file_name,
        )
    
    
    classifier_chain = cc.chaining_together(
            config_classifier=trained_classifiers, order=order)
        
    return classifier_chain


@lru_cache(maxsize=800)
def get_classifier(
    dataset_name, order, random_state, train_if_not_exist, cl_name, i, cl_file_name
):
    """
    If the classifier file doesn't exist, train it, otherwise load it

    :param dataset_name: the name of the dataset you want to use
    :param order: the order of the classifiers in the chain
    :param random_state: the random state used for the classifier
    :param train_if_not_exist: if True, the classifier will be trained if it doesn't exist. If False, an
    error will be raised
    :param cl_name: the name of the classifier you want to use
    :param i: the index of the classifier in the chain
    :param cl_file_name: the file name of the classifier
    :return: A list of trained classifiers.
    """
    order = list(order)
    if not os.path.exists(cl_file_name):
        if train_if_not_exist:
            train_chain(
                training_classifier_nr=i,
                dataset_name=dataset_name,
                classifier_name=cl_name,
                order=order,
                random_state=random_state,
            )
        else:
            raise FileNotFoundError(cl_file_name)

    with open(cl_file_name, "rb") as f:
        trained_classifiers = pickle.load(f)

    return trained_classifiers


def feedback_greedy(
    dataset_name="yeast",
    classifier="naive_bayes",
    pos=0,
    total_len=14,
    order=None,
    n_splits=3,
    random_state=42,
    train_if_not_exist=True,
    eval_metric="f1",
    noise=0,
):
    """
    It takes a dataset name, a classifier name, a position, a total length, an order, a number of
    splits, a random state, a train if not exist flag and an evaluation metric.

    It then loads the dataset, creates a shuffle split, loads the classifier, trains the classifier if
    it doesn't exist, and then evaluates the classifier.

    The evaluation metric is either f1, acc or hamming loss.

    The function returns the evaluation metric.

    The function is used in the following function:

    :param dataset_name: the name of the dataset to use, defaults to yeast (optional)
    :param classifier: the name of the classifier to use, defaults to naive_bayes (optional)
    :param pos: the position of the classifier in the chain, defaults to 0 (optional)
    :param total_len: the number of labels in the dataset, defaults to 14 (optional)
    :param order: the order of the classifiers in the chain
    :param n_splits: number of splits for the outer and inner cross-validation, defaults to 3 (optional)
    :param random_state: the random seed used for the outer and inner cross-validation, defaults to 42
    (optional)
    :param train_if_not_exist: If the classifier is not trained, train it, defaults to True (optional)
    :param eval_metric: the metric to evaluate the classifier chain on. Can be "f1", "acc" or "hamming",
    defaults to f1 (optional)
    :return: The average loss of the classifier chain.
    """
    X, Y = ds.fetch_openml_dataset(dataset_name)

    metric = 0

    total_number_evaluations = n_splits**2

    assert total_len == Y.shape[1]
    

    order = order if order else list(range(total_len))

    outer_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )
    inner_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )

    str_order: str = "_".join([str(x) for x in order])

    trained_classifier = _dict_of_classifiers[classifier]()

    for i_outer, (train_index, test_index) in enumerate(outer_fold.split(X)):
        X_train, _ = X.values[train_index], X.values[test_index]
        Y_train, _ = Y.values[train_index], Y.values[test_index]
        for i_inner, (train_index, test_index) in enumerate(inner_fold.split(X_train)):
            _, X_test_inner = X_train[train_index], X_train[test_index]
            _, Y_test_inner = Y_train[train_index], Y_train[test_index]
            i = pos
            cl_name = classifier

            cl_file_name = f"eval/{dataset_name}/{str_order}/pos_{i}/{cl_name}/outer_{i_outer}_inner_{i_inner}_rs_{random_state}.pkl"
            if not os.path.exists(cl_file_name):
                if train_if_not_exist:
                    train_chain(
                        training_classifier_nr=i,
                        dataset_name=dataset_name,
                        classifier_name=cl_name,
                        order=order,
                    )
                else:
                    raise FileNotFoundError(cl_file_name)

            with open(cl_file_name, "rb") as f:
                trained_classifier = pickle.load(f)

            classifier_chain_single = cc.chain_configuration_single_fit(
                single_classifier=trained_classifier,
                training_classifier_nr=pos,
                order=order,
            )
            Y_test_inner =make_noisy_lables(Y_test_inner,noise_level=noise)
            pred = classifier_chain_single.predict_single(X_test_inner, Y_test_inner)
            if eval_metric == "f1" or eval_metric == "f1_macro":
                metric += (
                    -f1_score(Y_test_inner[:, pos], pred, average="macro")
                ) / total_number_evaluations
            if eval_metric == "f1_micro":
                metric += (
                    -f1_score(Y_test_inner[:, pos], pred, average="micro")
                ) / total_number_evaluations
            elif eval_metric == "acc":
                # print(accuracy_score(Y_test_inner[:,pos],pred))
                metric += (
                    -accuracy_score(Y_test_inner[:, pos], pred)
                ) / total_number_evaluations
            else:
                metric += (
                    hamming_loss(Y_test_inner[:, pos], pred) / total_number_evaluations
                )

    result = metric
    return result



def make_noisy_lables(Y, noise_level=0.1):
    """
    It takes a vector of labels, and flips a random subset of them

    :param Y: the true labels
    :param noise_level: the probability of a label being flipped
    :return: A copy of the original Y array with a random selection of the values flipped.
    """

    Y_t = Y.copy()

    to_flip_indices = np.random.choice(
        [True, False], size=Y.shape, p=[noise_level, 1 - noise_level]
    )

    Y_t[to_flip_indices] = ~Y_t[to_flip_indices]

    return Y_t


def feedback_greedy2(
    classifier_list,
    dataset_name="yeast",
    order=None,
    n_splits=3,
    random_state=42,
    train_if_not_exist=True,
    eval_metric="f1",
    pos=0,
):
    """
    It takes a dataset name, a list of classifiers, an order, a number of splits, a random state, a
    boolean to train if not exist, and an evaluation metric.

    It then returns the average of the evaluation metric over the splits.


    :param dataset_name: The name of the dataset to be used, defaults to yeast (optional)
    :param classifier_list: list of classifiers to be used in the chain
    :param order: The order in which the classifiers are chained together
    :param n_splits: number of splits for the outer and inner cross-validation, defaults to 3 (optional)
    :param random_state: The random state used for the outer and inner folds, defaults to 42 (optional)
    :param train_if_not_exist: If the classifier is already trained, then it will not be trained again,
    defaults to True (optional)
    :param eval_metric: the metric to be used for evaluation. Can be "f1", "acc" or "hamming", defaults
    to f1 (optional)
    :return: The average of the metric over all the evaluations.
    """
    X, Y = ds.fetch_openml_dataset(dataset_name)

    metric = 0
    total_number_evaluations = n_splits**2


    order = order if order else list(range(len(classifier_list)))

    outer_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )
    inner_fold = ShuffleSplit(
        n_splits=n_splits, test_size=0.30, random_state=random_state
    )

    str_order: str = "_".join([str(x) for x in order])
    trained_classifiers = [None for _ in classifier_list]
    # for cl_name in classifier_list:
    #     assert cl_name in _dict_of_classifiers.keys()

    # temp1 = [calculate_metric(eval_metric, Y_test_inner, make_chain(dataset_name, classifier_list,
    #                           order, random_state, train_if_not_exist, str_order, trained_classifiers, i_outer, i_inner).predict(X_test_inner)) for (i_outer, i_inner),X_test_inner,Y_test_inner in generate_splits_test_inner(X, Y, outer_fold, inner_fold)]

    for i_outer, (train_index, test_index) in enumerate(outer_fold.split(X)):
        X_train, _ = X.values[train_index], X.values[test_index]
        Y_train, _ = Y.values[train_index], Y.values[test_index]
        for i_inner, (train_index, test_index) in enumerate(inner_fold.split(X_train)):
            _, X_test_inner = X_train[train_index], X_train[test_index]
            _, Y_test_inner = Y_train[train_index], Y_train[test_index]
            classifier_chain = make_chain(
                dataset_name,
                classifier_list,
                order,
                random_state,
                train_if_not_exist,
                str_order,
                trained_classifiers,
                i_outer,
                i_inner,
            )

            # prediction = classifier_chain.predict_parallel(X_test_inner,cores=8)
            prediction = classifier_chain.predict_up_to(X_test_inner,pos)
            metric += calculate_metric(eval_metric, Y_test_inner[:,pos], prediction)

    return metric / total_number_evaluations

def train_all_cl_loop(label_nr,dataset_name,order=None):
    for i,classifier_name in enumerate(_dict_of_classifiers.keys()):
        for pos in range(0,label_nr):
            train_chain(training_classifier_nr=pos,dataset_name=dataset_name,classifier_name=classifier_name, order=order)
            
            
def clear_cache_for_dataset_order(dataset_name,order):
    str_order: str = "_".join([str(x) for x in order])
    path_to_file=os.path.join("eval", dataset_name,str_order)
    #remove folder and all its contents
    shutil.rmtree(path_to_file)




if __name__ == "__main__":
    Y = np.random.choice([True, False], size=(5, 5))
    print(Y)
    print("-" * 10)
    print(make_noisy_lables(Y, noise_level=0.1))

    for i,classifier_name in enumerate(_dict_of_classifiers.keys()):
        for pos in range(0,14):
            train_chain(training_classifier_nr=pos,dataset_name="yeast",classifier_name=classifier_name, order=None)

    cl_list2 = ["naive_bayes" for _ in range(14)]
    cl_list2[9]=None
    greed2=feedback_greedy2(dataset_name="yeast", classifier_list=cl_list2,
                                             order=None, n_splits=3, random_state=42, train_if_not_exist=True, eval_metric="f1")
    
    greedy1 = feedback_greedy(
        dataset_name="yeast",
        classifier="naive_bayes",
        pos=0,
        total_len=14,
        order=None,
        n_splits=3,
        random_state=42,
        train_if_not_exist=True,
        eval_metric="acc",
    )
    print("_" * 100)

    cl_list1 = ["naive_bayes" for _ in range(14)]
    cl_list1[0] = "ridge"
    names, scores1 = evaluate_chain(
        dataset_name="yeast", classifier_list=cl_list1, order=None
    )
    feed_ga1 = feedback_ga(dataset_name="yeast", classifier_list=cl_list1, order=None)
    print(feed_ga1)
    print(names)
    for s in scores1[:3]:
        print(s)
