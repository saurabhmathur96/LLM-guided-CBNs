# File which contains the methods for training the classifiers.
import copy
import time
from zipfile import error

import numpy as np

from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)



def train_classifier(train_data, cv_split=5, n_bootstraps = 10,
                     model_config=None, **kwargs):
    """
    The method to train the classifier based on the data provided`
    :param train_data: The training data for the classifier
    :param cv_split: The cross validation split to use for the training
    :param n_bootstraps: The number of bootstrap samples to generate for the train data
    :param model_config: The configuration for the model to train
    :param kwargs: Additional arguments to pass.
    :return: The list of trained models (n_bootstraps) for the classifier
    """
    assert n_bootstraps > 0, "The number of bootstraps should be greater than 0. No bootstraps provided"

    if model_config is None:
        raise ValueError("The model configuration is not provided")

    model = model_config["model"]

    # Once, we have the model, we would check if we have params or param_grid
    if "params" in model_config.keys():
        model.set_params(**model_config["params"])
    elif "param_grid" in model_config.keys():
        param_grid = model_config["param_grid"]
        if kwargs and "grid_search_scoring" in kwargs.keys():
            scoring = model_config["grid_search_scoring"]
        else:
            scoring = "f1"

        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_split, n_jobs=-1, scoring=scoring, verbose=1,
                             error_score="raise")

        # Find the best parameters for the model
        model.fit(train_data["X"], train_data["y"])
        model = model.best_estimator_

    # We would train the model on the training data using the bootstrap samples
    bootstrap_models = []
    print(f"Training the model using {n_bootstraps} bootstrap samples")
    time.sleep(1)
    for _ in tqdm(range(n_bootstraps)):
        if n_bootstraps == 1:
            model.fit(train_data["X"], train_data["y"])
            bootstrap_models.append(copy.deepcopy(model))
            break

        bootstrap_samples_x, bootstrap_samples_y = resample(train_data["X"], train_data["y"], replace=True)
        model.fit(bootstrap_samples_x, bootstrap_samples_y)
        bootstrap_models.append(copy.deepcopy(model))

    return bootstrap_models

def train_score_cross_val_classifier(data, cv, model_config, **kwargs):
    """
    Method for training and scoring the classifier using cross validation
    """
    model = model_config["model"]

    # We currently do not support grid search for cross validation as there are not many data points
    if "params" in model_config.keys():
        model = model.set_params(**model_config["params"])
    elif "param_grid" in model_config.keys():
        raise ValueError("Grid search is not supported for cross validation at the moment")
    else:
        model = model

    # Once we have the model, we would train the cross validate model on the data
    cv_results = cross_validate(model, data["X"], data["y"], cv=cv, scoring=kwargs["scoring"], n_jobs=-1, verbose=1,
                             return_estimator=True, return_train_score=True)

    return cv_results

def evaluate_classifier(models, eval_data, metrics, **kwargs):
    """
    The method to evaluate the classifier based on the data provided
    :param models: The models to evaluate
    :param eval_data: The evaluation data for the classifier
    :param metrics: The metrics to evaluate the model
    :param kwargs: Additional arguments to pass.
    """
    if not isinstance(models, list):
        models = [models]

    results = {m : [] for m in metrics}
    for i, model in tqdm(enumerate(models)):
        y_true = eval_data["y"]
        y_pred = model.predict(eval_data["X"])

        # We would consider the probability of the positive class
        y_pred_proba = model.predict_proba(eval_data["X"])[:, 1]
        # We would evaluate the model based on the evaluation data
        for metric in metrics:
            if metric == "accuracy":
                results[metric].append(accuracy_score(y_true, y_pred))
            elif metric == "precision":
                results[metric].append(precision_score(y_true, y_pred))
            elif metric == "recall":
                results[metric].append(recall_score(y_true, y_pred))
            elif metric == "f1":
                results[metric].append(f1_score(y_true, y_pred))
            elif metric == "roc_auc":
                results[metric].append(roc_auc_score(y_true, y_pred_proba))
            elif metric == "average_precision":
                results[metric].append(average_precision_score(y_true, y_pred_proba))
            else:
                raise ValueError("The metric is not supported for evaluation")

    if "compute_mean_std" in kwargs.keys() and kwargs["compute_mean_std"]:
        results = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in results.items()}
    return results


# We need to write the code to train the generative models. The code is not one to one compatible with sklearn like
# xgboost and sklearn models.
def train_pgm_classifier():
    pass

def evaluate_pgm_classifier():
    pass





