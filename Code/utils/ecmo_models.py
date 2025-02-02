from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from pgmpy.models.BayesianNetwork import BayesianNetwork

discriminative_count_classifier_models_cv = {
"RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": 50,
            "max_depth": 3,
            "bootstrap": False,
            "class_weight": 'balanced'
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {
            "C": 1,
            "penalty": 'l2',
            "solver": 'liblinear',
            "class_weight": 'balanced'
        }
    },
    "SVC": {
        "model": SVC(),
        "params": {
            "C": 2,
            "kernel": 'rbf',
            "gamma": 'scale',
            "class_weight": 'balanced',
            "probability": True
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": 4,
            "min_samples_split": 5,
            "class_weight": 'balanced'
        }
    },
    "XGBoost": {
        "model": XGBClassifier(),
        "params": {
            "eta": 0.3,
            "max_depth": 2,
            "gamma": 1,
            "lambda": 1,
            "alpha": 1,
            # Update it to the actual data distribution
            "scale_pos_weight": 1.13,
            "n_estimators": 20,
            "objective": "binary:logistic"
        }
}}
discriminative_count_classifier_models_bootstrap = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "param_grid": {
            "n_estimators": [40],
            "max_depth": [3],
            "bootstrap": [True],
            "class_weight": ['balanced']
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
        "param_grid": {
            "C": [1],
            "penalty": ['l2'],
            "solver": ['liblinear'],
            "class_weight": ['balanced']
        }
    },
    "SVM": {
        "model": SVC(),
        "param_grid": {
            "C": [1],
            "kernel": ['rbf'],
            "gamma": ['auto'],
            "class_weight": ['balanced'],
            "probability": [True]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "param_grid": {
            "max_depth": [4],
            "min_samples_split": [10],
            "class_weight": ['balanced']
        }
    },
    # TODO: Correct the Xgboost parameter grid
    "XGBoost": {
        "model": XGBClassifier(),
        "param_grid": {
            "eta": [0.3],
            "max_depth": [2],
            "gamma": [1],
            "lambda": [1],
            "alpha": [1],
            # Update it to the actual data distribution
            "scale_pos_weight": [1.13],
            "n_estimators": [20],
            "objective": ["binary:logistic"]
        }
    }}

# Generative models for the count based classifiers
generative_count_classifier_models = {
    "MultinomialNaiveBayes": {
        "model": MultinomialNB(),
        "param_grid": {
            "alpha": [0.5],
            "fit_prior": [True, False]
        }
    },
    "BayesianNetwork_Pre": {
        "model": BayesianNetwork(),
        "edges": [("vis_score_pre", "map_pre"), ("pco2_pre", "ph_pre"),
                  ("ph_pre", "injury"), ("platelet_pre", "injury"),
                  ("map_pre", "_combined_ast_creatinine_lactate_pre"),
                  ("_combined_ast_creatinine_lactate_pre", "injury")],
        "params": [{"estimator": "MaximumLikelihoodEstimator"},
                   {"estimator": "BayesianEstimator", "prior_type": "K2"},
                   {"estimator": "BayesianEstimator", "prior_type": "BDeu", "equivalent_sample_size": 10}]},
    "BayesianNetwork_Post": {
        "model": BayesianNetwork(),
        "edges": [("relative_map_post", "ph_post"), ("ph_post", "injury"),
                  ("vis_score_post", "map_post"), ("map_post", "_combined_ast_creatinine_lactate_post"),
                  ("_combined_ast_creatinine_lactate_post", "injury"), ("_combined_platelet_ptt_post", "injury"),
                  ("map_post", "injury")],
        "params": [{"estimator": "MaximumLikelihoodEstimator"},
                   {"estimator": "BayesianEstimator", "prior_type": "K2"},
                   {"estimator": "BayesianEstimator", "prior_type": "BDeu", "equivalent_sample_size": 10}]}}