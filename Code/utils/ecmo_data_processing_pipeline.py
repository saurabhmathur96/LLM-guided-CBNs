# TODO: Add data_process for ptt if found

# The data processing pipelines for the different models

"""
How do these models work. We are defining the pipeline with a dictionary
which can define the pipeline for pre-canulation model, post-canulation model or entire range
If we consider the example below, the pre-canulation pipeline has the following steps:-
    aggregation_1, single_value_transformation_2, series_transformation_3, aggregation_4
    
    Each step (key) is defined as <OPERATION>_<STEP NO>. The operations would execute in increasing STEP_NO

We have 3 kinds of operations for the time being
1) Aggregation operation, which aggregate over the dataframe. We wither provide a dict with key value pairs
    indicating the feature name and the aggregation method used
2) Single Value Transformation: Used to indicate that the value being transformed is a single value. Won't work otherwise
3) Series Transformation: Applies the provided transformation to the entire series

NOTES:
1) The transformations are provided as a lambda function
2) When multiple aggregators are provided, we return a dict with the aggregator as the key
   and the computed value as the value
    2a) The dictionary would not contain the delta or relative aggregation. Instead, those aggregations are provided
    separately as (delta|relative)_<feature> key in the feature-aggregated value dictionary returned after computation.
    This is done due to the fact that delta and relative values are considered as separate features, whereas other
    aggregated values are often aggregated together in a single expression. Might change based on the model changes
"""
data_process_for_count_classifier_1 = {
    "pre": {
        "aggregation_1": {
            # We would perform the following aggregation as the first steo
            "heart_rate": "frac",
            "sys_bp": "frac",
            "dia_bp": "frac",
            "map": "frac",
            "spo2": "median",
            "vis_score": "max",
            # lactate requires both a median as well as any value
            "lactate": ["median", "max"],
        },
        "single_value_transformation_2": {
            # We would perform transformation on the already aggregated value
            # For the heart rate sys_bp, dia_bp, map, if the data is already age normalized,
            # we do not need to do any further processing
            "heart_rate": lambda x: x,
            "sys_bp": lambda x: x,
            "dia_bp": lambda x: x,
            "map": lambda x: x,
            # For the spo2, we bin using median O2 saturation less than 88%
            "spo2": lambda x: 1 if x < 88 else 0,
            # For the lactate, we need to check whether the median > 3 or any value > 5
            "lactate": lambda x: 1 if x["median"] > 3 or x["max"] > 5 else 0,
            "vis_score": lambda x: 1 if x > 10 else 0
        },

        "series_transformation_3": {
            # For the acid base group, it is a simple mapping between their flag and
            "ph": lambda x: 1 if x < 7.15 else 0,
            "pco2": lambda x: 1 if x > 70 else 0,
            "base_excess": lambda x: 1 if x < -5 else 0,

            # For the end organ perfusion of the data, we would consider the

            "creatinine": lambda x: 1 if x > 1.2 else 0,
            "ast": lambda x: 1 if x > 250 else 0,
            "alt": lambda x: 1 if x > 250 else 0,
            "total_bilirubin": lambda x: 1 if x > 6 else 0,

            # For the coagulation or blood count
            "platelet": lambda x: 1 if x < 50 else 0,
            "fibrinogen": lambda x: 1 if x < 100 else 0

        },
        # Once the feature transformation is performed, we need to perform a final aggregation for the features not aggregated before
        "aggregation_4": {
            "ph": "frac",
            "pco2": "frac",
            "base_excess": "frac",
            "creatinine": "frac",
            "ast": "frac",
            "alt": "frac",
            "total_bilirubin": "frac",
            "platelet": "frac",
            "fibrinogen": "frac"
        }
    },
    "post": {
        "aggregation_1": {
            # We would perform the following aggregation as the first steo
            "heart_rate": "frac",
            "sys_bp": "frac",
            "dia_bp": "frac",
            "map": "frac",
            "map_non_normalized": ["delta", "relative"],
            "spo2": "median",
            "pco2": ["delta", "relative"],
            "vis_score": "max",
            # lactate requires both a median as well as any value
            "lactate": ["median", "max"],
        },
        # Single value transformation is required if we want to transform already aggregated value
        "single_value_transformation_2": {
            # For the heart rate sys_bp, dia_bp, map, if the data is already age normalized,
            # we do not need to do any further processing
            "heart_rate": lambda x: x,
            "sys_bp": lambda x: x,
            "dia_bp": lambda x: x,
            # For map, we would consider the actual fraction, whether delta map > 50% and relative map > 50%
            "map": lambda x: x,
            "delta_map_non_normalized": lambda x: 1 if x > 0.5 else 0,
            "relative_map_non_normalized": lambda x: 1 if x > 0.5 else 0,
            "vis_score": lambda x: 1 if x > 10 else 0,

            # For pco2, we would consider whether delta pco2 > 50% and relative pco2 > 50%
            "delta_pco2": lambda x: 1 if x > 0.5 else 0,
            "relative_pco2": lambda x: 1 if x > 0.5 else 0,
            # For the spo2, we bin using median O2 saturation less than 88%
            "spo2": lambda x: 1 if x < 88 else 0,
            "lactate": lambda x: 1 if x["median"] > 3 or x["max"] > 5 else 0,
        },

        "series_transformation_3": {

            # For the acid base group, it is a simple mapping between their flag and
            "ph": lambda x: 1 if x < 7.15 else 0,
            "base_excess": lambda x: 1 if x < -5 else 0,
            # For the end organ perfusion of the data, we would consider the
            "creatinine": lambda x: 1 if x > 1.2 else 0,
            "ast": lambda x: 1 if x > 250 else 0,
            "alt": lambda x: 1 if x > 250 else 0,
            "total_bilirubin": lambda x: 1 if x > 6 else 0,
            "platelet": lambda x: 1 if x < 50 else 0,
            "fibrinogen": lambda x: 1 if x < 100 else 0,
            "free_hemoglobin": lambda x: 1 if x > 250 else 0

        },
        "aggregation_3": {
            "ph": "frac",
            "base_excess": "frac",
            "creatinine": "frac",
            "ast": "frac",
            "alt": "frac",
            "total_bilirubin": "frac",
            "platelet": "frac",
            "fibrinogen": "frac",
            "free_hemoglobin": "frac"
        }
    },
    "all": {
        "aggregation_1": {
            # We would perform the following aggregation as the first steo
            "heart_rate": "frac",
            "sys_bp": "frac",
            "dia_bp": "frac",
            "map": "frac",
            "map_non_normalized": ["delta", "relative"],
            "spo2": "median",
            "vis_score": "max",
            "pco2": ["delta", "relative"],
            # lactate requires both a median and max value
            "lactate": ["median", "max"],
        },
        # Single value transformation is required if we want to transform already aggregated value
        "single_value_transformation_2": {
            # For the heart rate sys_bp, dia_bp, map, if the data is already age normalized,
            # we do not need to do any further processing
            "heart_rate": lambda x: x,
            "sys_bp": lambda x: x,
            "dia_bp": lambda x: x,
            # For map, we would consider the actual fraction, whether delta map > 50% and relative map > 50%
            "map": lambda x: x,
            "delta_map_non_normalized": lambda x: 1 if x > 0.5 else 0,
            "relative_map_non_normalized": lambda x: 1 if x > 0.5 else 0,
            # For pco2, we would consider whether delta pco2 > 50% and relative pco2 > 50%
            "delta_pco2": lambda x: 1 if x > 0.5 else 0,
            "relative_pco2": lambda x: 1 if x > 0.5 else 0,
            # For the spo2, we bin using median O2 saturation less than 88%
            "spo2": lambda x: 1 if x < 88 else 0,
            "lactate": lambda x: 1 if x["median"] > 3 or x["max"] > 5 else 0,
            "vis_score": lambda x: 1 if x > 10 else 0
        },
        "series_transformation_3": {

            # For the acid base group, it is a simple mapping between their flag and
            "ph": lambda x: 1 if x < 7.15 else 0,
            "base_excess": lambda x: 1 if x < -5 else 0,
            # For the end organ perfusion of the data, we would consider the
            "creatinine": lambda x: 1 if x > 1.2 else 0,
            "ast": lambda x: 1 if x > 250 else 0,
            "alt": lambda x: 1 if x > 250 else 0,
            "total_bilirubin": lambda x: 1 if x > 6 else 0,
            "pco2": lambda x: 1 if x > 70 else 0,
            "platelet": lambda x: 1 if x < 50 else 0,
            "fibrinogen": lambda x: 1 if x < 100 else 0,
            "free_hemoglobin": lambda x: 1 if x > 250 else 0

        },
        "aggregation_3": {
            "pco2": "frac",
            "ph": "frac",
            "base_excess": "frac",
            "creatinine": "frac",
            "ast": "frac",
            "alt": "frac",
            "total_bilirubin": "frac",
            "platelet": "frac",
            "fibrinogen": "frac",
            "free_hemoglobin": "frac"
        }
    }
}

data_process_for_psb_abstract = {
    "pre": {
        "aggregation_1": {
            # We would perform the following aggregation as the first step
            "map_low": "boolean",
            "vis_score": "max",
            # lactate requires both a median as well as any value
            "lactate": "max",
        },
        "single_value_transformation_2": {
            # We would perform transformation on the already aggregated value
            # For the heart rate sys_bp, dia_bp, map, if the data is already age normalized,
            # we do not need to do any further processing
            "map_low": lambda x: x,
            # For the lactate, we need to check whether the median > 3 or any value > 5
            "lactate": lambda x: 1 if x  > 3 else 0,
            "vis_score": lambda x: 1 if x > 10 else 0
        },
        "series_transformation_3": {
            # For the acid base group, it is a simple mapping between their flag and
            "ph": lambda x: 1 if x < 7.1 else 0,
            "pco2": lambda x: 1 if x > 70 else 0,
            # For the end organ perfusion of the data, we would consider the
            "creatinine": lambda x: 1 if x > 1.3 else 0,
            "ast": lambda x: 1 if x > 150 else 0,
            # For the coagulation or blood count
            "platelet": lambda x: 1 if x < 50 else 0,
        },
        # Once the feature transformation is performed, we need to perform a final aggregation
        # for the features not aggregated before
        "aggregation_4": {
            "ph": "boolean",
            "pco2": "boolean",
            "creatinine": "boolean",
            "ast": "boolean",
            "platelet": "boolean",
        }
    },
    "post": {
        "aggregation_1": {
            # We would perform the following aggregation as the first step
            "map_low": "boolean",
            "map_high": "boolean",
            "vis_score": "max",
            "lactate": "max",
            "pco2": "relative",
        },
        "single_value_transformation_2": {
            # We would perform transformation on the already aggregated value
            # For the heart rate sys_bp, dia_bp, map, if the data is already age normalized,
            # we do not need to do any further processing
            "map_low": lambda x: x,
            "map_high": lambda x: x,
            "relative_pco2": lambda x: 1 if x > 0.3 else 0,
            # For the lactate, we need to check whether the median > 3 or any value > 5
            "lactate": lambda x: 1 if x  > 3 else 0,
            "vis_score": lambda x: 1 if x > 10 else 0
        },

        "series_transformation_3": {
            # For the acid base group, it is a simple mapping between their flag and
            "ph": lambda x: 1 if x < 7.1 else 0,
            # For the end organ perfusion of the data, we would consider the
            # For the coagulation or blood count
            "platelet": lambda x: 1 if x < 50 else 0,
        },
        # Once the feature transformation is performed, we need to perform a final aggregation
        # for the features not aggregated before
        "aggregation_4": {
            "ph": "boolean",
            "platelet": "boolean",
        }
    },
}

data_process_for_psb_abstract_non_boolean = {
    "pre": {
        "aggregation_1": {
            # We would perform the following aggregation as the first step
            "map_low": "boolean",
            "vis_score": "max",
            # lactate requires both a median as well as any value
            "lactate": "max",
            "ph": "min",
            "pco2": "max",
            "creatinine": "max",
            "ast": "max",
            "platelet": "min",
        }
    },
    "post": {
        "aggregation_1": {
            # We would perform the following aggregation as the first step
            "map_low": "boolean",
            "map_high": "boolean",
            "vis_score": "max",
            "lactate": "max",
            "pco2": "relative",
            "ptt": "median",
            "ph": "min",
            "creatinine": "max",
            "ast": "max",
            "platelet": "min",
        }
    },
}