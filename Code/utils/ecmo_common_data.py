import numpy as np
possible_value_ranges = {
    "ast": [5, np.inf],
    "alt": [5, np.inf],
    "sys_bp": [40, 180],
    "dia_bp": [20, 100],
    "map": [20, 180],
    "heart_rate": [0, 300],
    "pco2": [15, 150],
    "lactate": [0, 30],
    "creatinine": [0, 10],
    "ph": [6.5, 8],
    "spo2": [0, 100],
    "fibrinogen": [20, 1000],
    "total_bilirubin": [0, 30],
    "free_hemoglobin": [0, 4000],
    "base_excess": [-30, 50],
    "vis_score": [0, np.inf],
    "ptt": [30, 250]
}

conversion_factor = {
    "d": {"d": 1, "m": 30, "y": 365},
    "m": {"d": 1/30, "m": 1, "y": 12},
    "y": {"d": 1/365, "m": 1/12, "y": 1}
}

# TODO: Confirm what is the mapping of features platelet and ptt
old_to_new_feature_mapping = {
    "bckt":             {"features": ["time"],                    "mapping": "identity"},
        "heart_rate":       {"features": ["HR (bpm)"],                "mapping": "identity"},
        "map":              {"features": ["ARTm (mmHg)"],             "mapping": "identity"},
        "sys_bp":           {"features": ["ARTs (mmHg)"],             "mapping": "identity"},
        "dia_bp":           {"features": ["ARTd (mmHg)"],             "mapping": "identity"},
        "spo2":             {"features": ["SpO2 (%)"],                "mapping": "identity"},
        "creatinine":       {"features": ["CREATININE, SERUM"],       "mapping": "identity"},
        "fibrinogen":       {"features": ["FIBRINOGEN LEVEL"],        "mapping": "identity"},
        "alt":              {"features": ["ALT (SGPT)"],              "mapping": "identity"},
        "ast":              {"features": ["AST (SGOT)"],              "mapping": "identity"},
        "platelet":         {"features": ["PLATELET COUNT"],          "mapping": "identity"},
        "total_bilirubin":  {"features": ["BILIRUBIN, TOTAL"],        "mapping": "identity"},
        "free_hemoglobin":  {"features": ["FREE HEMOGLOBIN PLASMA"],  "mapping": "identity"},
        "ptt":              {"features": ["PARTIAL THROMBOPLASTIN TIME"], "mapping": "identity"},

    "ph":               {"features": ["PH ARTERIAL, POCT", "PH ARTERIAL"],
                         "mapping": "min"},
    "pco2":             {"features": ["PCO2 ARTERIAL, POCT", "PCO2 ARTERIAL"],
                         "mapping": "max"},
    "lactate":          {"features": ["LACTATE BLOOD, POCT", "LACTATE", "LACTATE, WHOLE BLOOD"],
                         "mapping": "max"},
    "base_excess":      {"features": ["BASE EXCESS ARTERIAL, POCT", "BASE EXCESS ARTERIAL"],
                         "mapping": "max"},
    "vis_score":        {"features": ["vis_score"], "mapping": "identity"}
}

# TODO: Add the new data required for removing the noisy artifacts from the data
possible_value_ranges_pelican_ver = {
    "ARTm (mmHg)": [20, 180],
    "ARTs (mmHg)": [40, 180],
    "ARTd (mmHg)": [20, 100],
}

predicate_parameterization = {
    "declarative_predicates": {
        "map": ["patient_id", "map", "time"],
        "sys_bp": ["patient_id", "sys_bp", "time"],
        "dia_bp": ["patient_id", "dia_bp", "time"],
        "heart_rate": ["patient_id", "heart_rate", "time"],
        "spo2": ["patient_id", "spo2", "time"],
        "creatinine": ["patient_id", "creatinine", "time"],
        "fibrinogen": ["patient_id", "fibrinogen", "time"],
        "alt": ["patient_id", "alt", "time"],
        "ast": ["patient_id", "ast", "time"],
        "platelet": ["patient_id", "platelet", "time"],
        "total_bilirubin": ["patient_id", "total_bilirubin", "time"],
        "free_hemoglobin": ["patient_id", "free_hemoglobin", "time"],
        "ph": ["patient_id", "ph", "time"],
        "pco2": ["patient_id", "pco2", "time"],
        "lactate": ["patient_id", "lactate", "time"],
        "base_excess": ["patient_id", "base_excess", "time"],
        "vis_score": ["patient_id", "vis_score", "time"],
        "neurological_injury": ["patient_id"],
    },
    "computed_predicates": {

    }
}

rate_of_change_filtering_pelican_ver = {
    "ARTm (mmHg)": [10, 300],
    "ARTs (mmHg)": [10, 300],
    "ARTd (mmHg)": [10, 300]
}