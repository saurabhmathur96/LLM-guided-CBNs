# This is the file for creating the datasets used for PSB abstract
import copy
import pickle
import argparse
import os
import pandas as pd
from pathlib import Path
from utils.ecmo_data_processing_pipeline import data_process_for_psb_abstract
from utils.process_ecmo_data_utils import construct_patient_aggregated_features

tld = Path(__file__).resolve().parents[1]
pelican_path_str = os.path.join(tld, "data", "processed", "ffill_imputed__normalize_age_based_percentile_pelican.pkl")
emr_bp_path = os.path.join(tld, "aime", "emr_bp.pkl")

parser = argparse.ArgumentParser("The argument parser for the PSB abstract data")
parser.add_argument("--pelican_data", default=pelican_path_str, help="path to the pelican data directory")
parser.add_argument("--replace_with_emr", default=False, help="Whether to replace the ARTm data with "
                                                              "the EMR data")
parser.add_argument("--emr_data_path", default=emr_bp_path,
                    help="path to the emr data directory")

args = parser.parse_args()
if __name__ == "__main__":
    # We would use the same first load the dataset which has been created by our method
    with open(args.pelican_data, "rb") as f:
        df_dict = pickle.load(f)


    if args.replace_with_emr:
        # EMR MAP dict
        with open(args.emr_data_path, "rb") as f:
            emr_map_dict = pickle.load(f)

    # We would substitute the data in the original with the data in the map filtered. Additionally, we would clip the
    # data so that we only consider bckt from -24 to 24
    for p_id, p_data in df_dict.items():
        df1 = p_data["data"]

        if args.replace_with_emr:
            p_id_f = "P-" + str(p_id).zfill(3)
            df2 = emr_map_dict[p_id_f]
            m = df1.merge(df2, on='bckt', how='left')
            df1["map"] = m['map_y']

        df_dict[p_id]["data"] = df1[(df1["bckt"] >=-24) & (df1["bckt"] <= 24)]

    # Once the temporal data has been properly constructed, we can construct the relevant boolean dataset
    # One dataset for pre canulation and one dataset for post canulation
    datasets = {"post": pd.DataFrame()}
    for dataset_type in datasets.keys():
        for p_id, p_data in df_dict.items():
            fr = [-24, -1] if dataset_type == "pre" else [0, 24]
            patient_features_dict = construct_patient_aggregated_features(copy.deepcopy(p_data["data"]), fr,
                                                                        data_process_for_psb_abstract[dataset_type],
                                                              time_col="bckt")
            patient_features_dict.update({"injury": p_data["outcome"]})
            patient_features_dict = {k: [v] for k, v in patient_features_dict.items()}
            temp_df = pd.DataFrame(patient_features_dict)
            datasets[dataset_type] = pd.concat([datasets[dataset_type], temp_df], axis=0, ignore_index=True)

    for dataset_type, dataset in datasets.items():
        save_file_name = f"{dataset_type}_pelican.csv" if not args.replace_with_emr \
                        else f"{dataset_type}_pelican_emr.csv"
        dataset.to_csv(save_file_name)
