import copy
import os
import argparse
import pickle
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from aime.handle_emr import emr_data_loc
from utils.ecmo_common_data import possible_value_ranges, possible_value_ranges_pelican_ver, \
    rate_of_change_filtering_pelican_ver
from utils.process_ecmo_data_utils import ecmo_clear_abnormal_values, ecmo_feature_mapping_pelican_to_old_hourly, \
    ecmo_impute_missing_values, ecmo_normalize_data, subsample_temporal_feature, remove_noisy_artifacts

parser = argparse.ArgumentParser("Please provide the arguments for the data")

# Define the arguments
parser.add_argument("--load_raw_data_pickle", type=bool, default=False,
                    help="Load raw data from pickle instead of loading the data directly")
parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(__file__), "data"),
                    help="Add the location to access the data directory")
parser.add_argument("--patient_data_folder_name", type=str, default="pelican",
                    help="Name of the folder storing the patient hourly data")
parser.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(__file__),
                                                                  "data", "processed"),
                    help="Path to folder for saving the processed data")
parser.add_argument('--raw_data_file_name', type=str, default='pelican_hourly_data_dict.pkl',
                    help='Name of the raw data file (default: pelican_hourly_data_dict.pkl)')
parser.add_argument("--remove_missing_level", type=int, default=2,
                    help="Level of missing data removal, \
                    {0: No removal, 1: Remove in case of missing label, 2: Remove in case of missing any patient file}")
parser.add_argument("--injury_cut_off", type=float, default=8.0,
                    help="The cutoff to indicate neurological injury")
parser.add_argument('--imputed_data_file_name', type=str, default='pelican_hourly_data_imputed.pkl',
                    help='Name of the imputed data file (default: pelican_hourly_data_imputed.pkl)')
parser.add_argument('--impute_method', type=str, default="ffill",
                    help='Method to impute data (default: ffill_bfill)')
parser.add_argument('--normalize_mode', type=str, default="age_based_percentile",
                    help='Method to normalize data (default: "age_based")')
parser.add_argument('--normalize_data_file_name', type=str, default="10_90_percentiles_age.csv",
                    help='Name of the file containing data required for normalization')
parser.add_argument('--vis_file_name', type=str, default="p_vis_scores.csv",
                    help='Name of the file containing data required for normalization')
parser.add_argument('--patient_age_file_path', type=str, default=os.path.join(os.path.dirname(__file__),
                                                                              "data", "pelican",
                                                                              "pelican_patient_age.csv"),
                    help='Path to the file containing patient age')
parser.add_argument('--remove_noisy_artifacts', type=bool, default=True,
                    help='Path to the file containing patient age')
# parser.add_argument("--use_emr_bp_data", default=os.path.join("aime", "emr_bp.pkl"),
#                     help="Use the EMR data for replacing the MAP values")
parser.add_argument("--use_emr_bp_data", default=None,
                    help="Use the EMR data for replacing the MAP values")
args = parser.parse_args()


def harmonize_old_hourly_pelican(df: pd.DataFrame, subsampling_strategy: str | None = None, **kwargs):
    # 1)  Map the column names from the pelican data to match the one in the new data
    updated_columns_df = ecmo_feature_mapping_pelican_to_old_hourly(df)

    # 2) Clear the abnormal values from the data
    updated_columns_df = ecmo_clear_abnormal_values(updated_columns_df, possible_value_ranges)

    # 3) If subsampling strategy is provided, we would subsample the data
    if subsampling_strategy:
        assert subsampling_strategy in ["mean", "max", "min", "first", "last"], "Not tested subsampling strategy"
        # To harmonize the old data, we would first need to match the columns with the new data
        # Once the columns have been matched, we would need to subsample the data to hourly data

        # 1) We need to get the minium and maximum hour in the data
        updated_columns_df = subsample_temporal_feature(updated_columns_df, 'hour', 'hour', "bckt",
                                                        subsampling_strategy)

    # 4) One additional thing we need to do is make sure that the time series is contiguous. If there is any missing
    # bckt value, we would insert an empty row with None values
    bckt_range = range(updated_columns_df["bckt"].min(), updated_columns_df["bckt"].max() + 1)

    # Get the missing values for the temporal column and create a temporary dataframe with same columns as
    # updated_columns_df and missing_values for bckt column
    missing_values = set(bckt_range) - set(updated_columns_df["bckt"])
    temp_df = pd.DataFrame({'bckt': list(missing_values)})
    # Insert empty rows in the temporary dataframe
    for col in updated_columns_df.columns:
        if col != "bckt":
            temp_df[col] = np.nan

    updated_columns_df = pd.concat([updated_columns_df, temp_df]).sort_values(by=['bckt'])

    if kwargs and "emr_data" in kwargs.keys():
        # We would replace the map values in the data with the EMR values for the rows where the bckt value is the same
        updated_columns_df = updated_columns_df.merge(kwargs["emr_data"], on="bckt", how="left")
        updated_columns_df["map"] = updated_columns_df["map_y"]
        updated_columns_df = updated_columns_df.drop(columns=["map_x", "map_y"])
    return updated_columns_df


def process_t3_epic_flowsheets(df: pd.DataFrame, cols: dict | None = None, **kwargs):
    """
    :param df: The dataframe to process
    :param cols: The column names to use
    :param kwargs: Additional arguments required by the method
    """
    if cols is None:
        cols = ["time", "feature", "value"]
        df.columns = cols

    elif isinstance(cols, dict):
        df = df.rename(columns=cols)
        df = df[list(cols.values())]
        assert "time" in df.columns and "feature" in df.columns and "value" in df.columns

    else:
        raise ValueError("Incorrect cols value provided. Accepts only a dict with mapping or None")

    # We would remove any rows which have multiple values for the same time and would only keep the first one
    df = df.drop_duplicates(subset=["time", "feature"], keep="first")
    df = df.pivot(index="time", columns="feature", values="value")
    df = df.reset_index()

    # We would convert the time column to type int
    df["time"] = df["time"].astype(int)

    # If we need to remove the noisy artifacts from the data, remove them
    if "remove_noisy_artifacts" in kwargs.keys() and kwargs["remove_noisy_artifacts"]:
        noisy_params = kwargs["noisy_artifacts_params"]
        df = remove_noisy_artifacts(df, **noisy_params)

    # Sort the dataframe based on the index
    df = df.sort_values(by=["time"])

    # if temporal_subsampling in kwargs and subsampling strategy is provided, we would subsample the data
    if "temporal_subsampling" in kwargs.keys():
        if kwargs["temporal_subsampling"] is not None:
            df = subsample_temporal_feature(df, kwargs["resolution_given"], kwargs["resolution_to"], "time",
                                            kwargs["temporal_subsampling"])

    # Sort the dataframe based on the index
    df = df.sort_values(by=["time"])

    return df


def filter_missing_pelican_data(pt_data, outcomes, filter_level, **kwargs):
    """
    :param pt_data: The data to filter
    :param outcomes: The outcomes to filter
    :param filter_level: The level of missing data to filter
    :param kwargs: Additional arguments required by the method
    """
    # Once we have parsed through the directory, we would check if any patient is missing one of the files
    # Check is done only in case when missing level is 2. We would remove such missing patients from the data

    patients_list = list(pt_data.keys())
    if filter_level >= 2:
        patients_to_remove = []
        for pt_index in patients_list:
            # We would perform a mapping
            all_data_present = all(isinstance(pt_data[pt_index][data_type], pd.DataFrame)
                                   for data_type in patient_data[pt_index].keys())

            if not all_data_present:
                patients_to_remove.append(pt_index)
                pt_data.pop(pt_index, None)
                outcomes.pop(pt_index, None)

        print(f"Removed Patients {patients_to_remove} due to missing one or "
              "more of the three (t3/epic/flowsheets) file")

        patients_list = list(pt_data.keys())

    if filter_level >= 1:
        patients_to_remove = []
        # We would perform filtering based on the outcomes. If a patient is missing the outcome, we would remove it
        for pt_index in patients_list:
            if pt_index not in outcomes.keys():
                patients_to_remove.append(pt_index)
                pt_data.pop(pt_index, None)

        print(f"Removing patients {patients_to_remove} due to missing the outcome")
        patients_list = list(pt_data.keys())

    if filter_level >= 0:
        patients_to_remove = []
        # We need to remove patients which have no data either for pre-or-post-canulation
        if kwargs and "time_col" in kwargs.keys():
            time_col = kwargs["time_col"]
        else:
            time_col = "time"
        for pt_index in patients_list:
            # If there is no data for the patient pre-canulation, we would remove that patient
            if kwargs and "filter_on_datatypes" in kwargs.keys():
                data_types_to_filter = kwargs["filter_on_datatypes"]
            else:
                data_types_to_filter = pt_data[pt_index].keys()

            # Removing data with missing pre or post canulation data
            if kwargs and "filter_temporal" in kwargs.keys():
                if kwargs["filter_temporal"] == "pre" or kwargs["filter_temporal"] == "both":
                    if all([pt_data[pt_index][data_type][time_col].min() >= 0 for data_type in data_types_to_filter]):
                        patients_to_remove.append(pt_index)

                # If there is no data for the patient post-canulation, we would remove the patient
                if kwargs["filter_temporal"] == "post" or kwargs["filter_temporal"] == "both":
                    if all([pt_data[pt_index][data_type][time_col].max() < 0 for data_type in data_types_to_filter]):
                        patients_to_remove.append(pt_index)

            # Once we have all the patients in the patient_to_remove list, we would remove them from the data
            for patient in patients_to_remove:
                pt_data.pop(patient, None)
                outcomes.pop(patient, None)

        print(f"Removed Patients {patients_to_remove} due to having either no pre or post canulation data in "
              f"{data_types_to_filter} files")

    return pt_data, outcomes


def process_pelican_imaging(data_path, s_path=None, cutoff_score=10000, val_to_select="last"):
    scores = pd.read_csv(data_path)

    # Map the subject id to integers
    scores["SUBJECTID"] = scores["SUBJECTID"].transform(lambda x: int(re.sub(r"^0+", '',
                                                                             x.split("_")[0].split("-")[1])))
    scores["TIMEOFIMAGINGINSEC"] = scores["TIMEOFIMAGINGINSEC"].astype(int)

    scores["NISTOTAL"] = scores["NISTOTAL"].astype(float)
    # We would drop any rows without a NISTOTAL score
    scores = scores.dropna(subset=["NISTOTAL"])

    # Once we have the all this data, we would get the value based on the method
    if val_to_select == "last" or val_to_select == "first":
        # We would sort the scores by SUBJECTID as well as TIMEOFIMAGINGINSEC
        scores = scores.sort_values(by=["SUBJECTID", "TIMEOFIMAGINGINSEC"])
        # For patients with multiple scores, we would only keep the latest score
        scores = scores.drop_duplicates(subset=["SUBJECTID"], keep=val_to_select)
        scores = scores[["SUBJECTID", "NISTOTAL"]]

    elif val_to_select == "average":
        # Select the relevant subset of features
        scores = scores[["SUBJECTID", "NISTOTAL"]]
        scores = scores.groupby("SUBJECTID")["NISTOTAL"].mean()

    else:
        raise NotImplementedError("Any aggregation except first, last or mean is not implemented")

    scores_dict = scores.to_dict()
    s_dict = {}
    for index, patient_key in scores_dict["SUBJECTID"].items():
        s_dict[patient_key] = 1 if scores_dict["NISTOTAL"][index] >= cutoff_score else 0

    if s_path is not None:
        with open(os.path.join(s_path, "outcomes.pkl"), "rb") as fi_outcomes:
            pickle.dump(s_dict, fi_outcomes)

    return s_dict


def process_pelican_vis(data_dict: dict, vis_file_path: str, update_pickle_path: str = "",
                        remove_missing_vis=False):
    """
    :param vis_file_path: The file path to file containing the vis information for the patients
    :param data_dict: The data dict we can update. If None is provided, a dataframe containing
    the vis scores indexed by the subject id is provided
    :param update_pickle_path: The parameter to update the data pickle if provided
    :param remove_missing_vis: Whether to remove records with missing vis values or not
    """
    c_mapping = {
        "STUDY_ID": "patient_id",
        "TS": "bckt",
        "DOPA_VAL": "vis_dop",
        "EPI_VAL": "vis_epi",
        "MILR_VAL": "vis_mil",
        "VASO_VAL": "vis_vaso",
        "NOEPI_VAL": "vis_norepi",
        "VIS": "vis_score"
    }
    vis_df = pd.read_csv(vis_file_path)
    vis_df = vis_df.rename(columns=c_mapping)

    # We would process the patient id in the form of "P-<PATIENT_ID>
    vis_df["patient_id"] = vis_df["patient_id"].transform(lambda x: x.split("-")[1])

    # We would convert the 3 digit patient id into an integer
    vis_df["patient_id"] = vis_df["patient_id"].transform(lambda x: int(re.sub(r"^0+", '', x)))

    unique_patient_ids = list(vis_df["patient_id"].unique())
    vis_data_dict = {k: None for k in unique_patient_ids}

    # Once we have the unique ids, we would iterate over the different patient
    for pt_id in unique_patient_ids:
        # We would get the dataframe containing only the value for the patient
        patient_vis_df = copy.deepcopy(vis_df[vis_df["patient_id"] == pt_id])[list(c_mapping.values())]
        # We would convert tall the data into float and remove the patient id from the data
        patient_vis_df = patient_vis_df.astype(float).drop(labels="patient_id", axis=1)

        # We would process the data by subsampling from the vis dataframe
        patient_vis_df = subsample_temporal_feature(patient_vis_df, temporal_feature_name="bckt")
        vis_data_dict[pt_id] = patient_vis_df

    # Once we have the vis dict for the different patients, we would merge them with the patient data dict, using the
    # bckt value as the join value for the different patients
    vis_col_list = [c for c in c_mapping.values() if c != "bckt" and c != "patient_id"]
    patients_with_missing_vis = []
    for pt_id, patient_data_dict in data_dict.items():
        if pt_id in vis_data_dict.keys():
            # We need to remove any columns from the vis column list
            for col in vis_col_list:
                if col in data_dict[pt_id]["data"].columns:
                    data_dict[pt_id]["data"] = data_dict[pt_id]["data"].drop(labels=col, axis=1)

            data_dict[pt_id]["data"] = data_dict[pt_id]["data"].merge(vis_data_dict[pt_id], how="outer", on="bckt")
        else:
            # We would add additional columns to the dataframe which are not bckt or patient id
            # We would add additional columns with None values to the list
            data_dict[pt_id]["data"][vis_col_list] = None
            patients_with_missing_vis.append(pt_id)

    if patients_with_missing_vis:
        print("Patients with missing vis values", patients_with_missing_vis)
    if remove_missing_vis:
        for pt_id in patients_with_missing_vis:
            del vis_data_dict[pt_id]

    if update_pickle_path:
        with open(update_pickle_path, 'wb') as fi_vis:
            pickle.dump(data_dict, fi_vis)

    return data_dict


def get_data_dict(data_path: str, s_path=None):
    """
    """
    column_mapping = {
        "t3": None,
        "epic": {"MEASURE_TIME": "time", "RECORD_NAME": "feature", "MEASURE_VALUE": "value"},
        "flowsheets": {"MEASURE_TIME": "time", "RECORD_NAME": "feature", "MEASURE_VALUE": "value"}
    }
    print(f"Getting the data from {data_path}")
    if args.remove_noisy_artifacts:
        print("\tNoisy artifacts removal is enabled\n")
    for p_file in tqdm(os.listdir(data_path)):
        patient_id = int(re.sub(r"^0+", '', p_file.split("_")[0].split("-")[1]))
        data_type = p_file.split("_")[1].replace(".csv", "")
        if patient_id not in pelican_data.keys():
            pelican_data[patient_id] = {"t3": None, "epic": None, "flowsheets": None}
        pelican_data[patient_id][data_type] = pd.read_csv(os.path.join(data_path, p_file))
        noisy_artifacts_params = None
        if args.remove_noisy_artifacts:
            noisy_artifacts_params = {
                "threshold": possible_value_ranges_pelican_ver,
                "rate_of_change": rate_of_change_filtering_pelican_ver,
                "temporal_feature_name": "time",
                "imputation": args.impute_method
            }

        pelican_data[patient_id][data_type] = process_t3_epic_flowsheets(df=pelican_data[patient_id][data_type],
                                                                         cols=column_mapping[data_type],
                                                                         temporal_subsampling="mean",
                                                                         resolution_given="sec", resolution_to="hour",
                                                                         remove_noisy_artifacts=args.remove_noisy_artifacts,
                                                                         noisy_artifacts_params=noisy_artifacts_params)

    if s_path is not None:
        os.makedirs(s_path, exist_ok=True)
        with open(os.path.join(s_path, "pelican_raw_data_dict.pkl"), "wb") as fi_1:
            pickle.dump(pelican_data, fi_1)

    # Once the data has been loaded, we would do load the
    return pelican_data


if __name__ == "__main__":
    pelican_data = {}
    normal_save_name_prefix = "" if not args.impute_method else f"{args.impute_method}_imputed_"

    # We would read the pelican data. We need to make sure that each patient has the t3, flowshhets
    # as well as the epic labs data available
    pelican_data_loc = str(os.path.join(args.data_path, args.patient_data_folder_name))

    # Process the pelican imaging scores
    print(f"Processing the pelican imaging scores")
    pelican_outcomes = process_pelican_imaging(data_path=os.path.join(pelican_data_loc, "pelican_imaging.csv"),
                                               cutoff_score=args.injury_cut_off, val_to_select="last")

    # Get the raw patient data dictionary containing the t3, epic as well as the flowsheets data for each patient
    print(f"Getting the raw patient data")
    if not args.load_raw_data_pickle:
        patient_data = get_data_dict(data_path=os.path.join(pelican_data_loc, "patient_data"),
                                     s_path=os.path.join(os.path.dirname(__file__), "temp"))
    else:
        with open(os.path.join(os.getcwd(), "temp", "pelican_raw_data_dict.pkl"), "rb") as f:
            patient_data = pickle.load(f)

    # Filtering out any missing values
    print(f"Filtering out the missing values")
    patient_data, pelican_outcomes = filter_missing_pelican_data(pt_data=patient_data, outcomes=pelican_outcomes,
                                                                 filter_level=args.remove_missing_level,
                                                                 filter_on_datatypes=["t3"],
                                                                 filter_temporal="both")

    # Once, we have the filtered data, we would combine the three dataframes in patient_data into one dataframe joining
    # on the time column
    emr_data = None
    if args.use_emr_bp_data:
        print("We would be replacing the MAP values with the ones from EMR")
        # We would load the emr data for the different patients
        with open(args.use_emr_bp_data, "rb") as f:
            emr_data = pickle.load(f)

    print(f"Combining the three dataframes into a single dataframe. \n"
          f"Noisy Artifacts Removal argument {args.remove_noisy_artifacts}")
    time.sleep(1)
    for p_id, p_data in tqdm(patient_data.items()):
        t3, epic, flowsheets = p_data["t3"], p_data["epic"], p_data["flowsheets"]
        # We would merge the three dataframes into a single dataframe
        patient_data[p_id] = {"outcome": pelican_outcomes[p_id], "chd": 0}
        patient_data[p_id]["data"] = t3.merge(epic, on="time", how="outer").merge(flowsheets, on="time", how="outer")
        # Once we have merged the data, we would need to sort the data on the time column again
        patient_data[p_id]["data"] = patient_data[p_id]["data"].sort_values(by=["time"])
        # Removing the noise from the data if the corresponding argument is passed
        if args.remove_noisy_artifacts:
            patient_data[p_id]["data"] = remove_noisy_artifacts(patient_data[p_id]["data"],
                                                                threshold=possible_value_ranges_pelican_ver,
                                                                rate_of_change=rate_of_change_filtering_pelican_ver,
                                                                temporal_feature_name="time")

    # We would harmoize the data to have the same columns for all the patients
    print("Harmonizing the data to match the old hourly data")
    time.sleep(1)
    for p_id, p_data in tqdm(patient_data.items()):
        if emr_data is not None:
            patient_data[p_id]["data"] = harmonize_old_hourly_pelican(df=p_data["data"],
                                                                  emr_data=emr_data["P-" + str(p_id).zfill(3)])
        else:
            patient_data[p_id]["data"] = harmonize_old_hourly_pelican(df=p_data["data"])

    # We would save the data as a pickle file
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, args.raw_data_file_name), "wb") as f:
        pickle.dump(patient_data, f)

    # In case, we need to process the vis data, we would also consider the vis data
    if args.vis_file_name:
        print("We need to add the vis information for the patients as well")
        vis_path = str(os.path.join(args.data_path, args.patient_data_folder_name, args.vis_file_name))
        u_pickle_path = str(os.path.join(args.save_path, args.raw_data_file_name))
        patient_data = process_pelican_vis(patient_data, vis_path, u_pickle_path)

    # Once we have saved the data, we would also check if we need to impute the missing values
    if args.impute_method is not None:
        print(f"Imputing the missing values using the {args.impute_method} method")
        time.sleep(1)
        for p_id, p_data in tqdm(patient_data.items()):
            patient_data[p_id]["data"] = ecmo_impute_missing_values(p_data["data"],
                                                                    imputation_method=args.impute_method)

        print(f"Saving the imputed data to {os.path.join(args.save_path, args.imputed_data_file_name)}")
        with open(os.path.join(args.save_path, args.imputed_data_file_name), "wb") as f:
            pickle.dump(patient_data, f)

    # If we need to normalize the data, we would do that as well
    if args.normalize_mode is not None:
        print(f"Normalizing the data using the {args.normalize_mode} method")
        # We would get the patient ages
        patient_ages_df = pd.read_csv(args.patient_age_file_path)
        patient_ages_df["Patient ID"] = patient_ages_df["Patient ID"].astype(int)

        # We would load the normalization data file
        normalize_mode = args.normalize_mode
        normalize_desc_df = pd.read_csv(str(os.path.join(args.data_path, args.normalize_data_file_name)))
        print(f"Saving the imputed and possibly normalized data to "
              f"{os.path.join(args.save_path, f'{normal_save_name_prefix}_normalize_{normalize_mode}_pelican')}.pkl")

        print("Normalizing the data for the different patients")
        time.sleep(1)
        for p_id, p_data in tqdm(patient_data.items()):
            patient_data[p_id]["data"] = ecmo_normalize_data(data=p_data["data"], p_id=p_id,
                                                             normalize_method=normalize_mode,
                                                             patient_ages=copy.deepcopy(patient_ages_df),
                                                             normalize_desc=copy.deepcopy(normalize_desc_df))

        # Once we have normalized the data, we would save the data to a pickle file
        full_save_path = os.path.join(args.save_path, f'{normal_save_name_prefix}_normalize_{normalize_mode}')
        if args.use_emr_bp_data:
            full_save_path += "_emr_bp"

        full_save_path += "_pelican.pkl"
        with open(full_save_path, "wb") as fi:
            pickle.dump(patient_data, fi)

    print("Data processing complete!")




