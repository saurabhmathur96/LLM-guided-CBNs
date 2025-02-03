import copy
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.utils.fixes import percentile

from utils.ecmo_common_data import conversion_factor, old_to_new_feature_mapping

def ecmo_clear_abnormal_values(data: pd.DataFrame, possible_ranges: dict, replacement_val: dict | None=None):
    """
    :param data: The dataframe we want to sanitize
    :param possible_ranges: Dictionary containing the columns as well as their normal ranges
    :param replacement_val: The replacement values for such incorrect data points
    """
    for feature, possible_range in possible_ranges.items():
        # We would check whether each item in the feature series falls inside the
        if feature in data.columns:
            if isinstance(replacement_val, dict):
                data[feature] = data[feature].apply(lambda x: replacement_val[feature]
                                                              if x < possible_range[0] or x > possible_range[1] else x)
            elif replacement_val is None:
                data[feature] = data[feature].apply(lambda x: None
                                                              if x < possible_range[0] or x > possible_range[1] else x)
            else:
                raise TypeError("Incorrect Data Type Provided. Only accepts dicts or None")
    return data

def ecmo_impute_missing_values(data: pd.DataFrame, imputation_method:str ="ffill", time_col_name:str = "bckt"):
    """
    This method is used for imputing the missing values in the dataframe based on the method provided
    :param data: The data frame to impute
    :param imputation_method: The imputation method to impute
    :param time_col_name: The name of the time column which would be used for splitting imputation
    :returns: The imputed data frame
    """
    if imputation_method == "ffill_bfill":
        # We need to perform imputation separately for data before time 0 and after time 0        
        pre_canulation_data = data[data[time_col_name] < 0].ffill(axis=0).bfill(axis=0)
        post_canulation_data = data[data[time_col_name] >=0].ffill(axis=0).bfill(axis=0)
    
    elif imputation_method == "ffill":
        pre_canulation_data = data[data[time_col_name] < 0].ffill(axis=0)
        post_canulation_data = data[data[time_col_name] >=0].ffill(axis=0)
    
    elif imputation_method == "interpolation_both" or imputation_method == "interpolation_forward":
        direction = imputation_method.split("_")[1]
        pre_canulation_data = data[data[time_col_name] < 0].interpolate(method="linear", limit_direction=direction)
        post_canulation_data = data[data[time_col_name] >= 0].interpolate(method="linear", limit_direction=direction)
    
    else:
        raise NotImplementedError("The imputation method is not implemented or supported")
    
    # We would concatenate the two dataframes together
    data = pd.concat([pre_canulation_data, post_canulation_data], axis=0)
    
    return data

def ecmo_process_temporal_value(temporal_val: str, val_type: str, conv_factor: dict):
    """
    :param temporal_val: The patient age
    :param val_type: String indicating whether the value provided is a single value or a ranged value
    :param conv_factor: The conversion factor used for the problem
    """
    # We would match the regular expression to parse the value as well as the unit (d/m/y)
    if val_type == "single":
        match = re.match(r'(\d+)(\w+)', temporal_val)
        if match is not None:
            temporal_val = int(match.group(1))
            if match.group(2).lower().startswith("d"):
                temporal_val = temporal_val * conv_factor["d"]
            elif match.group(2).lower().startswith("m"):
                temporal_val = temporal_val * conv_factor["m"]
            elif match.group(2).lower().startswith("y"):
                temporal_val = temporal_val * conv_factor["y"]
            else:
                raise ValueError("Incorrect date! Should be like <num><d*|m*|y*>")
        else:
            raise TypeError("Likely Incorrect or Missing value for the patient age")
        return temporal_val

    elif val_type == "range":
        try:
            match = re.match(r'(\d+)-(\d+)\s*([a-zA-Z]+)', temporal_val)
            if match is not None:
                begin_val, end_val = int(match.group(1)), int(match.group(2))
                if match.group(3).lower().startswith("d") :
                    cf = conv_factor["d"]
                elif match.group(3).lower().startswith("m"):
                    cf = conv_factor["m"]
                elif match.group(3).lower().startswith("y"):
                    cf = conv_factor["y"]
                else:
                    raise ValueError("TIme unit can't be parsed. Should start with d or m or y")

                temporal_val = begin_val*cf, end_val*cf
            else:
                raise ValueError("Incorrect value provided for the range")
        except Exception as e:
            print(f"Error in parsing the value {temporal_val} with error {e}")

        return temporal_val


    else:
        raise ValueError("Incorrect Value Type provided. Only single or range accepted")


def ecmo_mean_phys_normalization(data, p_id,  patient_ages, mean_normals, unit_to_use = "m", keep_old_columns=False):
    assert unit_to_use == "d" or unit_to_use == "m" or unit_to_use == "y"

    # We would find the row in the patient_ages with the corresponding patient id
    p_age = patient_ages[patient_ages["Patient ID"] == p_id]["Age"].values[0]
    p_age = ecmo_process_temporal_value(temporal_val=p_age, val_type="single",
                                        conv_factor=conversion_factor[unit_to_use])

    # We would parse the temporal units from the mean_normals file so that we can use proper values
    mean_normals["age"] = mean_normals["age"].transform(lambda x:
                                                        ecmo_process_temporal_value(temporal_val = x,
                                                                                    val_type = "range",
                                                                                    conv_factor= conversion_factor[unit_to_use]))
    mean_normals["age_low"] = mean_normals["age"].transform(lambda x: x[0])
    mean_normals["age_high"] = mean_normals["age"].transform(lambda x: x[1])

    # We would get the relevant row for the data
    mean_normals_for_patient = mean_normals[(mean_normals["age_low"] <= p_age) &
                                            (mean_normals["age_high"] > p_age)].iloc[0]

    # We would remove any columns which do not have columns beginning with
    columns_to_consider = [c for c in mean_normals_for_patient.index if c.startswith("mean")]
    mean_normals_for_patient = mean_normals_for_patient[columns_to_consider].astype(float)
    columns_to_consider = list(set([c.replace("mean_", "") for c in columns_to_consider]))

    for col in columns_to_consider:
        data[col] = data[col]/mean_normals_for_patient[f"mean_{col}"]

    return data

def ecmo_percentile_phys_10_90_binning(data: pd.DataFrame, p_id:int, patient_ages, percentile_range, unit_to_use ="m",
                                       keep_old_columns=False):
    assert unit_to_use == "d" or unit_to_use == "m" or unit_to_use == "y", "Incorrect unit provided"

    # We would find the row in the patient_ages with the corresponding patient id
    pt_age = patient_ages[patient_ages["Patient ID"] == p_id]["Age on ECMO"]
    pt_age = pt_age.values[0]
    pt_age = ecmo_process_temporal_value(pt_age, "single", conversion_factor[unit_to_use])

    # We would parse the temporal units from the mean_normals file so that we can use proper values
    percentile_range["age"] = percentile_range["age"].transform(lambda x:
                                                                ecmo_process_temporal_value(x, "range",
                                                                                       conversion_factor[unit_to_use]))
    percentile_range["age_low"] = percentile_range["age"].transform(lambda x: x[0])
    percentile_range["age_high"] = percentile_range["age"].transform(lambda x: x[1])

    # We would get the relevant row for the patient
    try:
        percentile_10_90_for_patient = percentile_range[(percentile_range["age_low"] <= pt_age) &
                                                        (percentile_range["age_high"] > pt_age)].iloc[0]
    except:
        print(f"Error in finding the row for patient {p_id} with age {pt_age}")
        return data

    # We would remove any columns which end with _10 or _90
    columns_to_consider = [c for c in percentile_10_90_for_patient.index if (c.endswith("_10") or c.endswith("_90"))]
    percentile_10_90_for_patient = percentile_10_90_for_patient[columns_to_consider].astype(float)
    columns_to_consider = list(set([c.replace("_10", "").replace("_90", "") for c in columns_to_consider]))
    for col in columns_to_consider:
        # if we want to keep the old columns, we would add the columns to the data as the <COL>_non_normalized column
        # where col is the original column name
        if keep_old_columns:
            data[f"{col}_non_normalized"] = data[col]

        data[f"{col}_low"] = data[col].transform(lambda x: 1 if percentile_10_90_for_patient[f"{col}_10"] > x else 0)
        data[f"{col}_high"] = data[col].transform(lambda x: 1 if percentile_10_90_for_patient[f"{col}_90"] < x else 0)
        data[col] = data[col].transform(lambda x: 1 if (percentile_10_90_for_patient[f"{col}_10"] > x or
                                                        percentile_10_90_for_patient[f"{col}_90"] < x) else 0)
    return data


def ecmo_percentile_phys_50_ratio(data: pd.DataFrame, p_id:int, patient_ages, percentile_range, unit_to_use ="m",
                                       keep_old_columns=False):
    """
    """
    assert unit_to_use == "d" or unit_to_use == "m" or unit_to_use == "y", "Incorrect unit provided"

    # We would find the row in the patient_ages with the corresponding patient id
    pt_age = patient_ages[patient_ages["Patient ID"] == p_id]["Age on ECMO"]
    pt_age = pt_age.values[0]
    pt_age = ecmo_process_temporal_value(pt_age, "single", conversion_factor[unit_to_use])

    # We would parse the temporal units from the mean_normals file so that we can use proper values
    percentile_range["age"] = percentile_range["age"].transform(lambda x:
                                                                ecmo_process_temporal_value(x, "range",
                                                                                       conversion_factor[unit_to_use]))
    percentile_range["age_low"] = percentile_range["age"].transform(lambda x: x[0])
    percentile_range["age_high"] = percentile_range["age"].transform(lambda x: x[1])

    percentile_50_patient = pd.Series()
    # We would get the relevant row for the patient
    try:
        percentile_50_patient = percentile_range[(percentile_range["age_low"] <= pt_age) &
                                                        (percentile_range["age_high"] > pt_age)].iloc[0]
    except:
        print(f"Error in finding the row for patient {p_id} with age {pt_age}")

    columns_to_consider = [c for c in percentile_50_patient.index if "age" not in c]
    for col in columns_to_consider:
        # if we want to keep the old columns, we would add the columns to the data as the <COL>_non_normalized column
        # where col is the original column name
        if keep_old_columns:
            data[f"{col}_non_normalized"] = data[col]

        data[col] = data[col].transform(lambda x: x / percentile_50_patient[col])

    return data


def ecmo_feature_normal_range_based_binning(data: pd.DataFrame, feature_ranges):
    """
    The method to bin the different features based on whether the feature value is inside or outside the normal range
    """
    for feature, normal_range in feature_ranges.items():
        low, high = normal_range
        data[feature] = data[feature].transform(lambda x: 0 if low <= x <= high else 1)

    return data


def ecmo_normalize_data(data: pd.DataFrame, p_id: int,  normalize_method: str = "age_based_percentile", **kwargs):
    """
    Helper method for normalizing the ECMO data
    :param data: The data to normalize
    :param p_id: The patient id considered
    :param normalize_method: The normalization method we want to use to normalize our data
    :param kwargs: Additional arguments
    :returns: The normalized dataframe
    """

    # If we want to normalize using the percentile data (The latest approach)
    if normalize_method == "age_based_percentile":
        data = ecmo_percentile_phys_10_90_binning(data=data, p_id=p_id, patient_ages=kwargs["patient_ages"],
                                                  percentile_range=kwargs["normalize_desc"], keep_old_columns=True)
    # If we want to normalize using the age means of physiological variables
    elif normalize_method == "age_based_mean":
        data = ecmo_mean_phys_normalization(data=data, p_id = p_id, patient_ages=kwargs["patient_ages"],
                                            mean_normals=kwargs["normalize_desc"],
                                            unit_to_use=kwargs["unit_to_use"], keep_old_columns=True)
    elif normalize_method == "age_based_percentile_ratio":
        data = ecmo_percentile_phys_50_ratio(data=data, p_id=p_id, patient_ages=kwargs["patient_ages"],
                                             percentile_range=kwargs["normalize_desc"], keep_old_columns=True)
    # We have not implemented or tested any other kind of normalization
    else:
        raise NotImplementedError("Other normalization methods are not implemented")

    return data

def ecmo_feature_mapping_pelican_to_old_hourly(x: pd.DataFrame, columns_map: dict = old_to_new_feature_mapping):
    """
    The method for creating the new dataframe for the series
    :param x: The pelican dataframe to map to old_hourly
    :param columns_map: The columns to map from old to new featurs
    """
    updated_pelican_df = pd.DataFrame()
    function_map = {"identity": {"func": lambda d: d,"axis": 0},
                    "min": {"func": lambda d: d.min(), "axis": 1},
                    "max": {"func": lambda d: d.max(), "axis": 1},
                    "mean": {"func": lambda d: d.mean(), "axis": 1}}

    for old_col_name, new_cols in columns_map.items():
        # We would get the data for the new columns and transform the data based on the mapping
        temp_df = x.reindex(columns=new_cols["features"])
        temp_df = temp_df.map(lambda y: extract_numeric(value=y))
        updated_pelican_df[old_col_name] = temp_df.apply(**function_map[new_cols["mapping"]])

    return updated_pelican_df

def ecmo_delta_or_relative_computation(x, col, mode="delta", time_slice="bckt", index_0 = 24, index_1 = -1):
    # We can't subtract using Nones
    if not np.isin(index_0, x[time_slice].values) or not np.isin(index_1, x[time_slice].values):
        print("The dataframe considered is missing the indices considered for comparison")
        return 0
    a, b = x[x[time_slice] == index_0][col].values[0],  x[x[time_slice] == index_1][col].values[0]
    if a is not None and b is not None:
        if mode == "delta":
            return np.abs(a - b)
        elif mode == "relative":
            return np.abs((a - b)/ b) if b != 0 else 0
        else:
            raise ValueError("Incorrect value for mode provided")

    return 0

def extract_numeric(value, pattern=r'[><]?(-?\d*\.?\d+)'):
    """
    The method would extract the numeric value from the string
    :param value: The value to extract the numeric value from
    :param pattern: The regex pattern used to extract the numeric value
    :return: The numeric value extracted from the string
    """
    pattern = re.compile(pattern)
    if isinstance(value, str):
        match = pattern.match(value)
        if match:
            return float(match.group(1))
    elif isinstance(value, (int, float)):
        return value
    else:
        return None

    return None


def aggregate_temporal_data(df: pd.DataFrame, filter_range: list, time_col: str, aggregation_type: str | dict,
                            **kwargs):
    """
    The method to aggregate the temporal data based on the time column
    :param df: The data frame to aggregate
    :param filter_range: The range of values to filter the data. It should be a list of two values and is inclusive
    :param time_col: The time column to use for aggregation
    :param aggregation_type: The type of aggregation to use
    :returns: The aggregated feature values or None in case of error
    """
    bins = 5
    if kwargs:
        bins = kwargs["bins"] if kwargs["bins"] else 5
    labels = [i+1 for i in range(bins)]

    filtered_range_data = df[(df[time_col] >= filter_range[0]) & (df[time_col] <= filter_range[1])]
    additional_values = {}


    # We would aggregate the different columns based on the aggregation type
    string_to_func_map = {"mean": "mean", "max": "max", "min": "min", "sum": "sum", "median": "median",
                          "count": "count",
                          "boolean": lambda x: 1 if x.any() else 0,
                          "frac": lambda x: x.sum()/len(x),
                          "bins": lambda x: pd.cut(x.sum()/len(x), bins=bins, labels=labels),
                          # Delta and relative are defined seoara
                          "delta": lambda x, col: ecmo_delta_or_relative_computation(x, col, "delta"),
                          "relative": lambda x, col: ecmo_delta_or_relative_computation(x, col, "relative")
                          }

    # If we need to apply a single aggregation across all columns
    if isinstance(aggregation_type, str):
        assert aggregation_type in string_to_func_map, "Incorrect aggregation type provided"
        agg_values_dict = {}
        for col in filtered_range_data:
            key_name = f"{aggregation_type}_{col}" if aggregation_type == "delta" or aggregation_type == "relative" \
                else col
            if filtered_range_data[col].isna().all():
                agg_values_dict[key_name] = np.nan
            else:
                agg_values_dict[key_name] = filtered_range_data[col].agg(string_to_func_map[aggregation_type])

        return agg_values_dict


    elif isinstance(aggregation_type, dict):
        # Get the aggregation function for the different columns (Currently ignore the ones in the form of the list)
        for feature, agg in aggregation_type.items():
            if isinstance(agg, str):
                if agg not in string_to_func_map.keys():
                    raise ValueError("Incorrect aggregation function provided")
            elif isinstance(agg, list):
                if not all([agg_f in string_to_func_map.keys() for agg_f in agg]):
                    raise ValueError("Incorrect aggregation function provided")
            else:
                raise ValueError("Aggregation functions can't be parsed if "
                                 "not provided as a single string or list of functions")

        col_func = {col: func for col, func in aggregation_type.items() if not isinstance(func, list)}
        col_func_list = {col: func for col, func in aggregation_type.items() if  isinstance(func, list)}

        # We need to handle the delta and relative functions
        cols_to_delete_from_col_func = []
        for col, agg_type in col_func.items():
            if agg_type == "delta" or agg_type == "relative":
                if df[col].isna().all():
                    additional_values[f"{agg_type}_{col}"] = np.nan
                else:
                    additional_values[f"{agg_type}_{col}"] = string_to_func_map[agg_type](df, col)
                # Remove the column from the filtered data range as it has already been computed
                filtered_range_data = filtered_range_data.drop(columns=[col])
                cols_to_delete_from_col_func.append(col)

            elif filtered_range_data[col].isna().all():
                additional_values[col] = np.nan
                filtered_range_data = filtered_range_data.drop(columns=[col])
                cols_to_delete_from_col_func.append(col)

            else:
                continue

        # Delete the columns from the col_func dict which have already been processed
        col_func = {k: v for k, v in col_func.items() if k not in cols_to_delete_from_col_func}

        # We need to handle columns which contain multiple aggregations.
        # Mainly used when we have delta or relative computation in addition
        for col, agg_list in col_func_list.items():
            for agg_type in agg_list:
                # We handle delta and relative aggregation separately. We need to pass in the full time series
                # instead of just passing the filtered range
                if agg_type == "delta" or agg_type == "relative":
                    if df[col].isna().all():
                        additional_values[f"{agg_type}_{col}"] = np.nan
                    else:
                        additional_values[f"{agg_type}_{col}"] = string_to_func_map[agg_type](df, col)

                # Otherwise, we just assign the aggregated value
                else:
                    if col not in additional_values.keys():
                        additional_values[col] = dict()

                    # Handling series with only None Values
                    if filtered_range_data[col].isna().all():
                        additional_values[col][agg_type] = np.nan
                    else:
                        if isinstance(string_to_func_map[agg_type], str):
                            additional_values[col][agg_type] = filtered_range_data[col].agg(agg_type)
                        else:
                            additional_values[col][agg_type] = string_to_func_map[agg_type](filtered_range_data[col])

            filtered_range_data = filtered_range_data.drop(columns=[col])

        col_func = {col: string_to_func_map[func] for col, func in col_func.items()}
        agg_values_dict = filtered_range_data.agg(col_func).to_dict()
        agg_values_dict.update(additional_values)
        return agg_values_dict

    else:
        raise ValueError("Incorrect aggregation type provided. Only string or dict accepted")


# TODO: Alter the code to allow for the same data to be  transformed and aggregated even after being aggregated before
def construct_patient_aggregated_features(df: pd.DataFrame, filter_range, config, **kwargs):
    # The config is the value for the data_count_classifiers_dataset dictionary
    # We would aggregate the different columns based on the aggregation type

    # If we want to use ordinal binning, we need to provide the number of bins as an additional argument
    patient_features_dict = {}
    operations_order = sorted(config.keys(), key = lambda x: x.split("_")[-1])
    for operation in operations_order:
        if "aggregation" in operation:
            aggregated_values = aggregate_temporal_data(df=copy.deepcopy(df), filter_range=filter_range,
                                                        time_col = kwargs["time_col"],
                                                        aggregation_type=config[operation])
            patient_features_dict.update(aggregated_values)
        elif "series_transformation" in operation:
            columns_to_transform = list(config[operation].keys())
            df[columns_to_transform] = df[columns_to_transform].transform(config[operation])

        elif "single_value_transformation" in operation:
            for col, func in config[operation].items():
                if col in patient_features_dict.keys():
                    patient_features_dict[col] = func(patient_features_dict[col])
                else:
                    raise ValueError(f"The feature {col} is not present in the dictionary of aggregated values")
        else:
            raise ValueError("Operation not supported! Only feature transformation and "
                             "aggregation supported in the current iteration")

    return patient_features_dict

def subsample_temporal_feature(df: pd.DataFrame, resolution_given: str = "sec", resolution_to: str = "hour",
                               temporal_feature_name = "time", subsampling_strategy: str | None = "mean" ):
    assert subsampling_strategy in ["mean", "max", "min", "first", "last"], "Not supported subsampling strategy"
    conversion_dict = {"sec": {"sec": 1, "min": 1/60, "hour": 1/3600},
                       "min": {"sec": 60, "min": 1, "hour": 1/60 },
                       "hour": {"sec": 3600, "min": 60, "hour": 1}}
    cf = conversion_dict[resolution_given][resolution_to]

    # Extract the numerical values from expressions like >20
    df = df.map(lambda x: extract_numeric(x)).astype(float)
    df[temporal_feature_name] = df[temporal_feature_name].transform(lambda x: np.floor(x*cf).astype(int))
    df = df.groupby(temporal_feature_name, as_index=False).agg(subsampling_strategy)
    return df

def remove_hanging_data_snippets(df, max_time_for_drop_in_sec=200, change_from_previous_none_percent=0.5):
    for feature in df.columns:
        data = pd.Series(copy.deepcopy(df[feature].values))
        # We would create a mask for the None values
        not_nan = data.notna()

        # We would identify the groups of continuous not None values
        groups = (not_nan != not_nan.shift()).cumsum()

        # We would calculate the length and mean for each segment
        segment_info = data.groupby(groups).agg(
            length='size',
            mean='mean',
            start_index= 'idxmin',
        )

        segment_info['previous_mean'] = segment_info['mean'].shift()
        segment_info['previous_mean_impute'] = segment_info['previous_mean'].ffill(axis=0)

        segment_info['deviation'] = abs(segment_info['mean'] - segment_info['previous_mean_impute']) / abs(
            segment_info['previous_mean_impute'])

        # Identify segments to replace with None based on length and deviation
        noise_segments = segment_info.index[
            (segment_info['length'] <= max_time_for_drop_in_sec // 5) &
            (segment_info['deviation'] > change_from_previous_none_percent)
            ]

        # Replace identified noise segments with None in the original data
        for group_id in noise_segments:
            data[groups == group_id] = None

        df[feature] = data

        return df

# Method to remove the rate of change artifacts from the data
def remove_noisy_artifacts(df: pd.DataFrame, **kwargs):
    """
    Method to remove noisy artifacts from the data
    """
    original_cols = df.columns
    if not kwargs:
        return df

    imputation_method = "ffill"
    if kwargs and "imputation" in kwargs:
        imputation_method = kwargs["imputation"]

    if kwargs and kwargs["threshold"]:
        # We need to perform threshold based filtering
        for feature, feature_threshold in kwargs["threshold"].items():
            if feature not in original_cols:
                continue
            low, high = feature_threshold
            df[feature] = df[feature].astype(float)
            df[f"exclude_threshold_{feature}"] = (df[feature].astype(float) < low ) | (df[feature].astype(float) > high)
            df.loc[df[f"exclude_threshold_{feature}"], feature] = np.nan
        # Once the threshold based filtering has been performed, we would impute the data and remove
        # any dangling threads

        # TODO: Add code to selectively remove data snippets only in cases when it is warranted
        df = remove_hanging_data_snippets(df)
        df = ecmo_impute_missing_values(df, imputation_method=imputation_method, time_col_name="time")

    if kwargs and kwargs["rate_of_change"]:
        t_feature = kwargs["temporal_feature_name"]
        # We need to perform rate of change based filtering on the data
        for feature, change in kwargs["rate_of_change"].items():
            if feature not in original_cols:
                continue
            feature_rate_of_change, interval = change
            time_steps_to_exclude = []
            df[feature] = df[feature].astype(float)
            positive_change = np.append(False, np.diff(df[feature]) > feature_rate_of_change)
            negative_change = np.append(False, np.diff(df[feature]) < -1*feature_rate_of_change)

            for k in [(positive_change, negative_change), (negative_change, positive_change)]:
                for i in range(1, len(k[0]) - interval//5):
                    if k[0][i]:
                        max_j = None
                        for j in range(i+1, i + interval//5):
                            if k[1][j]:
                                max_j = j

                        if max_j is not None:
                            start_time = df.loc[i,t_feature]
                            end_time = df.loc[max_j, t_feature]
                            time_steps_to_exclude.append((start_time, end_time))

            df[f"exclude_roc_{feature}"] = False
            for (t_begin, t_end) in time_steps_to_exclude:
                df.loc[(df[t_feature] >= t_begin) & (df[t_feature] <= t_end), f"exclude_roc_{feature}"] = True

            df.loc[df[f"exclude_roc_{feature}"], feature] = np.nan

        df = remove_hanging_data_snippets(df)
        df = ecmo_impute_missing_values(df, imputation_method=imputation_method, time_col_name="time")

    return df[original_cols]