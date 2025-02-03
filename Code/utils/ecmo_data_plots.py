# These are the utility functions for plotting the different ECMO plots, we need to consider for our data
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Plot the relation between the number of patients and the minimum time on ECMO
def plot_count_ecmo_time(p_data_dict: dict, time_col_name: str, save_path: str | None = None):
    """
    :param p_data_dict: The data dictionary containing the different patients
    :param time_col_name: The column name for the time column in the data
    :param save_path: The path to save the figure at
    """
    max_bckt_vals = []
    min_bckt_vals = []
    for p_id, patient_data in p_data_dict.items():
        p_df = patient_data["data"]
        # We would get the minimum and the maximum time for the number of patients
        min_bckt_vals.append(p_df[time_col_name].min())
        max_bckt_vals.append(p_df[time_col_name].max())
    
    
    # Once we have the minimum and maximum time for the different patients, we would plot
    fig, (ax_min, ax_max) = plt.subplots(1, 2, figsize=(24, 12))
    
    # We would filter any values greater than 0 from min_bckt_vals
    min_bckt_vals = [m for m in min_bckt_vals if m < 0]
    ax_min.hist(data=min_bckt_vals, bins=range(min(min_bckt_vals), max(max_bckt_vals) + 2), edgecolor='blue', alpha=0.7)
    ax_min.xlabel('Total Time Pre ECMO')
    ax_min.ylabel('Number of Patients')
    ax_min.title('Histogram of Time Pre Canulation')
    
    # We would filter any values less than 0 from max_bckt vals
    ax_max.hist(data=max_bckt_vals, bins=range(min(max_bckt_vals), max(max_bckt_vals) + 2), edgecolor='orange', alpha=0.7)
    ax_max.xlabel('Total Time Post ECMO')
    ax_max.ylabel('Number of Patients')
    ax_max.title('Histogram of Time Post Canulation')
                 
    plt.tight_layout()
    if not save_path:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "ecmo_count-time.svg"), bbox_inches='tight')

def plot_df_characteristics(df: pd.DataFrame, to_plot: str | list, **kwargs):
    """
    :param df: The dataframe to consider for plot
    :param to_plot: The statistics about the dataframe to plot
    :param kwargs: Additional arguments to pass
    """
    if isinstance(to_plot, str):
        to_plot = [to_plot]
    
    for m in to_plot:
        if m == "mutual_information":
            assert "target" in kwargs.keys()
            # We would train a sklearn classifier
            X = df.drop(columns=kwargs["target"])
            y = df[kwargs["target"]]

            # We would train a classifier
            mi = mutual_info_classif(X, y)

            # We would plot the mutual information
            plt.figure(figsize=(24, 12))
            plt.bar(X.columns, mi)
            plt.xticks(rotation=90)
            plt.ylabel("Mutual Information")
            plt.xlabel("Features")

            plt.title("Mutual Information")
            if kwargs and "save_path" in kwargs.keys():
                plt.savefig(os.path.join(kwargs["save_path"], "mutual_information.png"),  bbox_inches='tight')
            plt.show()

            
        if m == "covariance":
            # We would plot the covariance matrix
            plt.figure(figsize=(24, 12))
            plt.imshow(df.cov())
            plt.colorbar()
            plt.xticks(range(len(df.columns)), df.columns, rotation=90)
            plt.yticks(range(len(df.columns)), df.columns)
            plt.title("Covariance Matrix")
            plt.show()
            plt.close()
    

def plot_metrics_dict(metrics, **kwargs):
    """
    The method for plotting the different metrics on the result
    :param metrics: The metrics to plot
    :param kwargs: Additional arguments to pass
    """

    # Refactor the metrics dictionary for easy plotting
    refactored_metrics_dict = {}
    # We want to plot the same metric for the different models together
    for data_to_consider, models in metrics.items():
        for model_name, model_metrics in models.items():
            for metric_name, metric_vals in model_metrics.items():
                refactored_metrics_dict[(data_to_consider, model_name, metric_name)] = metric_vals

    # Once we have the refactored metrics, we would plot the metrics, we would plot the metric values for the different
    # models for the same metrics together with the data to consider being used to color the different values
    data_types = list(set([d[0] for d in refactored_metrics_dict.keys()]))
    model_names = list(set([d[1] for d in refactored_metrics_dict.keys()]))
    metric_names = list(set([d[2] for d in refactored_metrics_dict.keys()]))

    # We would create a subplot for each of the metrics
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 8 * len(metric_names)))

    if len(metric_names) == 1:
        axes = [axes]

    # We would also get the color map for the different data types
    color_map = plt.cm.get_cmap('tab20', len(data_types))

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        for j, data_type in enumerate(data_types):
            for model_name in model_names:
                metric_mean, metric_std = refactored_metrics_dict[(data_type, model_name, metric_name)]["mean"], \
                                            refactored_metrics_dict[(data_type, model_name, metric_name)]["std"]
                ax.errorbar(j, metric_mean, yerr=metric_std, fmt='o', color=color_map(j), label=model_name)

        ax.set_xticks(range(len(data_types)))
        ax.set_xticklabels(data_types)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} for the different models")
        ax.legend()

    plt.tight_layout()
    if kwargs and "save_path" in kwargs.keys():
        plt.savefig(kwargs["save_path"], bbox_inches='tight')
    plt.show()