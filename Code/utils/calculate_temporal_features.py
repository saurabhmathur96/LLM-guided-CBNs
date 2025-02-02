# This is the file for calculating the temporal features for different patients.
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import bds


class UnivariateTemporalFeatures:
    @staticmethod
    def compute_curvature(data: pd.DataFrame, temporal_feature ="time", curve_type="quadratic", feature_list=None):
        curvature = dict()
        if curve_type == "quadratic":
            def quadratic(x, a, b, c):
                return a * x**2 + b * x + c

            for col in data.columns:
                try:
                    if col != temporal_feature and col in feature_list:
                        if data[col].isnull().any():
                            curvature[col] = {"params": None, "covariances": None}
                        else:
                            params, covariance = curve_fit(quadratic, data[temporal_feature], data[col])
                            curvature[col] = {"params": params, "covariance": covariance}
                except TypeError:
                    curvature[col] = {"params": None, "covariance": None}

        else:
            raise NotImplementedError

        return curvature

    @staticmethod
    def compute_kurtosis(data: pd.DataFrame, ignore_features: list | None = None, feature_list=None):
        kurtosis_dict = data.kurt().to_dict()
        if ignore_features is not None and len(ignore_features) > 0:
            for feature in ignore_features:
                del kurtosis_dict[feature]
        kurtosis_dict = {k: v for k, v in kurtosis_dict.items() if k in feature_list}
        return kurtosis_dict

    @staticmethod
    def compute_bds_linearity(data: pd.DataFrame, temporal_feature="time", feature_list=None):
        linearity = dict()
        for col in data.columns:
            if col != temporal_feature and col in feature_list:
                try:
                    bds_stat, p_value = bds(data[col])
                    linearity[col] = {"bds_stat": float(bds_stat), "p_value": float(p_value)}
                except ValueError:
                    linearity[col] = {"bds_stat": None, "p_value": None}

        return linearity

    @staticmethod
    def compute_chao_shen_shannon_entropy(data: pd.DataFrame, temporal_feature="time",
                                          discretizing_bins: dict | int = 3, feature_list=None):
        entropy = dict()
        for col in data.columns:
            n_bins = None
            if col != temporal_feature and col in feature_list:
                try:
                    if isinstance(discretizing_bins, int):
                        n_bins = discretizing_bins
                    # if the data is not already categorical, we would discretize it into bins
                    elif col in discretizing_bins.keys():
                        if discretizing_bins[col]:
                            n_bins = discretizing_bins[col]
                        else:
                            continue
                    else:
                        pass
                    d = pd.cut(data[col], n_bins, labels=["0", "1", "2"])

                    # Once the column has been discretized, we would compute the entropy
                    symbol_counts = Counter(d)
                    n = len(d)
                    k = len(symbol_counts)

                    probabilities = np.array([count / n for count in symbol_counts.values()])
                    shannon_entropy = -np.sum(probabilities * np.log(probabilities) / np.log(2))
                    alpha = (k * (k - 1)) / (2 * n)
                    entropy[col] = float(shannon_entropy + alpha/n)
                except ValueError:
                    entropy[col] = np.nan

        return entropy

    @staticmethod
    def compute_skewness(data: pd.DataFrame, temporal_feature="time", feature_list=None):
        skewness = dict()
        for col in data.columns:
            if col != temporal_feature and col in feature_list:
                skewness[col] = float(skew(data[col]))

        return skewness

    @staticmethod
    def compute_trend(data: pd.DataFrame, temporal_feature="time", mode="linear_regression", feature_list=None):
        trends = dict()
        # We would impute the data using both ffill and bfill as data with m
        if mode == "linear_regression":
            for col in data.columns:
                if col != temporal_feature and col in feature_list:
                    try:
                        m = LinearRegression()
                        m.fit(pd.DataFrame(data[temporal_feature]), data[col])
                        trends[col] = m.coef_[0]
                    except:
                        trends[col] = None
        else:
            raise NotImplementedError

        return trends


    def compute_metrics(self, data: pd.DataFrame, metrics: dict | None):
        metric_computations = dict()
        if metrics is None:
            metrics = dict()

        for metric, metric_arguments in metrics.items():
            if metric == "curvature":
                metric_computations[metric] = self.compute_curvature(data, **metric_arguments)
            elif metric == "kurtosis":
                metric_computations[metric] = self.compute_kurtosis(data, **metric_arguments)
            elif metric == "bds_linearity":
                metric_computations[metric] = self.compute_bds_linearity(data, **metric_arguments)
            elif metric == "shannon_entropy":
                metric_computations[metric] = self.compute_chao_shen_shannon_entropy(data, **metric_arguments)
            elif metric == "skewness":
                metric_computations[metric] = self.compute_skewness(data, **metric_arguments)
            elif metric == "trend":
                metric_computations[metric] = self.compute_trend(data, **metric_arguments)
            else:
                raise NotImplementedError

        return metric_computations

if __name__ == "__main__":
    a = np.linspace(0, 1, 100)
    b = np.linspace(10, 20, 100)
    c = np.linspace(100, 1200, 100)

    df = pd.DataFrame({"a": a, "b": b, "c": c})

    cv =  UnivariateTemporalFeatures.compute_curvature(df, temporal_feature="a", curve_type="quadratic")
    kurto = UnivariateTemporalFeatures.compute_kurtosis(df, feature_list=["b", "c"])
    entr = UnivariateTemporalFeatures.compute_chao_shen_shannon_entropy(df, 'a', 3, feature_list=["b", "c"])
    trend = UnivariateTemporalFeatures.compute_trend(df, "a", feature_list=["b", "c"])
    bds_lin = UnivariateTemporalFeatures.compute_bds_linearity(df, "a", feature_list=["b", "c"])
    skew = UnivariateTemporalFeatures.compute_skewness(df, "a", feature_list=["b", "c"])
    print(cv, kurto, entr, trend, bds_lin, skew, sep="\n")




