from os import path
import numpy as np
import pandas as pd

from structure_learning import refine_bn, StructuredBicScore

from pgmpy.readwrite import BIFWriter
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

name = "post_data_only"
data = pd.read_csv(path.join("ecmo_data", f"post_pelican.csv"))


M0 = BayesianNetwork()
M0.add_nodes_from(data.columns)

state_names = { col: list(range(2)) for col in data.columns}

M1 = refine_bn(M0.copy(), data, state_names, scoring_method="bicscore", tabu_length=0, black_list=[])
M1.fit(data, state_names=state_names, estimator=BayesianEstimator,  prior_type="dirichlet", pseudo_counts=1)
BIFWriter(M1).write_bif(path.join("results", f"{name}--1.bif"))

scorer = StructuredBicScore(data, state_names=state_names)
M2 = refine_bn(M0.copy(), data, state_names=state_names, scoring_method=scorer, tabu_length=0, black_list=[])
cpds = []
for node in M2.nodes:
    parents = M2.get_parents(node)
    _, cpd = scorer.local_score(node, parents, return_cpd=True)
    cpds.append(cpd)
M2.add_cpds(*cpds)
BIFWriter(M2).write_bif(path.join("results", f"{name}--2.bif"))