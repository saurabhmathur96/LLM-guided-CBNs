from os import path
import numpy as np
import pandas as pd

from structure_learning import refine_bn, StructuredBicScore
from pgmpy.readwrite import BIFWriter
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, BicScore

name = "post_pelican"
initial_edges  = "combined"
data = pd.read_csv(path.join("ecmo_data", f"{name}.csv")).astype(int)

edges = []
for _, row in pd.read_csv(f"initial_edges_{initial_edges}.csv").iterrows():
    if (row.X not in data.columns) or (row.Y not in data.columns):
        print (f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
    else:
        edges.append((row.X, row.Y))
print (f"Loaded {len(edges)} edges.")

black_list = []
for i, row in pd.read_csv("black_list_post.csv").iterrows():
    if (row.X not in data.columns) or (row.Y not in data.columns):
        print (f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
    else:
        black_list.append((row.X, row.Y))
print (f"Loaded {len(black_list)} edges as black list.")

M0 = BayesianNetwork()
M0.add_nodes_from(data.columns)
M0.add_edges_from(edges)

state_names = { col: list(range(2)) for col in data.columns}

M1 = refine_bn(M0.copy(), data, state_names, scoring_method=BicScore(data, state_names=state_names),
               tabu_length=0, black_list=black_list)
M1.fit(data, state_names=state_names, estimator=BayesianEstimator,  prior_type="dirichlet", pseudo_counts=1)
BIFWriter(M1).write_bif(path.join("results", f"{name}_{initial_edges}-1.bif"))

scorer = StructuredBicScore(data, state_names=state_names)
M2 = refine_bn(M0.copy(), data, state_names=state_names, scoring_method=scorer, tabu_length=0, black_list=black_list)
cpds = []
for node in M2.nodes:
    parents = M2.get_parents(node)
    _, cpd = scorer.local_score(node, parents, return_cpd=True)
    cpds.append(cpd)
M2.add_cpds(*cpds)
BIFWriter(M2).write_bif(path.join("results", f"{name}_{initial_edges}-2.bif"))