import os
from os import path
import pandas as pd
from pathlib import Path
from structure_learning import refine_bn, StructuredBicScore

from pgmpy.readwrite import BIFWriter
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

tld = Path(__file__).resolve().parent.parent
data_path = os.path.join(tld, "datasets")
if __name__ == "__main__":
    results_path = os.path.join(tld, "results")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "data_only"), exist_ok=True)
    print("Refining with data only")
    for file in os.listdir(data_path):
        if file.endswith(".csv") and file.startswith("bootstrap"):
            sample_id = file.split("_")[2][:-4]
            print("Processing bootstrap sample", sample_id)
            name = f"gss_{sample_id}"
            data = pd.read_csv(path.join(data_path, file))

            M0 = BayesianNetwork()
            M0.add_nodes_from(data.columns)

            state_names = { col: list(range(2)) for col in data.columns}

            M1 = refine_bn(M0.copy(), data, state_names, scoring_method="bicscore", tabu_length=0, black_list=[])
            M1.fit(data, state_names=state_names, estimator=BayesianEstimator,  prior_type="dirichlet", pseudo_counts=1)
            BIFWriter(M1).write_bif(path.join(results_path, "data_only", f"{name}--1.bif"))

            scorer = StructuredBicScore(data, state_names=state_names)
            M2 = refine_bn(M0.copy(), data, state_names=state_names, scoring_method=scorer, tabu_length=0, black_list=[])
            cpds = []
            for node in M2.nodes:
                parents = M2.get_parents(node)
                _, cpd = scorer.local_score(node, parents, return_cpd=True)
                cpds.append(cpd)
            M2.add_cpds(*cpds)
            BIFWriter(M2).write_bif(path.join(results_path, "data_only", f"{name}--2.bif"))