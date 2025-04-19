import os
import pandas as pd

from pathlib import Path
from structure_learning import refine_bn, StructuredBicScore
from pgmpy.readwrite import BIFWriter, BIFReader
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, BicScore

tld = Path(__file__).parent
data_path = os.path.join(tld, "datasets")
results_path = os.path.join(tld, "results")
subtractive_load_path = os.path.join(tld, "results", "llm", "subtractive")

if __name__ == "__main__":
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm", "full"), exist_ok=True)
    for model_file in os.listdir(subtractive_load_path):
        if model_file.endswith("1.bif"):
            continue
        llm_type = model_file.split("_")[0]
        sample_id = model_file.split("_")[1].split(".")[0][0]
        print("Considering the edges from", llm_type, "and bootstrap sample",  sample_id)
        data = pd.read_csv(os.path.join(data_path, f"bootstrap_sample_{sample_id}.csv")).astype(int)
        model_edges = BIFReader(os.path.join(subtractive_load_path, model_file)).get_model().edges()
        print("Loaded", len(model_edges), "edges and data from sample", sample_id)

        black_list = []
        for i, row in pd.read_csv("black_list_post.csv").iterrows():
            if (row.X not in data.columns) or (row.Y not in data.columns):
                print(f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
            else:
                black_list.append((row.X, row.Y))
        print(f"Loaded {len(black_list)} edges as black list.")

        M0 = BayesianNetwork()
        M0.add_nodes_from(data.columns)
        M0.add_edges_from(model_edges)

        state_names = {col: list(range(2)) for col in data.columns}

        M1 = refine_bn(M0.copy(), data, state_names, scoring_method=BicScore(data, state_names=state_names),
                       tabu_length=0, black_list=black_list)
        M1.fit(data, state_names=state_names, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
        BIFWriter(M1).write_bif(os.path.join("results", "llm", "full", f"{llm_type}_{sample_id}-1.bif"))

        scorer = StructuredBicScore(data, state_names=state_names)
        M2 = refine_bn(M0.copy(), data, state_names=state_names, scoring_method=scorer, tabu_length=0,
                       black_list=black_list)
        cpds = []
        for node in M2.nodes:
            parents = M2.get_parents(node)
            _, cpd = scorer.local_score(node, parents, return_cpd=True)
            cpds.append(cpd)
        M2.add_cpds(*cpds)
        BIFWriter(M2).write_bif(os.path.join("results", "llm", "full", f"{llm_type}_{sample_id}-2.bif"))