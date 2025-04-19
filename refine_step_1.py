import os
import pandas as pd
from pathlib import Path
from structure_learning import refine_bn, StructuredBicScore
from pgmpy.readwrite import BIFWriter
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, BicScore

tld = Path(__file__).parent
data_path = os.path.join(tld, "datasets")
results_path = os.path.join(tld, "results")
initial_edges_path = os.path.join(tld, "initial_edge_lists")

if __name__ == "__main__":
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "llm", "subtractive"), exist_ok=True)
    for llm_edges_file in os.listdir(initial_edges_path):
        if llm_edges_file.startswith("initial_edges"):
            llm_type = " ".join(llm_edges_file.split("_")[2:])
            llm_type = llm_type.replace(".csv", "")
            print("Considering the LLM edges from", llm_type)
            initial_edges_df = pd.read_csv(os.path.join(initial_edges_path, llm_edges_file))

            # Remove any edges from initial_edges_df that are present in black_list
            black_list_df = pd.read_csv(os.path.join("black_list_post.csv"))
            print("Removing any black list edges from the initial edges")

            initial_edges_df = initial_edges_df.merge(black_list_df, on=["X", "Y"], how="left", indicator=True)
            initial_edges_df = initial_edges_df[initial_edges_df["_merge"] == "left_only"].drop(columns=["_merge"])

            # Once we have processed the initial edge list, we would iterate through the data path to
            # get the bootstrap data
            for file in os.listdir(data_path):
                if file.startswith("bootstrap") and file.endswith(".csv"):
                    # We will load the bootstrap data
                    data = pd.read_csv(os.path.join(data_path, file)).astype(int)
                    sample_id = file.split("_")[-1].split(".")[0]
                    print("\nProcessing bootstrap data for sample", sample_id)

                    edges = []
                    for _, row in initial_edges_df.iterrows():
                        if (row.X not in data.columns) or (row.Y not in data.columns):
                            print(f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
                        else:
                            edges.append((row.X, row.Y))
                    print(f"Loaded {len(edges)} edges.")

                    black_list = []
                    for i, row in black_list_df.iterrows():
                        if (row.X not in data.columns) or (row.Y not in data.columns):
                            print(f"Edge {(row.X, row.Y)} contains an unknown node. Skipping.")
                        else:
                            black_list.append((row.X, row.Y))
                    print(f"Loaded {len(black_list)} edges as black list.")

                    M0 = BayesianNetwork()
                    M0.add_nodes_from(data.columns)
                    M0.add_edges_from(edges)

                    state_names = {col: list(range(2)) for col in data.columns}

                    M1 = refine_bn(M0.copy(), data, state_names, scoring_method=BicScore(data, state_names=state_names),
                                   tabu_length=0, black_list=black_list, subtractive_refinement_only=True)
                    M1.fit(data, state_names=state_names, estimator=BayesianEstimator, prior_type="dirichlet",
                           pseudo_counts=1)
                    BIFWriter(M1).write_bif(os.path.join(results_path,
                                                         "llm", "subtractive",  f"{llm_type}_{sample_id}-1.bif"))

                    scorer = StructuredBicScore(data, state_names=state_names)
                    M2 = refine_bn(M0.copy(), data, state_names=state_names, scoring_method=scorer, tabu_length=0,
                                   black_list=black_list,
                                   subtractive_refinement_only=True)
                    cpds = []
                    for node in M2.nodes:
                        parents = M2.get_parents(node)
                        _, cpd = scorer.local_score(node, parents, return_cpd=True)
                        cpds.append(cpd)
                    M2.add_cpds(*cpds)
                    BIFWriter(M2).write_bif(os.path.join(results_path,
                                                         "llm", "subtractive", f"{llm_type}_{sample_id}-2.bif"))