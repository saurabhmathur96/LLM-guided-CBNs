# This is the method for learning the data using PGMPy PC algorithm
import os
import pandas as pd
import time
import json
from pathlib import Path
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork

tld = Path(__file__).parent.parent
data_path = os.path.join(tld, "datasets")
results_path = os.path.join(tld, "results")

if __name__ == '__main__':
    for file in os.listdir(data_path):
        if file.endswith(".csv") and file.startswith("bootstrap_sample_"):
            sample_id = file.split("_")[-1].split(".")[0]
            data = pd.read_csv(os.path.join(data_path, file)).astype(int)
            time.sleep(2)
            print(f"\nProcessing sample {sample_id} using PC algorithm")
            c = PC(data)
            pdag = c.build_skeleton()
            dag = c.estimate(return_type="dag")
            bn = BayesianNetwork()
            bn.add_nodes_from(data.columns)
            if len(dag.edges()) > 0:
                bn.add_edges_from(dag.edges())

            # Save the learned structure to a file
            output_file = os.path.join(results_path, "data_only", f"pc_{sample_id}.graph")

            # Write a json file with the learned structure containing a list of nodes and edges
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            graph = {"nodes": list(bn.nodes()), "edges": list(bn.edges())}
            with open(output_file, 'w') as f:
                json.dump(graph, f, indent=3)


#