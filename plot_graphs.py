# This is the file for plotting all the graphs
import os
import json
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader
tld = Path(__file__).parent
results_dir = os.path.join(tld, "results")
if __name__ == "__main__":
    # We would perform a recursive search over the files in the results directory and plot them while maintaining the
    # same substructure
    for dirpath, dirnames, filenames in os.walk(results_dir):
        dirpath_2 = dirpath.replace("/results", "/cbn_graphs")
        os.makedirs(dirpath_2, exist_ok=True)
        for file in filenames:
            if not (file.endswith(".graph") or file.endswith(".bif")):
                continue
            print("Plotting graph in file", os.path.join(dirpath, file))
            with open(os.path.join(dirpath, file), "r") as f:
                if file.endswith(".graph"):
                    graph = json.load(f)
                    bn = BayesianNetwork()
                    bn.add_nodes_from(graph["nodes"])
                    bn.add_edges_from(graph["edges"])
                    graph_name = file.replace(".graph", "")
                elif file.endswith(".bif"):
                    bn = BIFReader(path=os.path.join(dirpath, file)).get_model()
                    graph_name = file.replace(".bif", "")
                else:
                    raise ValueError("Incorrect file name")

            bn.to_graphviz().draw(os.path.join(dirpath_2, f"{graph_name}.png"), prog="dot")
