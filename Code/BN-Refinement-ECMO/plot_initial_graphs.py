# This is the file for plotting the initial edges provided by the user

import os
import pandas as pd
from pgmpy.models import BayesianNetwork

# In my working directory, if there are any files beginning with initial edges, we load it
# and plot the edges

if __name__ == "__main__":
    working_directory = os.getcwd()
    nodes = ["LowMAP", "HighMAP", "LowPlatelet", 'HighVIS', "LowPH", "HighLactate", "RelativePCO2", "NeurologicalInjury"]
    for file in os.listdir(working_directory):
        if file.startswith("initial_edges"):
            print(f"Plotting the edges from {file}")
            initial_edges_df = pd.read_csv(file)

            # We would construct a bayesian network from the dataframe
            # and plot the edges using pygraphviz
            bn = BayesianNetwork(
                list(zip(initial_edges_df["X"].tolist(), initial_edges_df["Y"].tolist()))
            )
            bn.add_nodes_from(nodes)
            bn.to_graphviz().draw(f"p_{file}.png", format="png", prog="dot")