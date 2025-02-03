import pygraphviz as pgv
from pgmpy.readwrite import BIFReader
import os

# 1. Define the path to the directory containing .bif files.
bif_dir = os.path.join(os.getcwd(), "results", "results_refine_sub")  # Replace with your directory path

# 2. Get a list of all .bif files in the directory.
bif_files = [f for f in os.listdir(bif_dir) if f.endswith('.bif')]

# 3. Loop through each .bif file and plot it.
for bif_file in bif_files:
    bif_file_path = os.path.join(bif_dir, bif_file)

    # Read the .bif file
    reader = BIFReader(bif_file_path)
    model = reader.get_model()

    # Convert to pygraphviz
    graph = pgv.AGraph(directed=True)
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())

    # Layout and draw
    graph.layout(prog='dot')
    output_filename = bif_file[:-4] + '.png'  # Replace .bif with .png
    graph.draw(os.path.join(bif_dir, output_filename))

print("All .bif files have been plotted.")