# This is the method for learning the data using PGMPy PC algorithm
from pgmpy.estimators import PC
import pandas as pd
import numpy as np

data = pd.read_csv("ecmo_data/post_pelican.csv").astype(int)
print(data)
c = PC(data)
pdag = c.build_skeleton()
print(pdag)

# # Learn the structure
dag = c.estimate(return_type="dag")
print(dag.edges())

#