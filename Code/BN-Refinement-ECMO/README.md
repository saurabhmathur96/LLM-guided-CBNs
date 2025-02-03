# BN-Refinement

This repository uses part of the code from "Modeling multiple adverse pregnancy outcomes:\\ Learning from diverse data sources" AIME (2024).
The files used as is are
  - `structure_learning.py` is a module that implements the BN refinement procedure.
  - `refine.py` reads an initial BN structure and a data set from disk and calls the refinement procedure to obtain a refined BN.

## Requirements 
- pgmpy==0.1.24
- scikit-learn==1.1.3

