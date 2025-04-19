# LLM-guided-CBN
This repository contains the supplementary material and the code for **LLM-guided Causal Bayesian Network construction for pediatric patients on ECMO** for **AIME 2025** conference.


## Components
### Supplementary Text
The file **Supplementary Text.pdf** contains additional information, including expert-elicited and data-driven graphs, LLM prompts, and all 5 responses for each LLM, as well as the graphs before and after subtractive and full refinement. Do note that all the results added in the supplementary are on the original dataset and not on bootstrap samples

### Running the Code
1. It is advised to use an IDE like PyCharm to run the code, as some of the code might cause errors if used from the terminal

2. Assuming the code is properly opened in an IDE, and all the dataset file is provided (not included as part of the code)
   1. Create a conda environment using the ```requirements.txt``` and set it as the environment for the project
      **NOTE*:* You might run into errors installing the cdt package used to compute metrics. Please refer to [Causal Discovery Tool Box]{https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html} to properly install the
      package

    2. Create bootstrap samples for the data by running the file ```datasets/create_bootstrap_samples.py``` file. Set the bootstrap samples by changing the variable ```num_samples``` in the file.

    3. Run the data only baselines by running the files `in ```data_based_methods``` folder. Run the following files in order:
        1. Run ```learn_from_data_gss.py``` for greedy search and score on the data
        2. Run ```learn_from_data_oc.py``` for PC (Peter and Clarke) algorithm on the data
        3. Run ```learn_from_data_fci.py``` for FCI (Fast Causal Inference) algorithm on the data  

    4. For subtractive refinement, run ```refine_1.py```

    5. For full refinement, run ```refine_1.py``` followed by ```refine_2.py```

    6. Compute the metrics (SHD, SID, Spurious Edges) by running the ```compute_metrics.py```. Ensure the number of bootstrap samples is the same as the number of samples

    7. Plot the graphs for the different methods and samples by running the file ```plot_graphs.py```
