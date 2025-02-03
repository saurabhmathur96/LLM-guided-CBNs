# Submission for AIME 2025
This is the repository containing the supplementary and the code for **LLM-guided Causal Bayesian Network construction for pediatric patients on ECMO**.


## Attachments
1. **Code**: Contains all the code relevant to the project (The data is absent)

2. **Supplementary Text.pdf**: The pdf containing all the additional information, including expert elicited and data driven graphs, LLM prompts and all 5 responses for each LLM, as well as the graphs, before and after subtractive and full refinement. It is advised to download the pdf file directory as the file content may be too big to be rendered properly online

## Running the Code
1. To run the code in the Code directory, please follow the following steps (Assuming all the relevant data files are available and in relevant subdirectories).
    1. Create the environment
        ```bash
        cd Code
        conda create -n "ecmo" python=3.10
        conda activate ecmo
        pip install -r requirements.txt
    2. Process the pelican data to construct an hourly dataset for the 71 patients
        ```bash
        python process_pelican_data.py
        ```
    3. Construct the boolean dataset for ECMO
        ```bash
        python create_dataset_for_pelican.py
        ```
    4. Move the dataset to the relevant area and move to the directory BN-Refinement
        ```bash
        mkdir BN-Refinement-ECMO/ecmo_data
        mv post_pelican ./BN-Refinement-ECMO/ecmo_data
        cd BN_Refinement-ECMO
        ```
2. Once the dataset is generated, we can run the deletion-only refinement by selecting the relevant set of initial edges for combined by running 
    ```bash 
    python refine.py
    ```
    1. To change the graph used, change the initial edges provided by setting the initial edges to one of combined, gpt40, gemini, llama, deepseek in ```refine.py```
    2. To perform full refinement, comment out line 15 in ```structure_learning.py``` and comment line 19. This changes the kind of HillClimb used

3. To learn causal graphs using data
    ```bash
    python learn_from_data_pc.py # For Learning a PC model
    python refine_only_data.py # For Learning a Greedy Score and Search model
    ```

4. To compute the SHD and SID for the graphs run
    ```bash
    python compute_shd_sid_spurious.py
    ```
