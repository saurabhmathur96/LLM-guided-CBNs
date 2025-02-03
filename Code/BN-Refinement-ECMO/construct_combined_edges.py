# This is the file for combinining the different edge lists provid
import pandas as pd

combined_df = pd.DataFrame( columns=["X", "Y"])

# Add the deepseek edges to the combined dataframe
deepseek_df = pd.read_csv("initial_edges_deepseek.csv")
deepseek_df = deepseek_df.rename(columns={"X": "X", "Y": "Y"})
combined_df = pd.concat([combined_df, deepseek_df], ignore_index=True)

# Add the gpt4o edges to the combined dataframe
gpt4o_df = pd.read_csv("initial_edges_gpt4o.csv")
gpt4o_df = gpt4o_df.rename(columns={"X": "X", "Y": "Y"})
combined_df = pd.concat([combined_df, gpt4o_df], ignore_index=True)

# Add the llama edges to the combined dataframe
llama_df = pd.read_csv("initial_edges_llama.csv")
llama_df = llama_df.rename(columns={"X": "X", "Y": "Y"})
combined_df = pd.concat([combined_df, llama_df], ignore_index=True)

# Add the gemini edges to the combined dataframe
gemini_df = pd.read_csv("initial_edges_gemini.csv")
gemini_df = gemini_df.rename(columns={"X": "X", "Y": "Y"})
combined_df = pd.concat([combined_df, gemini_df], ignore_index=True)

# Group identical rows and output their count
combined_df = combined_df.groupby(combined_df.columns.tolist()).size().reset_index(name='count')

# Sort the combined dataframe by count in descending order and drop duplicates
combined_df = combined_df.sort_values(by='count', ascending=False)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("combined_edges.csv", index=False)
