import copy

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader
import numpy as np
import networkx as  nx
from cdt.metrics import SID
from cdt.metrics import SHD as shd

expert_bn = BayesianNetwork([
    ('HighVIS', "LowMAP"),
    ("LowMAP", "HighLactate"),
    ("HighLactate", "LowPH"),
    ("HighLactate", "NeurologicalInjury"),
    ("LowPH", "NeurologicalInjury"),
    ("RelativePCO2", "LowPH"),
    ("LowPlatelet", "NeurologicalInjury"),
    ("HighMAP", "NeurologicalInjury")
    ])

data_bn = BayesianNetwork([
    ("LowMAP", "HighVIS"),
    ("LowMAP", "HighLactate"),
    ("HighLactate", "LowPlatelet"),
    ("LowPH", "RelativePCO2"),
    ("NeurologicalInjury", "RelativePCO2"),
])

pc_bn = BayesianNetwork(
    [('HighVIS', 'LowMAP')]
)
data_bn.add_nodes_from(expert_bn.nodes())
pc_bn.add_nodes_from(expert_bn.nodes())


# Compute the SHD between the expert and the data BN
print("SHD between expert and data", shd(copy.deepcopy(expert_bn), data_bn, double_for_anticausal=False))
print("SID between expert and data", SID(copy.deepcopy(expert_bn), data_bn))
print("Spurious edges between expert and data", len(set(data_bn.edges()).difference(set(expert_bn.edges()))))

# Compute the SHD between the expert and the pc BN
print("SHD between expert and pc", shd(copy.deepcopy(expert_bn), pc_bn, double_for_anticausal=False))
print("SID between expert and pc", SID(copy.deepcopy(expert_bn), pc_bn))
print("Spurious edges between expert and pc", len(set(pc_bn.edges()).difference(set(expert_bn.edges()))))

# We would get the initial LLM based BN's by loading from the files
llm_gpt = pd.read_csv("initial_edges_gpt4o.csv")
llm_gpt_bn = BayesianNetwork(
    list(zip(llm_gpt["X"].tolist(), llm_gpt["Y"].tolist()))
)

llm_llama = pd.read_csv("initial_edges_llama.csv")
llm_llama_bn = BayesianNetwork(
    list(zip(llm_llama["X"].tolist(), llm_llama["Y"].tolist()))
)
llm_gemini = pd.read_csv("initial_edges_gemini.csv")
llm_gemini_bn = BayesianNetwork(
    list(zip(llm_gemini["X"].tolist(), llm_gemini["Y"].tolist()))
)
llm_deepseek = pd.read_csv("initial_edges_deepseek.csv")
llm_deepseek_bn = BayesianNetwork(
    list(zip(llm_deepseek["X"].tolist(), llm_deepseek["Y"].tolist()))
)
llm_combined = pd.read_csv("initial_edges_combined.csv")
llm_combined_bn = BayesianNetwork(
    list(zip(llm_combined["X"].tolist(), llm_combined["Y"].tolist()))
)

# We would get the refined BN's by loading from the files
print("SHD between expert and gpt4o", shd(copy.deepcopy(expert_bn), llm_gpt_bn, double_for_anticausal=False))
print("SHD between expert and llama", shd(copy.deepcopy(expert_bn), llm_llama_bn , double_for_anticausal=False))
print("SHD between expert and gemini", shd(copy.deepcopy(expert_bn), llm_gemini_bn, double_for_anticausal=False))
print("SHD between expert and deepseek", shd(copy.deepcopy(expert_bn), llm_deepseek_bn, double_for_anticausal=False))
print("SHD between expert and combined", shd(copy.deepcopy(expert_bn), llm_combined_bn, double_for_anticausal=False))
print("SID between expert and gpt4o", SID(copy.deepcopy(expert_bn), llm_gpt_bn))
print("SID between expert and llama", SID(copy.deepcopy(expert_bn), llm_llama_bn))
print("SID between expert and gemini", SID(copy.deepcopy(expert_bn), llm_gemini_bn))
print("SID between expert and deepseek", SID(copy.deepcopy(expert_bn), llm_deepseek_bn))
print("SID between expert and combined", SID(copy.deepcopy(expert_bn), llm_combined_bn))

refined_gpt = BIFReader("results/results_twice_refined_sub_first/post_pelican_gpt4o-2.bif").get_model()
refined_llama = BIFReader("results/results_twice_refined_sub_first/post_pelican_llama-2.bif").get_model()
refined_gemini = BIFReader("results/results_twice_refined_sub_first/post_pelican_gemini-2.bif").get_model()
refined_deepseek = BIFReader("results/results_twice_refined_sub_first/post_pelican_deepseek-2.bif").get_model()
refined_combined = BIFReader("results/results_twice_refined_sub_first/post_pelican_combined-2.bif").get_model()



# We would compute the shd
shd_expert_refined_gpt = shd(copy.deepcopy(expert_bn), refined_gpt , double_for_anticausal=False)
sid_expert_refined_gpt = SID(copy.deepcopy(expert_bn), refined_gpt)
print("SHD between expert and refined gpt4o", shd_expert_refined_gpt)
print("SID between expert and refined gpt4o", sid_expert_refined_gpt)
shd_expert_refined_deepseek = shd(copy.deepcopy(expert_bn), refined_deepseek, double_for_anticausal=False)
sid_expert_refined_deepseek = SID(copy.deepcopy(expert_bn), refined_deepseek)
print("SHD between expert and refined deepseek", shd_expert_refined_deepseek)
print("SID between expert and refined deepseek", sid_expert_refined_deepseek)
shd_expert_refined_gemini = shd(copy.deepcopy(expert_bn), refined_gemini, double_for_anticausal=False)
sid_expert_refined_gemini = SID(copy.deepcopy(expert_bn), refined_gemini)
print("SHD between expert and refined gemini", shd_expert_refined_gemini)
print("SID between expert and refined gemini", sid_expert_refined_gemini)
shd_expert_refined_llama = shd(copy.deepcopy(expert_bn), refined_llama, double_for_anticausal=False)
sid_expert_refined_llama = SID(copy.deepcopy(expert_bn), refined_llama)
print("SHD between expert and refined llama", shd_expert_refined_llama)
print("SID between expert and refined llama", sid_expert_refined_llama)
shd_expert_refined_combined = shd(copy.deepcopy(expert_bn), refined_combined , double_for_anticausal=False)
sid_expert_refined_combined = SID(copy.deepcopy(expert_bn), refined_combined)
print("SHD between expert and refined combined", shd_expert_refined_combined)
print("SID between expert and refined combined", sid_expert_refined_combined)


# We would also count the number of edges which match between the expert and the llm
expert_edges = set(expert_bn.edges())
llm_gpt_edges = set(llm_gpt_bn.edges())
llm_llama_edges = set(llm_llama_bn.edges())
llm_gemini_edges = set(llm_gemini_bn.edges())
llm_deepseek_edges = set(llm_deepseek_bn.edges())
llm_combined_edges = set(llm_combined_bn.edges())
refined_gpt_edges = set(refined_gpt.edges())
refined_llama_edges = set(refined_llama.edges())
refined_gemini_edges = set(refined_gemini.edges())
refined_deepseek_edges = set(refined_deepseek.edges())
refined_combined_edges = set(refined_combined.edges())

# We would compute the spurious edges by perfornin a set difference between the llm and the expert
spurious_gpt = llm_gpt_edges.difference(expert_edges)
spurious_llama = llm_llama_edges.difference(expert_edges)
spurious_gemini = llm_gemini_edges.difference(expert_edges)
spurious_deepseek = llm_deepseek_edges.difference(expert_edges)
spurious_combined = llm_combined_edges.difference(expert_edges)
print("Spurious edges in gpt4o", len(spurious_gpt))
print("Spurious edges in llama", len(spurious_llama))
print("Spurious edges in gemini", len(spurious_gemini))
print("Spurious edges in deepseek", len(spurious_deepseek))
print("Spurious edges in combined", len(spurious_combined))
print("Spurious edges in refined gpt4o", len(refined_gpt_edges.difference(expert_edges)))
print("Spurious edges in refined llama", len(refined_llama_edges.difference(expert_edges)))
print("Spurious edges in refined gemini", len(refined_gemini_edges.difference(expert_edges)))
print("Spurious edges in refined deepseek", len(refined_deepseek_edges.difference(expert_edges)))
print("Spurious edges in refined combined", len(refined_combined_edges.difference(expert_edges)))
