import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import time
from collections import defaultdict

INPUT_MAESTRO_DATA = "prot/MAESTRO-d6178bdd-identified_variants_merged_protein_regions-main.tsv"
variants = pd.read_csv(INPUT_MAESTRO_DATA, sep="\t", low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 20000)

def select_representative_proteins(peptide_protein_map, protein_peptide_map):
    # Step 1: Select proteins with peptides matched uniquely to those proteins
    unique_proteins = set()
    peptide_counts = defaultdict(int)
    for protein_list in peptide_protein_map.values():
        if len(protein_list) == 1:
            unique_proteins.update(protein_list)
        for protein in protein_list:
            peptide_counts[protein] += 1
    print(len(unique_proteins))
    # Step 2: Discard subset and subsumable proteins
    subset_proteins = set()
    subsumable_proteins = set()
    for protein1 in unique_proteins:
        if protein1 in subset_proteins:
            continue
        for protein2 in unique_proteins:
            if protein1 == protein2:
                continue
            if protein_peptide_map[protein2].issubset(protein_peptide_map[protein1]):
                subset_proteins.add(protein2)
    selected_proteins = unique_proteins.difference(subset_proteins)
    for protein in subset_proteins:
        for peptide in peptide_protein_map[protein]:
            peptide_protein_map.discard(protein)
            protein_peptide_map.pop(protein)
    for protein1 in selected_proteins:
        count = 0
        for peptide1 in protein_peptide_map[protein1]:
            if len(peptide_protein_map[peptide1]) > 1:
                count += 1
        if count == len(protein_peptide_map[protein1]):
            for peptide1 in protein_peptide_map[protein1]:
                peptide_protein_map[peptide1].discard(protein1)
            subsumable_proteins.add(protein1)
            protein_peptide_map.pop(protein1)
    selected_proteins = selected_proteins.difference(subsumable_proteins)

    print(len(selected_proteins))
    # Step 3: Find minimal set of proteins that matches all remaining peptides
    remaining_peptides = set(peptide_protein_map.keys())
    best_protein_covered = set()
    protein_list = []
    while remaining_peptides:
        best_protein = None
        best_protein_count = 0
        # find highest
        for protein in selected_proteins:
            if len(protein_peptide_map[protein]) > best_protein_count or best_protein is None:
                best_protein = protein
                best_protein_count = len(protein_peptide_map[protein])
        # remove peptide mapping to the protein
        for peptide in protein_peptide_map[best_protein]:
            remaining_peptides.discard(peptide)
            for protein in selected_proteins:
                if protein == best_protein:
                    continue
                if peptide in protein_peptide_map[protein]:
                    protein_peptide_map[protein].discard(peptide)

        best_protein_covered.add(best_protein)
        protein_list.append(best_protein)
        # remove the best protein from the selection set
        selected_proteins.discard(best_protein)


        subset_proteins = set()
        subsumable_proteins = set()
        # remove current subset
        for protein1 in selected_proteins:
            if protein1 in subset_proteins:
                continue
            for protein2 in selected_proteins:
                if protein1 == protein2:
                    continue
                if protein_peptide_map[protein2].issubset(protein_peptide_map[protein1]):
                    subset_proteins.add(protein2)
        selected_proteins -= subset_proteins

        for protein in subset_proteins:
            for peptide in peptide_protein_map[protein]:
                peptide_protein_map.discard(protein)
                protein_peptide_map.pop(subset_proteins)

        for protein1 in selected_proteins:
            count = 0
            for peptide1 in protein_peptide_map[protein1]:
                if len(peptide_protein_map[peptide1]) > 1:
                    count += 1
            if count == len(protein_peptide_map[protein1]):
                for peptide1 in protein_peptide_map[protein1]:
                    peptide_protein_map[peptide1].discard(protein1)
                subsumable_proteins.add(protein1)
                protein_peptide_map.pop(subsumable_proteins)
        selected_proteins = selected_proteins.difference(subsumable_proteins)
        # print(len(selected_proteins))
        if best_protein is None:
            print(len(remaining_peptides))
            break

    return best_protein_covered, protein_list

peptide = variants['Peptide'].tolist()
protein = variants['Proteins'].tolist()
peptide_protein = defaultdict(set)
assert(len(peptide) == len(protein))
for i in range(len(peptide)):
    substrings = protein[i].split(";")
    protein_list = [substring.strip() for substring in substrings]
    for j in protein_list:
        peptide_protein[peptide[i]].add(j)
# print(peptide_protein)
assert(len(peptide) == len(peptide_protein.keys()))

protein_peptide = defaultdict(set)
for peptide, protein_list in peptide_protein.items():
    for protein in protein_list:
        protein_peptide[protein].add(peptide)

representive_protein, proteins = select_representative_proteins(peptide_protein, protein_peptide)
print(len(representive_protein))
print(proteins[:10])