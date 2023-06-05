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


variants_processed = variants[['Peptide'] + [c for c in variants.columns if 'intensity_for_peptide_variant' in c]]

variants_processed.replace(0.0, np.nan, inplace=True)

variants_processed = variants_processed.set_index('Peptide')

variants_processed = variants_processed.T
# print(variants_processed)
variants_processed.index = variants_processed.index.map(lambda x: '.'.join(x.split('.')[:2]))

conditions = variants_processed.index.map(lambda x: x.split('.')[0])


# drop the rows with empty and norm
variants_processed = variants_processed.drop(['_dyn_#Empty.Empty', '_dyn_#Norm.Norm'], axis=0)
print(variants_processed.shape)

# drop columns with all NANs
variants_processed = variants_processed.dropna(axis=1, how='all')

variants_processed = variants_processed.fillna(0)
#set the condition to the first element
conditions = variants_processed.index.map(lambda x: x.split('.')[0])
# print(conditions.value_counts())
label_dict = {"_dyn_#Healthy": 0, "_dyn_#Symptomatic-non-COVID-19": 1, "_dyn_#Non-severe-COVID-19": 2, "_dyn_#Severe-COVID-19": 3}

X = variants_processed.values
y = conditions.map(label_dict).values
assert X.shape[0] == y.shape[0]

N = 10
models = []
accuracies = []

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test.shape[1])
print(X_test.shape[0])
feature_names = [i for i in range(X_test.shape[1])]
# Fitting Random Forest Classification to the Training set
model = RandomForestClassifier(n_estimators=1000, random_state = 22)
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)
print(y_pred)
print()
print(y_test)
# Making the Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.savefig("./plot/confusion_matrix.jpg")
# plt.show()
print(len(y_pred))
accuracies = [y_pred[i] == y_test[i] for i in range(len(y_pred))]
accuracy = accuracies.count(True) / len(accuracies)
print("accuracy = " + str(accuracy))
start_time = time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)
# Select the top 10 elements
top_10_elements = forest_importances.nlargest(10)
key_name = [variants_processed.columns.values[i] for i in top_10_elements.keys()]
print(key_name)
# fig, ax = plt.subplots()
# plt.bar(key_name, top_10_elements.values)
# plt.xticks(rotation=15, ha='right')
# fig.set_size_inches(16, 14)
# ax.set_xticklabels(key_name, fontsize=8.5)
# # top_10_elements.plot.bar(ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# plt.savefig("./plot/feature_importance.png")
# plt.show()


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
    for peptide, protein_list in peptide_protein_map.items():
        if len(protein_list) > 1:
            for protein in protein_list:
                if protein not in unique_proteins:
                    subset_proteins.add(protein)
        else:
            protein = next(iter(protein_list))
            if protein not in unique_proteins:
                subsumable_proteins.add(protein)

    # Step 3: Find minimal set of proteins that matches all remaining peptides using a greedy approach
    selected_proteins = unique_proteins | (subset_proteins - subsumable_proteins)
    remaining_peptides = set(peptide_protein_map.keys())
    p_list = []
    print(len(selected_proteins))
    while remaining_peptides:
        best_protein = None
        best_protein_count = 0
        best_protein_covered = set()

        for protein, peptide_list in protein_peptide_map.items():
            count = len(peptide_list.intersection(remaining_peptides))
            if count > best_protein_count:
                best_protein = protein
                best_protein_count = count
                best_protein_covered = peptide_list.intersection(remaining_peptides)

        if best_protein:
            p_list.append(best_protein)
            remaining_peptides -= best_protein_covered
        else:
            print(len(remaining_peptides))
            break

    return p_list

peptide = variants['Peptide'].tolist()
protein = variants['Proteins'].tolist()
peptide_protein = defaultdict(set)
assert(len(peptide) == len(protein))
for i in range(len(peptide)):
    if peptide[i] not in key_name:
        continue
    substrings = protein[i].split(";")
    protein_list = [substring.strip() for substring in substrings]
    for j in protein_list:
        peptide_protein[peptide[i]].add(j)
# print(peptide_protein)
assert(10 == len(peptide_protein.keys()))

protein_peptide = defaultdict(set)
for peptide, protein_list in peptide_protein.items():
    for protein in protein_list:
        protein_peptide[protein].add(peptide)

representive_protein = select_representative_proteins(peptide_protein, protein_peptide)
# print("here")
print(representive_protein)