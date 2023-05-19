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

INPUT_MAESTRO_DATA = "prot/MAESTRO-d6178bdd-identified_variants_merged_protein_regions-main.tsv"
variants = pd.read_csv(INPUT_MAESTRO_DATA, sep="\t", low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 20000)

# count = 1
# for col in variants.columns.values:
#     print(col)
#     count += 1
#     if count % 5 == 0:
#         print("")
# print(variants)
variants_processed = variants[['Peptide'] + [c for c in variants.columns if 'intensity_for_peptide_variant' in c]]

variants_processed.replace(0.0, np.nan, inplace=True)

variants_processed = variants_processed.set_index('Peptide')

variants_processed = variants_processed.T
# print(variants_processed)
variants_processed.index = variants_processed.index.map(lambda x: '.'.join(x.split('.')[:2]))

conditions = variants_processed.index.map(lambda x: x.split('.')[0])
print(conditions.value_counts())

# print(variants_processed['Condition'])
# variants_processed = variants_processed[(variants_processed.Condition == "_dyn_#Severe-COVID-19")
#                                         | (variants_processed.Condition == "_dyn_#Non-severe-COVID-19")]


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
plt.savefig("./plot/confusion_matrix.jpg")
plt.show()
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
fig, ax = plt.subplots()
plt.bar(key_name, top_10_elements.values)
plt.xticks(rotation=15, ha='right')
fig.set_size_inches(16, 14)
ax.set_xticklabels(key_name, fontsize=8.5)
# top_10_elements.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
plt.savefig("./plot/feature_importance.png")
plt.show()