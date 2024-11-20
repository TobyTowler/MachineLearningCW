# -----------------------------------
"""
Number of cases: 274
Number of attributes: 9
Number of classes: 2
Class distribution (cases per class):
 positive: 79,
 negative : 195
List attributes and different values:
Attrbute 0: 6
Attrbute 1: 3
Attrbute 2: 11
Attrbute 3: 7
Attrbute 4: 2
Attrbute 5: 3
Attrbute 6: 2
Attrbute 7: 5
Attrbute 8: 2

DecisionTreeClassifier
entropy
5
26
accuracyy_score(on training set):  0.8310502283105022
accuracyy_score(on testing set):  0.7454545454545455

Ensemble
accuracy_score(on test set): 0.7818181818181819
"""

# ----------------------------------
import csv
from typing import final
import pandas as pd
from sklearn import preprocessing as skp
from sklearn import model_selection as skms
from sklearn import tree as skt
from sklearn import metrics as skm
from matplotlib import pyplot as plt
import numpy as np

data = []
with open("breast-cancer.data") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:  # each row is a list
        data.append(row)

X = []
y = []
for i in data:
    if "?" in i:
        data.remove(i)
    else:
        y.append(i.pop(0))
        X.append(i)


XColumns = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]
yColumns = ["Class"]

print(X)
X = pd.DataFrame.from_records(X, columns=XColumns)
oEnc = skp.OrdinalEncoder()
X = oEnc.fit_transform(X)

y = pd.DataFrame(y, columns=yColumns)
lEnc = skp.LabelEncoder()
y = lEnc.fit_transform(y)
print("X after transofrm \n", X)

# print("\n\nY", y)
print(f"Number of cases: {len(X)}")
print(f"Number of attributes: {len(X[0])}")
print(f"Number of classes: {len(set(y))}")
posCount = 0
negCount = 0
for i in y:
    if i == 0:
        negCount += 1
    else:
        posCount += 1

print(
    f"Class distribution (cases per class): \n positive: {posCount}, \n negative : {negCount}"
)
print("List attributes and different values: ")
map = [set(), set(), set(), set(), set(), set(), set(), set(), set()]
for j in X:
    for i in range(len(j)):
        if j[i] in map[i]:
            continue
        else:
            map[i].add(j[i])


print(f"Attrbute 0: {len(map[0])}")
print(f"Attrbute 1: {len(map[1])}")
print(f"Attrbute 2: {len(map[2])}")
print(f"Attrbute 3: {len(map[3])}")
print(f"Attrbute 4: {len(map[4])}")
print(f"Attrbute 5: {len(map[5])}")
print(f"Attrbute 6: {len(map[6])}")
print(f"Attrbute 7: {len(map[7])}")
print(f"Attrbute 8: {len(map[8])}")

SEED = 26
X_train, X_test, y_train, y_test = skms.train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

dt = skt.DecisionTreeClassifier(criterion="entropy", max_depth=5)
dt.fit(X_train, y_train)
skt.plot_tree(dt, node_ids=True)
# plt.show()

yTrainPred = dt.predict(X_train)
yAccuracyTrain = skm.accuracy_score(y_train, yTrainPred)

yTestPred = dt.predict(X_test)
yAccuracyTest = skm.accuracy_score(y_test, yTestPred)


print("\n'\nDecisionTreeClassifier")
print(dt.criterion)
print(dt.max_depth)
print(26)
print(f"accuracyy_score(on training set):  {yAccuracyTrain}")
print(f"accuracyy_score(on testing set):  {yAccuracyTest}")


# 2C
trees = []
for i in range(1, 10):
    tree = skt.DecisionTreeClassifier(
        criterion="gini", max_depth=i, max_features=9, random_state=SEED
    )
    trees.append(tree)


for i in trees:
    i.fit(X_train, y_train)

predictions = []
for i in trees:
    yPred = i.predict(X_test)
    predictions.append(yPred)


def majVote(predictions):
    finalPredictions = []
    for i in range(len(predictions[0])):
        true = 0
        false = 0
        for j in range(0, 9):
            if predictions[j][i] == 1:
                true += 1
            else:
                false += 1

        if true > false:
            finalPredictions.append(1)
        else:
            finalPredictions.append(0)
    return finalPredictions


finalPredictions = majVote(predictions)
# print(finalPredictions)

for i in range(0, len(finalPredictions)):
    print(f"case_id: {i}")
    for j in range(0, 9):
        print(f"dt{j}: {predictions[j][i]}")
    print(f"actual_class: {y_test[i]}")
    print(f"ensemble: {finalPredictions[i]}")

ensembleAccuracy = skm.accuracy_score(y_test, finalPredictions)
print(f"accuracy_score(on test set): {ensembleAccuracy}")
# dawd
