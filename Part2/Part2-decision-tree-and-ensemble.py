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
"""

# ---------------------------------- import sklearn.preprocessing as skp
import csv
import pandas as pd

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
            ()
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
