# Description: Evaluation of different classifer types from SKLEARN library
# Created on: 21/11/24
# --------------------------------------------------
from sklearn import datasets as skd, neighbors
from sklearn import model_selection as skms
from sklearn import ensemble as ske
from sklearn import neighbors as skn
from sklearn import tree as skt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# load data
X, y = skd.load_digits(return_X_y=True)


# seed constant for whole file
SEED = 26

# split data
X_train, X_test, y_train, y_test = skms.train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

# split data into subsets
splits = 5
kf = skms.KFold(n_splits=splits, shuffle=False)


# randomForest


randomForestTable = [
    # [estimators, depth, maxNodes, minSamples, avg accuracy]
    [50, 5, 5, 5, 0.0],
    [150, 12, 12, 12, 0.0],
    [300, 20, 20, 20, 0.0],
]


# function to caclulate average accuracy of random forest classifier with given params
def randomForestEval(estimators, depth, maxNodes, minSamples):
    predictions = []
    rfc = ske.RandomForestClassifier(
        n_estimators=estimators,
        max_depth=depth,
        max_leaf_nodes=maxNodes,
        min_samples_split=minSamples,
        random_state=SEED,
    )
    # cross eval
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        rfc.fit(X_train, y_train)

        # Test the model
        pred = rfc.predict(X_test)
        predictions.append(accuracy_score(y_test, pred))

    # return mean
    return sum(predictions) / 5


# testing randomForest
randomForestTable[0][4] = randomForestEval(
    randomForestTable[0][0],
    randomForestTable[0][1],
    randomForestTable[0][2],
    randomForestTable[0][3],
)
randomForestTable[1][4] = randomForestEval(
    randomForestTable[1][0],
    randomForestTable[1][1],
    randomForestTable[1][2],
    randomForestTable[1][3],
)
randomForestTable[2][4] = randomForestEval(
    randomForestTable[2][0],
    randomForestTable[2][1],
    randomForestTable[2][2],
    randomForestTable[2][3],
)

# Find highest accuracy and retest with full data
finalRF = 0
if randomForestTable[0][4] < randomForestTable[1][4]:
    if randomForestTable[1][4] < randomForestTable[2][4]:
        rfc = ske.RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            max_leaf_nodes=20,
            min_samples_split=20,
            random_state=SEED,
        )

        rfc.fit(X_train, y_train)
        finalRF = accuracy_score(y_test, rfc.predict(X_test))
    else:
        rfc = ske.RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            max_leaf_nodes=12,
            min_samples_split=12,
            random_state=SEED,
        )

        rfc.fit(X_train, y_train)
        finalRF = accuracy_score(y_test, rfc.predict(X_test))
else:
    rfc = ske.RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        max_leaf_nodes=5,
        min_samples_split=5,
        random_state=SEED,
    )

    rfc.fit(X_train, y_train)
    finalRF = accuracy_score(y_test, rfc.predict(X_test))


# print scores
print("\n randomForest")
print(f"Params: {randomForestTable[0][0:4]}, Scored: {randomForestTable[0][4]}")
print(f"Params: {randomForestTable[1][0:4]}, Scored: {randomForestTable[1][4]}")
print(f"Params: {randomForestTable[2][0:4]}, Scored: {randomForestTable[2][4]}")
print(f"Final result: {finalRF}")


# KNN

KNNtable = [
    # [neighbours, leafsize, p, avg accuracy]
    [5, 30, 2, 0.0],
    [15, 50, 5, 0.0],
    [30, 100, 12, 0.0],
]


# function to calculate KNN accuarcy with given paramaters
def KNNEval(neighbours, leafsize, p):
    predictions = []
    knc = skn.KNeighborsClassifier(
        n_neighbors=neighbours,
        leaf_size=leafsize,
        p=p,
    )
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        knc.fit(X_train, y_train)

        # Test the model
        pred = knc.predict(X_test)
        predictions.append(accuracy_score(y_test, pred))
    return sum(predictions) / 5


# testing randomForest
KNNtable[0][3] = KNNEval(
    KNNtable[0][0],
    KNNtable[0][1],
    KNNtable[0][2],
)
KNNtable[1][3] = KNNEval(
    KNNtable[1][0],
    KNNtable[1][1],
    KNNtable[1][2],
)
KNNtable[2][3] = KNNEval(
    KNNtable[2][0],
    KNNtable[2][1],
    KNNtable[2][2],
)

# find highest accuracy and retest with full data sample
finalKNN = 0
if KNNtable[0][3] < KNNtable[1][3]:
    if KNNtable[1][3] < KNNtable[2][3]:
        knc = skn.KNeighborsClassifier(
            n_neighbors=KNNtable[2][0],
            leaf_size=KNNtable[2][1],
            p=KNNtable[2][2],
        )

        knc.fit(X_train, y_train)
        finalKNN = accuracy_score(y_test, knc.predict(X_test))
    else:
        knc = skn.KNeighborsClassifier(
            n_neighbors=KNNtable[1][0],
            leaf_size=KNNtable[1][1],
            p=KNNtable[1][2],
        )

        knc.fit(X_train, y_train)
        finalKNN = accuracy_score(y_test, knc.predict(X_test))
else:
    knc = skn.KNeighborsClassifier(
        n_neighbors=KNNtable[0][0],
        leaf_size=KNNtable[0][1],
        p=KNNtable[0][2],
    )

    knc.fit(X_train, y_train)
    finalKNN = accuracy_score(y_test, knc.predict(X_test))


# print scores
print("\n KNN")
print(f"Params: {KNNtable[0][0:3]}, Scored: {KNNtable[0][3]}")
print(f"Params: {KNNtable[1][0:3]}, Scored: {KNNtable[1][3]}")
print(f"Params: {KNNtable[2][0:3]}, Scored: {KNNtable[2][3]}")
print(f"Final result: {finalKNN}")

# Decision Tree


decisionTreeTable = [
    # [depth, minSampleSplit, maxNodes, minSamplesLeaf, avg accuracy]
    [5, 2, 5, 5, 0.0],
    [12, 4, 12, 12, 0.0],
    [25, 8, 25, 20, 0.0],
]


# function to calculate average accuracy of decision tree using given params
def decisionTreeEval(depth, minSampleSplit, maxNodes, minSampleLeaf):
    predictions = []
    dtc = skt.DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=minSampleSplit,
        max_leaf_nodes=maxNodes,
        min_samples_leaf=minSampleLeaf,
        random_state=SEED,
    )
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        dtc.fit(X_train, y_train)

        # Test the model
        pred = dtc.predict(X_test)
        predictions.append(accuracy_score(y_test, pred))
    return sum(predictions) / 5


# testing randomForest
decisionTreeTable[0][4] = decisionTreeEval(
    decisionTreeTable[0][0],
    decisionTreeTable[0][1],
    decisionTreeTable[0][2],
    decisionTreeTable[0][3],
)
decisionTreeTable[1][4] = decisionTreeEval(
    decisionTreeTable[1][0],
    decisionTreeTable[1][1],
    decisionTreeTable[1][2],
    decisionTreeTable[1][3],
)
decisionTreeTable[2][4] = decisionTreeEval(
    decisionTreeTable[2][0],
    decisionTreeTable[2][1],
    decisionTreeTable[2][2],
    decisionTreeTable[2][3],
)

# find highest score and retrain with full data sample
finalDT = 0
if decisionTreeTable[0][4] < decisionTreeTable[1][4]:
    if decisionTreeTable[1][4] < decisionTreeTable[2][4]:
        dtc = skt.DecisionTreeClassifier(
            max_depth=decisionTreeTable[2][0],
            min_samples_split=decisionTreeTable[2][1],
            max_leaf_nodes=decisionTreeTable[2][2],
            min_samples_leaf=decisionTreeTable[2][3],
            random_state=SEED,
        )
        dtc.fit(X_train, y_train)
        finalDT = accuracy_score(y_test, dtc.predict(X_test))
    else:
        dtc = skt.DecisionTreeClassifier(
            max_depth=decisionTreeTable[2][0],
            min_samples_split=decisionTreeTable[2][1],
            max_leaf_nodes=decisionTreeTable[2][2],
            min_samples_leaf=decisionTreeTable[2][3],
            random_state=SEED,
        )
        dtc.fit(X_train, y_train)
        finalDT = accuracy_score(y_test, dtc.predict(X_test))

else:
    dtc = skt.DecisionTreeClassifier(
        max_depth=decisionTreeTable[2][0],
        min_samples_split=decisionTreeTable[2][1],
        max_leaf_nodes=decisionTreeTable[2][2],
        min_samples_leaf=decisionTreeTable[2][3],
        random_state=SEED,
    )
    dtc.fit(X_train, y_train)
    finalDT = accuracy_score(y_test, dtc.predict(X_test))


# print scores
print("\n Decision Tree")
print(f"Params: {decisionTreeTable[0][0:4]}, Scored: {decisionTreeTable[0][4]}")
print(f"Params: {decisionTreeTable[1][0:4]}, Scored: {decisionTreeTable[1][4]}")
print(f"Params: {decisionTreeTable[2][0:4]}, Scored: {decisionTreeTable[2][4]}")
print(f"Final result: {finalDT}")
