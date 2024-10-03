import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle

FEATURE = "TFIDF"  # Select the feautres (TF/TFIDF)

# Importing the Features
path = "Features/" + FEATURE + "/"
X_train = pd.read_csv(f"{path}x_train.csv").fillna(0).values
y_train = pd.read_csv(f"{path}y_train.csv").fillna(0).values.flatten()
X_valid = pd.read_csv(f"{path}x_valid.csv").fillna(0).values
y_valid = pd.read_csv(f"{path}y_valid.csv").fillna(0).values.flatten()
X_test = pd.read_csv(f"{path}x_test.csv").fillna(0).values
y_test = pd.read_csv(f"{path}y_test.csv").fillna(0).values.flatten()

# Normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

# Initializing hyperparameter and metrics
ns = [1, 10, 25, 50, 75, 100, 125, 150, 200]
ROCs = []  # ROC AUC
train_f1s = []  # F1 scores
valid_f1s = []  # F1 scores
train_accs = []  # Accuracy scores
valid_accs = []  # Accuracy scores


fig1 = plt.figure()  # Plot for ROCs
ax1 = fig1.add_subplot(111)

# ------ Training and Validation ------
for i in ns:
    classifier = RandomForestClassifier(
        random_state=42, n_estimators=i, class_weight={0: 0.8, 1: 0.2}
    )
    classifier.fit(X_train, y_train)  # Training

    y_pred_train = classifier.predict(X_train)
    y_pred_valid = classifier.predict(X_valid)  # Validation

    # F1 scores
    train_f1 = f1_score(y_train, y_pred_train)
    valid_f1 = f1_score(y_valid, y_pred_valid)

    train_f1s.append(train_f1)
    valid_f1s.append(valid_f1)

    # Accuracy scores
    train_acc = accuracy_score(y_train, y_pred_train)
    valid_acc = accuracy_score(y_valid, y_pred_valid)

    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

    # ROC AUCs
    rocauc = metrics.roc_auc_score(y_train, y_pred_train)
    ROCs.append(rocauc)

    # ROC curves
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)

    # Saving the models
    with open(
        f"models/Random Forest/{FEATURE}/Random_Forest_{FEATURE}_{i}.pkl", "wb"
    ) as f:
        clf = pickle.dump(classifier, f)

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
ax1.legend(ns)

print(f"ROCs: {ROCs}\n")
print(f"Train F1s: {train_f1s}")
print(f"Train Accs: {train_accs}\n")
print(f"Valid F1s: {valid_f1s}")
print(f"Valid Accs: {valid_accs}\n")

# Finding optimal value of n according to F1 scores
optiF1 = max(valid_f1s)
valid_f1s = np.array(valid_f1s)
optiInd_f1 = np.where(valid_f1s == optiF1)[0][0]
optin_f1 = ns[optiInd_f1]

# Plotting F1 scores vs n
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.plot(ns, train_f1s)
plt.plot(ns, valid_f1s)
plt.xlabel("n")
plt.ylabel("F1 Scores")
plt.plot(optin_f1, valid_f1s[optiInd_f1], marker="o")
ax2.legend(["Training F1", "Validation F1"])

# Finding optimal value of n according to accuracy score
optiacc = max(valid_accs)
valid_accs = np.array(valid_accs)
optiInd_acc = np.where(valid_accs == optiacc)[0][0]
optin_acc = ns[optiInd_acc]

# Plotting Accuracy scores vs n
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
plt.plot(ns, train_accs)
plt.plot(ns, valid_accs)
plt.xlabel("n")
plt.ylabel("Accuracy")
plt.plot(optin_acc, valid_accs[optiInd_acc], marker="o")
ax3.legend(["Training Accuracy", "Validation Accuracy"])

print(f"Optimal C: {optin_f1}")

# ------ Testing ------
# Loading optimal model
with open(
    f"models/Random Forest/{FEATURE}/Random_Forest_{FEATURE}_{optin_f1}.pkl",
    "rb",
) as f:
    classifier_optimal = pickle.load(f)

# Testing the model
y_pred_test = classifier.predict(X_test)

# Calculating test metrics
test_f1 = f1_score(y_test, y_pred_test)
test_acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)

print(f"Test F1: {test_f1}")
print(f"Test Acc: {test_acc}")
print(f"\nConfusion Matrix\n: {cm}")

plt.show()
