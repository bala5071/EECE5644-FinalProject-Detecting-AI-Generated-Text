from torch import nn
import torch
import NN_helper as helper
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

FEATURE = "TFIDF"  # Select the feautres (TF/TFIDF)

# Importing the Features
X_train, y_train, X_valid, y_valid, X_test, y_test = helper.get_features(
    path="Features", feature=FEATURE
)

# Initializing the model
model = helper.NN(X_train.shape[1])
criterion = nn.BCELoss()
epochs = 30

# Initializing hyperparameter and metrics
LRs = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ROCs = []
train_accs = []
valid_accs = []


# ------ Training and Validation ------
for i in LRs:
    optimizer = torch.optim.Adam(model.parameters(), lr=i)
    model.def_(criterion, optimizer)

    # Training the model
    (train_rocauc, train_acc, valid_acc, tpr, fpr, train_losses, valid_losses) = (
        model.train(X_train, y_train, X_valid, y_valid, epochs)
    )

    ROCs.append(np.mean(train_rocauc))
    train_accs.append(np.mean(train_acc))
    valid_accs.append(np.mean(valid_acc))

    # Saving the model
    torch.save(obj=model.state_dict(), f=f"models/Neural Network/NN_{i}.pth")

print(f"ROCs: {ROCs}")
print(f"Train F1s: {train_accs}")
print(f"Train Accs: {valid_accs}")

# Finding optimal value of LR according to validation accuracy
optiacc = max(valid_accs)
valid_accs = np.array(valid_accs)
optiInd_acc = np.where(valid_accs == optiacc)[0][0]
optiLR = LRs[optiInd_acc]

# Plotting Accuracy scores vs LR
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(LRs, train_accs)
plt.plot(LRs, valid_accs)
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.plot(optiLR, valid_accs[optiInd_acc], marker="o")
ax.legend(["Training Accuracy", "Validation Accuracy"])
ax.set_xscale("log", base=10)

print(f"Optimal Learning Rate: {optiLR}")

# ------ Testing ------
# Initializing Optimal Model
model_optimal = helper.NN(X_train.shape[1])
model_optimal.load_state_dict(torch.load(f"models/Neural Network/NN_{optiLR}.pth"))
epochs = 20

# Testing the model
cm, test_acc = model.test(X_test, y_test)

print(f"Test Acc: {test_acc}")
print(f"\nConfusion Matrix\n: {cm}")

plt.show()
