from torch import nn
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feature_extractor as fe
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


# To import features and convert them to tensors
def get_features(path, feature):
    path = path + "/" + feature + "/"
    X_train = pd.read_csv(f"{path}x_train.csv").fillna(0).values
    y_train = pd.read_csv(f"{path}y_train.csv").fillna(0).values
    X_valid = pd.read_csv(f"{path}x_valid.csv").fillna(0).values
    y_valid = pd.read_csv(f"{path}y_valid.csv").fillna(0).values
    X_test = pd.read_csv(f"{path}x_test.csv").fillna(0).values
    y_test = pd.read_csv(f"{path}y_test.csv").fillna(0).values

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).flatten()
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.FloatTensor(y_valid).flatten()
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).flatten()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# Neural Network Model
class NN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(input_features)  # For normalization
        self.fc1 = nn.Linear(input_features, 128)  # Hidden Layer
        self.out = nn.Linear(128, 1)  # Output Layer

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.out(x))
        return x

    # To set the criterion and optimizer of the model
    def def_(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    # To train the model
    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.epochs = epochs
        self.train_losses = []
        self.valid_losses = []
        self.train_rocauc = []
        self.train_acc = []
        self.valid_acc = []

        # Initializing loss variables
        train_loss = 0
        valid_loss = 0

        # Plot for ROCs
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)

        # ------ Training and Validation ------
        for i in range(epochs):
            y_pred_train = self.forward(X_train)  # Training
            y_pred_train = y_pred_train.flatten()

            loss = self.criterion(y_pred_train, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Validation
            with torch.no_grad():
                y_pred_valid = self.forward(X_valid)
                y_pred_valid = y_pred_valid.flatten()

                loss_valid = self.criterion(y_pred_valid, y_valid)
                valid_loss += loss_valid.item()

            # Calculating train and validation loss of current epoch
            tl = train_loss / len(X_train)
            vl = valid_loss / len(X_valid)
            self.train_losses.append(tl)
            self.valid_losses.append(vl)

            # Making predictions on train and validation sets
            y_pred_train = y_pred_train.detach().numpy()
            y_pred_valid = y_pred_valid.detach().numpy()

            y_pred_train = y_pred_train.round()
            y_pred_valid = y_pred_valid.round()

            # Calculating ROC
            troc = roc_auc_score(y_train, y_pred_train)
            self.train_rocauc.append(troc)

            # Calculating Accuracy scores
            tacc = accuracy_score(y_train, y_pred_train)
            vacc = accuracy_score(y_valid, y_pred_valid)
            self.train_acc.append(tacc)
            self.valid_acc.append(vacc)

            print(
                f"Epoch {i+1} \t Train ROC: {troc} \t Train Acc: {tacc} \t Valid Acc: {vacc}"
            )

            # ROC curve
            fpr, tpr, thresholds_train = roc_curve(y_train, y_pred_train)
            plt.plot(fpr, tpr)
            plt.xlabel("FPR")
            plt.ylabel("TPR")

        return (
            self.train_rocauc,
            self.train_acc,
            self.valid_acc,
            tpr,
            fpr,
            self.train_losses,
            self.valid_losses,
        )

    # ------ Testing ------
    def test(self, X_test, y_test):
        with torch.no_grad():
            y_pred_test = self.forward(X_test)
            y_pred_test = y_pred_test.flatten().detach().numpy()

            y_pred_test = y_pred_test.round()

        self.cm = confusion_matrix(y_test, y_pred_test)
        self.test_acc = accuracy_score(y_test, y_pred_test)

        return self.cm, self.test_acc
