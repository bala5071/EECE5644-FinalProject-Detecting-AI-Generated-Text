import numpy as np
import matplotlib.pyplot as plt

metrics_tf = [
    [0.8796, 0.9998, 0.9943],
    [0.9044, 0.9996, 0.9943],
    [0.7486, 0.8970, 0.8001],
    [0.7496, 0.8996, 0.8248],
]

metrics_tfidf = [
    [0.8844, 0.9994, 0.9988],
    [0.9086, 0.9996, 0.9995],
    [0.7509, 0.8994, 0.8148],
    [0.7541, 0.9006, 0.8336],
]


fig, ax = plt.subplots()
axs = []
ax1 = plt.plot(
    ["ROC", "Train Acc", "Valid Acc", "Test Acc"],
    metrics_tfidf,
    marker=".",
)
ax2 = plt.plot(
    ["ROC", "Train Acc", "Valid Acc", "Test Acc"],
    metrics_tf,
    marker=".",
    color="gray",
    linestyle="--",
)
axs.append(ax1)
axs.append(ax2)
l1 = plt.legend(["LR", "RF", "NN"], loc=1)
plt.legend([i[0] for i in axs], ["TFIDF", "TF"], loc=3)
plt.gca().add_artist(l1)
plt.grid(True, linestyle="--")
ax.yaxis.grid(False, which="major")
plt.show()
