"""
Experiment D: Hyper-Parameter Tuning

Train a MLP on MNIST, and evaluate it at various regularizations.
"""
import argparse
import datetime
import logging
import os
import pickle
import sys
import time

import lib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import mlp

EXPNAME = "classifier"

#
# Init Logging
#
ofolder = os.path.join(
    "output", EXPNAME, datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
)
os.makedirs(ofolder, exist_ok=True)
TARGETS = logging.StreamHandler(sys.stdout), logging.FileHandler(
    os.path.join(ofolder, "transcript.log")
)
logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s | %(name)14s | %(levelname)7s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=TARGETS)
logging.getLogger("matplotlib").setLevel(logging.WARN)


#
# Config
#
conf_level = 0.1
figsize = (3, 2)  # good for single-column IEEE
plt.rcParams.update({"font.size": 8, "legend.fontsize": 8})
loss_draw_lims = [0, 100]  # extend the plots to be valid over this range
selected_alphas = 1 - np.array([0.9, 0.95, 0.99])
batch_size = 1024
lamdas = np.logspace(-6, -2, num=12)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_cache",
    action="store_true",
    help=(
        "If set, the MLP will not be trained, but the last trained models are used. "
        "Beware that you must remember to NOT use the cache if you change any training settings... There is no validation that "
        "the models picked up from disk corresponds to the config you asked for."
    ),
)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()
logger.info(f"{args}")

#
# Generate Data
#
X_train, X_test1, y_train, y_test1 = lib.mnist()


#
# Train/Load models. Compute losses.
#
ress = []
losslists = []

for run, lamda in enumerate(lamdas):

    #
    # Train/Load model and compute losses
    #
    if args.use_cache:
        model = pickle.load(open(f"model_{run}.pkl", "rb"))
    else:
        model = mlp.MLP(
            X_train,
            y_train,
            seed=args.seed,
            batch_size=batch_size,
            learning_rate=0.01,
            epochs=100,
            weight_decay=lamda,
        )
        pickle.dump(model, open(f"model_{run}.pkl", "wb"))

    log_likliehood_of_data = model.predict_log_proba(X_test1)
    if not np.all(np.isfinite(log_likliehood_of_data)):
        print(log_likliehood_of_data)
        raise ValueError(
            f"At regularization {lamda}, the model predicts zeroes. See above. Not ok. Increase regularization!"
        )
    loss = np.take_along_axis(
        -log_likliehood_of_data, y_test1[:, np.newaxis], 1
    ).squeeze()

    #
    # Compute the cutoffs at the selected alphas
    #
    ndata = len(loss)
    cutoffs = np.zeros(len(selected_alphas))
    ks = np.ceil((ndata + 1) * (1 - selected_alphas)).astype(int)
    cutoffs[ks < ndata] = np.sort(loss)[
        (ks - 1)[ks < ndata]
    ]  # n.b. python is zero-indexed
    cutoffs[ks >= ndata] = np.inf
    for c, a in zip(cutoffs, selected_alphas):
        ress.append({"alpha": a, "lbar": c, "lambda": lamda})

    losslists.append(loss)

logger.info(f"Computed the results")


#
# Produce plots
#
df = pd.DataFrame(ress)
fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=(3, 3 * 2))
ax0.violinplot(losslists)
ax0.set_ylim([0, 1])
ax0.axhline(-np.log(0.5), color="red", alpha=0.2, label="Misclassification limit")
ax0.legend()
ax0.set_title("Test losses")
acc = [(np.exp(-loss) > 0.5).mean() for loss in losslists]
ax1.scatter(lamdas, acc)
ax1.set_xscale("log")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax1.set_title("Test accuracy")
risk = [loss.mean() for loss in losslists]
ax2.scatter(lamdas, risk)
ax2.set_xscale("log")
ax2.set_title("risk")
plt.tight_layout()
fig.savefig(os.path.join(ofolder, f"{EXPNAME}-details.pdf"))

fig, ax = plt.subplots(figsize=figsize)
n_ls = df["alpha"].unique().__len__()
norm = mcolors.LogNorm(vmin=df["alpha"].min(), vmax=df["alpha"].max())
cmap = mcolors.LinearSegmentedColormap.from_list("funk", ["C0", "C1"])

for alpha in np.sort(df["alpha"].unique()):
    lambdas = df[df["alpha"] == alpha]["lambda"]
    cutoffs = df[df["alpha"] == alpha]["lbar"]
    ax.scatter(lambdas, cutoffs, label=f"{alpha:4.0%}", color=cmap(norm(alpha)))
    ax.plot(lambdas, cutoffs, color=cmap(norm(alpha)), lw=1, ls="solid")

ax.set_xlabel("$\\lambda$")
ax.set_ylabel("$\\bar{\ell}_{\\alpha}(\\mathcal{D};\\lambda)$")
ax.legend(title="$\\alpha$")
ax.set_xscale("log")
plt.tight_layout()
fig.savefig(os.path.join(ofolder, f"{EXPNAME}-reg.pdf"))


#
# Finalize
#
logger.info(f"Saved result plots")
plt.show()
