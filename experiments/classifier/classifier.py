"""
Compare the interval length and the empirical coverage rate,
for a different given confidence levels
"""


import argparse
import datetime
import json
import logging
import os
import sys
import pathlib

import lib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sklearn.compose
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils


#
# Init logging
#
ofolder = pathlib.Path(__file__).parent / "output"
os.makedirs(ofolder, exist_ok=True)

logger = logging.getLogger(__name__)
TARGETS = (
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(os.path.join(ofolder, "transcript.log")),
)
FORMAT = "%(asctime)s | %(name)14s | %(levelname)7s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=TARGETS)
logging.getLogger("matplotlib").setLevel(logging.WARN)


#
# Config & Initialization
#
plt.rcParams.update(
    {
        "font.size": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3, 2),
        "figure.autolayout": True,
    }
)
parser = argparse.ArgumentParser()
parser.add_argument("--confidence_level", type=float, default=0.1)
parser.add_argument("--ndata", type=int, default=50)
parser.add_argument("--ndata_model", type=int, default=150)
parser.add_argument("--seed", type=int, default=123456789)
parser.add_argument("--loss_plot_lims", type=float, nargs=2, default=(0, 1.0))
parser.add_argument("--loss_level_plot_lims", type=float, nargs=2, default=(0, 1.0))
opts = parser.parse_args()
ss = np.random.SeedSequence(opts.seed)
rng = np.random.default_rng(ss.generate_state(1)[0])
logger.info(f"Config: {opts}")
with open(os.path.join(ofolder, "config.json"), "w") as f:
    json.dump(
        {"ofolder": str(ofolder), "args": vars(opts), "argv": sys.argv,}, f, indent=2,
    )


def brier_loss(model, X, y):
    """The loss function of the Brier score
    Assumes classes are coded 0,1,2 etc
    """
    y_onehot = np.zeros((len(y), y.max() + 1))
    y_onehot[np.arange(len(y)), y] = 1
    return ((model.predict_proba(X) - y_onehot) ** 2).sum(axis=1)


def loss_fn(model, X, y):
    """Complementary likliehood
    if P(y|x) is the model predicted probability of observing y given x, then
    this function returns
        1-P(y|x)
    """
    fXs = model.predict_proba(X)
    pred_proba_for_true_class = fXs[np.arange(len(X)), y]
    return 1 - pred_proba_for_true_class


#
# Load and Split data
#
df = lib.palmer_penguins().dropna()
y = df.iloc[:, 0]
label_encoder = sklearn.preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)
X = df.iloc[:, 1:]
X_model, X_test, y_model, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=opts.ndata_model, random_state=ss.generate_state(1)[0]
)
logger.info(
    f"There are {len(y)} samples in the penguin dataset after missing data is dropped"
)
logger.info(
    f"There are {len(X_model)} samples used in model training. {len(X_test)} is used in testing"
)
logger.info(f"The unique islands in the data are {df['Island'].unique().to_numpy()}")

#
# Train model
#
model = sklearn.pipeline.make_pipeline(
    sklearn.compose.make_column_transformer(
        (sklearn.preprocessing.OneHotEncoder(), ["Island", "Sex"]),
        remainder="passthrough",
    ),
    sklearn.preprocessing.FunctionTransformer(func=lambda X: X.astype(float)),
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LogisticRegressionCV(),
)

model.fit(X_model, y_model)


#
# Compute and plot an in-distribution curve
#
fig, ax = plt.subplots()

# Compute level-alpha-losses, and plot
idx = rng.choice(np.arange(len(X_test)), size=opts.ndata, replace=False)
X_data, y_data = X_test.iloc[idx], y_test[idx]
loss = loss_fn(model, X_data, y_data)

alphas = 1 - (np.arange(len(loss)) + 1) / (len(loss) + 1)
ellbar = np.sort(loss)
# Add data for correct plotting
if opts.loss_plot_lims[-1] > loss.max():
    alphas = np.append(alphas, 0)
    ellbar = np.append(ellbar, opts.loss_plot_lims[-1] + 0.01)
if opts.loss_plot_lims[0] < loss.min():
    alphas = np.insert(alphas, 0, 1)
    ellbar = np.insert(ellbar, 0, opts.loss_plot_lims[0] - 0.01)

ax.step(
    ellbar,
    alphas,
    label=f"$\\ell_{{\\alpha}}^{{\\beta}}(\\mathcal{{D}}_1)$",
    color="C2",
    ls="solid",
    where="post",
)

#
# Compute and plot an adversarial example curve
#
weights = loss_fn(model, X_test, y_test)  # model belief in the false Y's
weights /= weights.sum()
idx = rng.choice(np.arange(len(X_test)), size=opts.ndata, p=weights, replace=False)

idx_unique, i = np.unique(idx, return_counts=True)
logger.info(f"These samples are used as adversarial examples: {idx_unique}")
logger.info(f"Their multiplicities are: {i}")

X_data, y_data = X_test.iloc[idx], y_test[idx]
loss = loss_fn(model, X_data, y_data)
alphas = 1 - (np.arange(len(loss)) + 1) / (len(loss) + 1)
ellbar = np.sort(loss)

# Add data for correct plotting
if opts.loss_plot_lims[-1] > loss.max():
    alphas = np.append(alphas, 0)
    ellbar = np.append(ellbar, opts.loss_plot_lims[-1] + 0.01)
if opts.loss_plot_lims[0] < loss.min():
    alphas = np.insert(alphas, 0, 1)
    ellbar = np.insert(ellbar, 0, opts.loss_plot_lims[0] - 0.01)

ax.step(
    ellbar,
    alphas,
    label=f"$\\ell_{{\\alpha}}^{{\\beta}}(\\mathcal{{D}}_2)$",
    color="C1",
    ls="solid",
    where="post",
)


#
# Plot!
#
ax.set_ylabel("$\\alpha$")
ax.set_ylim(opts.loss_level_plot_lims)
ax.set_xlim(opts.loss_plot_lims)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("$\\ell_{{\\alpha}}^{{\\beta}}$")
ax.legend()
fig.savefig(os.path.join(ofolder, "penguin.pdf"))

#
# Finalize
#
logger.info(f"Completed output to {os.path.abspath(ofolder)}")
plt.show()
