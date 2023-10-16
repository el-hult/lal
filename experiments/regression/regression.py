"""
Experiment C: Regression Error Analysis
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import lib
import numpy as np
import argparse
import pathlib

import os
import datetime
import logging
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.pipeline

import sys

#
# Init Logging
#
ofolder = pathlib.Path(__file__).parent / "output"
os.makedirs(ofolder, exist_ok=True)
logger = logging.getLogger(__name__)
TARGETS = logging.StreamHandler(sys.stdout), logging.FileHandler(
    os.path.join(ofolder, "transcript.log")
)
FORMAT = "%(asctime)s | %(name)14s | %(levelname)7s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=TARGETS)
logging.getLogger("matplotlib").setLevel(logging.WARN)


#
# Config & Initialization
#
opts = argparse.Namespace(
    seed=123456,
    confidence_level=0.10,
    ndata_test=100,
    loss_plot_lims=(7.5, 20),
    alpha_plot_lims=(0, 0.20),
)
plt.rcParams.update({"font.size": 8, "legend.fontsize": 8, "figure.figsize": (3, 2)})
logger.info(f"Using options: {opts}")
ss = np.random.SeedSequence(opts.seed)


#
# Generate data
#
class TibshiraniAirfoil:
    def __init__(self, ndata_test, seed=0) -> None:
        """Create a class that samples data from UCI Airfoil in a manner similar (but not identical) to
        what is done for analyzing distribution shift in

        R. J. Tibshirani, R. Foygel Barber, E. Candès, and A. Ramdas, 'Conformal Prediction Under Covariate Shift', in Advances in Neural Information Processing Systems, 2019, vol. 32. [Online]. Available: https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf

        """
        df = lib.load_uci("Airfoil")
        df["Log frequency [log Hz]"] = np.log(df["Frequency [Hz]"])
        df["Suction side log displacement thickness [log m]"] = np.log(
            df["Suction side displacement thickness [m]"]
        )
        df = df[
            [
                "Log frequency [log Hz]",
                "Angle of attack [deg]",
                "Chord length [m]",
                "Free-stream velocity [m/s]",
                "Suction side log displacement thickness [log m]",
                "Scaled sound pressure level [dB]",
            ]
        ]

        beta = np.array([-1, 0, 0, 0, 1])
        w = np.exp(df.iloc[:, :-1] @ beta)
        idx = np.random.default_rng(seed).choice(
            np.arange(len(df)), size=ndata_test, replace=False, p=w / w.sum()
        )

        self.full_data_size = len(df)
        self.X_train = df.drop(idx).iloc[:, :-1]
        self.y_train = df.drop(idx).iloc[:, -1]
        self.X_test = df.iloc[idx, :-1]
        self.y_test = df.iloc[idx, -1]

    def gen_data_train(self):
        X = self.X_train.to_numpy()
        y = self.y_train.to_numpy()
        return X, y

    def gen_data_test(self):
        X = self.X_test.to_numpy()
        y = self.y_test.to_numpy()
        return X, y


data_generator = TibshiraniAirfoil(
    seed=ss.generate_state(1), ndata_test=opts.ndata_test
)
x_train, y_train = data_generator.gen_data_train()
x_test, y_test = data_generator.gen_data_test()
logger.info(f"In total, the dataset size is {data_generator.full_data_size =}")
logger.info(f"The data shape is: {x_train.shape=}, {x_test.shape=}")
logger.info(f"The training data shape is: {x_train.shape=}")
xy_train = np.c_[x_train, y_train]
xy_test = np.c_[x_test, y_test]


#
# Train the model and compute losses
#
model = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.SplineTransformer(), sklearn.linear_model.RidgeCV()
)
model.fit(x_train, y_train)

loss_over = np.maximum(model.predict(x_test) - y_test, 0)
loss_under = np.maximum(y_test - model.predict(x_test), 0)
loss_abs = np.abs(y_test - model.predict(x_test))


logger.info("Starting plotting")
fig, ax = plt.subplots()

#
# Compute and plot the quantile curve
#
for label, loss, linestyle, color in [
    ("overshoot", loss_over, "solid", "C2"),
    ("undershoot", loss_under, "dashed", "C3"),
    ("abs", loss_abs, "dotted", "C0"),
]:

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
        label=f"$\\ell_{{\\alpha}}^{{\\beta}}$ {label}",
        linestyle=linestyle,
        where="post",
        color=color,
    )

ax.legend()
ax.set_xlim(opts.loss_plot_lims)
ax.set_ylim(opts.alpha_plot_lims)
ax.set_ylabel("$\\alpha$")
ax.set_xlabel("$\\ell_{{\\alpha}}^{{\\beta}}$")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
plt.tight_layout()
fig.savefig(os.path.join(ofolder, f"airfoil.pdf"))


#
# Finalize
#
logger.info(f"Completed output to {os.path.abspath(ofolder)}")
plt.show()
