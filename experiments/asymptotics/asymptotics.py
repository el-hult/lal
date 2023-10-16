"""Illustrate the LAL for onesided asymptotics regions"""


import argparse
import datetime
import json
import logging
import os
import sys
import pathlib

import cycler
import lib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.special
import scipy.stats as sps
import sklearn.compose
import sklearn.gaussian_process
import sklearn.kernel_approximation
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils
import statsmodels.api as sm

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
logging.getLogger("PIL").setLevel(logging.WARN)


#
# Config & Initialization
#
opts = argparse.Namespace(seed=3416,alpha=0.05)
plt.rcParams.update(
    {
        "font.size": 8,
        "legend.fontsize": 8,
        "axes.prop_cycle": (
            cycler.cycler(color=plt.rcParams["axes.prop_cycle"].by_key()["color"][:5])
            + cycler.cycler(linestyle=["solid", "dashed", "dotted", "dashdot", "solid"])
        ),
    }
)
ss = np.random.SeedSequence(3416)
rng = np.random.default_rng(ss.generate_state(1)[0])
rs = np.random.RandomState(ss.generate_state(1))
logger.info(f"Config: {opts}")
with open(os.path.join(ofolder, "config.json"), "w") as f:
    json.dump(
        {"ofolder": str(ofolder), "args": vars(opts), "argv": sys.argv,}, f, indent=2,
    )
np.seterr(all="raise")


#
# Functions and classes
#
def tolerance(alpha, beta, m, losses):
    """Compute the LAL and the actual coverage and actual level
    
    set m==np.inf to get the asymptotic formula

    raise an error if the beta is not a proper fraction of m
    """
    n = len(losses)

    tot_mass = 0
    if m == np.inf:
        beta_true = beta
        j = 1 + sps.binom(n=n, p=beta).ppf(1 - alpha)
        assert int(j) == j
        j = int(j)
        alpha_true = 1 - sps.binom.cdf(j - 1, n=n, p=beta)

    elif 1 <= m:
        i = np.ceil(m * beta)
        beta_true = i / m
        threshold = alpha * scipy.special.binom(n + m, m)
        for j in range(n, 0, -1):
            delta = scipy.special.binom(n - j + m - i, m - i) * scipy.special.binom(
                j + i - 1, j
            )
            if tot_mass + delta >= threshold:
                break
            tot_mass += delta

        alpha_true = tot_mass / scipy.special.binom(n + m, m)

    if beta != beta_true:
        raise ValueError(
            "The supplied beta is not a true integer fraction of the samples"
        )

    kstar = j
    if kstar == n:
        lal = np.inf
    elif kstar == 0:
        lal = -np.inf
    else:
        lal = np.partition(losses, kstar)[kstar]

    return lal, alpha_true


def l1_loss(model, Z):
    """absolute error loss"""
    X, y = Z
    fXs = model(X)
    e = fXs - y
    return np.abs(e)


def sig_figs(i, n):
    """Round `i` to `n` significant digits"""
    return f'{float(f"{i:.{n}g}"):g}'


"""
California housing data
"""
ndata_train = 15000
ndata = 150


def fit_model(Z):
    X, y = Z
    log_y = np.log(y)
    model = sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.Pipeline(
            [
                ("scaler", sklearn.preprocessing.StandardScaler()),
                ("rbf", sklearn.kernel_approximation.RBFSampler()),
                ("ridge", sklearn.linear_model.Ridge(fit_intercept=True)),
            ]
        ),
        param_grid={
            "rbf__gamma": [1e-3,
                           1e-2, 1e-1
                           ],
            "rbf__n_components": [100, 
                                  1_000, 2_000,
                                  ],
            "ridge__alpha": [1e-4, 
                             1e-2, 1e-1
                             ],
        },
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting hyperparam search fitting")
    model.fit(X, log_y)
    logger.info("completed fitting")
    logger.info(f"{model.best_params_}")
    predict = lambda X: np.exp(model.predict(X))
    return predict


df = lib.load_cal_housing().dropna()
logger.info("Full data set has the following details:")
logger.info(f"{df.info()}")
y = df["medianHouseValue"]
X = df[[col for col in df.columns if col != "medianHouseValue"]]
X_train, X_cal, y_train, y_cal = sklearn.model_selection.train_test_split(
    X, y, train_size=ndata_train, random_state=ss.generate_state(1)[0]
)
Z_train = X_train, y_train
idx = rng.choice(np.arange(len(X_cal)), size=ndata)
logger.debug(
    f"Before selecting the actual test data, the data from shich one COULD pick test data is {len(X_cal)=}"
)
X_cal = X_cal.iloc[idx]
y_cal = y_cal.iloc[idx]
Z_cal = X_cal, y_cal

stringify_dollars = (
    lambda x, pos: f"${x / 1e6:1.0f}M"
    if x >= 1e6
    else f"${x / 1e3:1.0f}K"
    if x >= 1e3
    else f"${x:1.0f}"
)
loss_formatter = mtick.FuncFormatter(stringify_dollars)

opts.loss_plot_lims = (0, 100_000)
conf_levels = np.linspace(
    0.99, 0.01, 200
)  # called alpha in the paper


#
# Run the experiment
#
model = fit_model(Z_train)
loss_train = l1_loss(model, Z_train)
loss_cal = l1_loss(model, Z_cal)

logger.info(f"{loss_train.mean()=}")
logger.info(f"{loss_train.max()=}")
logger.info(f"{loss_train.min()=}")
logger.info(f"{loss_cal.mean()=}")
logger.info(f"{loss_cal.max()=}")
logger.info(f"{loss_cal.min()=}")


# HISTORGRAM PLOT #
fig, ax = plt.subplots(figsize=(3,2))
ax.hist(loss_cal)
mid_part = sig_figs(loss_cal.mean(), 2)
ax.axvline(
    loss_cal.mean(), color="black",
)
logger.info(f"CV result is: {loss_cal.mean():,}")
ax.xaxis.set_major_formatter(loss_formatter)
ax.set(
    xlabel=f"$\\ell(X,Y)$", ylabel="frequency",
)
fig.savefig(
    os.path.join(ofolder, "asymptotics_cv.pdf"), bbox_inches="tight", pad_inches=0
)


# DIFFERENT LAL PLOTS #
def plot_lal_curve(m, beta_, ax_solo_, label):
    lals = np.zeros(len(conf_levels))
    alpha_trues = np.zeros(len(conf_levels))
    for k in range(len(conf_levels)):
        lals[k], alpha_trues[k] = tolerance(conf_levels[k], beta_, m, loss_cal)

    lals_plot = lals[np.isfinite(lals)]
    alphas_plot = alpha_trues[np.isfinite(lals)]
    if len(lals_plot) > 0:
        # Add data for correct plotting
        if opts.loss_plot_lims[-1] > lals_plot.max():
            alphas_plot = np.append(alphas_plot, 0)
            lals_plot = np.append(lals_plot, opts.loss_plot_lims[-1] + 0.01)
        if opts.loss_plot_lims[0] < lals_plot.min():
            alphas_plot = np.insert(alphas_plot, 0, 1)
            lals_plot = np.insert(lals_plot, 0, opts.loss_plot_lims[0] - 0.01)

        logger.debug(f"{m=}, {beta_*m=}")
        ax_solo_.step(
            lals_plot, alphas_plot, label=label, where="post",
        )

    else:
        logger.error("No LAL were computed as finite!")


def make_standard_lal_plots():
    ndata_news_betas = [
        (1, [1]),
        (30, [0.2, 0.4, 0.6, 0.8]),
        (np.inf, [0.2, 0.4, 0.6, 0.8]),
    ]
    fig, axs = plt.subplots(1,3,figsize=(6,2),sharey=True,gridspec_kw = {'wspace':0.1})

    axs[0].set_ylabel("$\\alpha$")
    # axs[0].yaxis.set_label_coords(-0.1,1.2)
    axs[-1].xaxis.set_label_coords(1.2, -0.1)
    axs[-1].set_xlabel("$\\ell_{{\\alpha}}^{{\\beta}}(D)$")
    for ax,(m, betas) in zip(axs,ndata_news_betas):

        for beta in betas:
            plot_lal_curve(m, beta, ax, f"$\\beta={beta:.2f}$")

        ax.set(
            ylim=[0, 1],
            xlim=opts.loss_plot_lims,
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(loss_formatter)
        ax.xaxis.set_tick_params(rotation=90)

        ax.legend(prop={'size': 7})
        fig.savefig(
            os.path.join(ofolder, f"asymptotics_all.pdf"),
            bbox_inches="tight",
            pad_inches=0,
        )

make_standard_lal_plots()

# SHOWCASE PLOT #
def make_header_plot():

    fig, ax = plt.subplots(figsize=(6,2))
    plot_lal_curve(
        m=100, beta_=0.5, ax_solo_=ax, label=f"LAL-curve ($\\beta={0.5:.2f})$"
    )
    plot_lal_curve(
        m=100, beta_=0.75, ax_solo_=ax, label=f"LAL-curve ($\\beta={0.75:.2f}$)"
    )
    cv_loss = loss_cal.mean()
    ax.axvline(cv_loss, color="black", label="Expected loss from CV")

    ax.legend()
    ax.set(
        ylabel="$\\alpha$", ylim=[0, 1], xlim=opts.loss_plot_lims,
    )
    ax.grid(visible=True,which='both')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(.20))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(loss_formatter)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(20_000))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10_000))
    fig.savefig(
        os.path.join(ofolder, f"asymptotics_showcase.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )


make_header_plot()


#
# Finalize
#
logger.info(f"finished output to {os.path.abspath(ofolder)}")
plt.show()
