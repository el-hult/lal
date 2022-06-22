"""
Exeriment A. Distribution shift analysis
"""

import datetime
import os

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.compose
import sklearn.pipeline
import sklearn.utils
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
import logging
import sys


EXPNAME = "ccdf"

#
# Init logging
#
ofolder = os.path.join(
    "output", EXPNAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(ofolder, exist_ok=True)
logger = logging.getLogger(__name__)
TARGETS = logging.StreamHandler(sys.stdout), logging.FileHandler(
    os.path.join(ofolder, "transcript.log")
)
FORMAT = "%(asctime)s | %(name)14s | %(levelname)7s | %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=TARGETS)
logging.getLogger("matplotlib").setLevel(logging.WARN)


#
# Config Initialization
#
opts = argparse.Namespace(loss_plot_lims=[0, 5], mc_size=2_000, seed=999)
ndata_train = 100
ndata = 30
loss_fn = lambda f, x, y: np.abs(y - f.predict(x))
plt.rcParams.update(
    {
        "font.size": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3, 2),
        "figure.autolayout": True,
    }
)
confidence_levels = np.array(
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
)  # for the coverage plot
ss = np.random.SeedSequence(opts.seed)
rng = np.random.default_rng(ss.generate_state(1))
logger.info(f"Using config: {opts}")


#
# Generate data and train model
#
class Cubic:
    def cef(self, x):
        return x * (x - 1) * (x + 1)

    def gen_data_train(self, n, rng):
        X = rng.normal(scale=0.5, loc=1, size=(n, 1))
        x = X.squeeze()
        y = self.cef(x) + rng.normal(size=n, scale=1)
        return X, y

    def gen_data_test(self, n, rng, *, shift=False):
        if not shift:

            X = rng.normal(scale=0.5, loc=1, size=(n, 1))
        else:
            X = rng.normal(scale=0.75, loc=0.75, size=(n, 1))

        y = self.cef(X.squeeze()) + rng.normal(size=n, scale=1)
        return X, y


data_generator = Cubic()
X_model, y_model = data_generator.gen_data_train(ndata_train, rng)
X_test1, y_test1 = data_generator.gen_data_test(ndata, rng)
X_test2, y_test2 = data_generator.gen_data_test(ndata, rng, shift=True)

model = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=True),
    sklearn.linear_model.LinearRegression(
        fit_intercept=False,
    ),
)

model.fit(X_model, y_model)

logger.info("Fitted model")


#
# Plot data and model
#

xrange = np.linspace(
    min(X_test1.min(), X_test2.min()), max(X_test1.max(), X_test2.max())
)[:, np.newaxis]
frange = model.predict(xrange)
fig, ax_data = plt.subplots()

ax_data.plot(xrange.flatten(), frange, label="f(x)", color="black", linestyle="solid")
ax_data.scatter(X_test1, y_test1, label="$\\mathcal{D}_1$")
ax_data.scatter(X_test2, y_test2, label="$\\mathcal{D}_2$")
ax_data.set_xlabel("x")
ax_data.set_ylabel("y")
ax_data.legend()
plt.tight_layout()
fig.savefig(os.path.join(ofolder, f"{EXPNAME}-data.pdf"))
logger.info("Done with data plot")


#
# Plot level-alpha-loss curves
#
losses = [loss_fn(model, X_test1, y_test1), loss_fn(model, X_test2, y_test2)]
loss_obs_max = max([loss.max() for loss in losses])

fig, ax_lalc = plt.subplots()

for l, loss in enumerate(losses):
    alphas = 1 - (np.arange(len(loss)) + 1) / (len(loss) + 1)
    ellbar = np.sort(loss)
    # Add data for correct plotting
    if opts.loss_plot_lims[-1] > loss.max():
        alphas = np.append(alphas, 0)
        ellbar = np.append(ellbar, opts.loss_plot_lims[-1] + 0.01)
    if opts.loss_plot_lims[0] < loss.min():
        alphas = np.insert(alphas, 0, 1)
        ellbar = np.insert(ellbar, 0, opts.loss_plot_lims[0] - 0.01)

    ax_lalc.step(
        ellbar,
        alphas,
        label=f"$\\bar{{\\ell}}_{{\\alpha}}(\\mathcal{{D}}_{{{str(l + 1)}}})$",
        color="C" + str(l),
        ls="solid",
        where="post",
    )

ax_lalc.legend()
ax_lalc.set_ylim([0, 1])
ax_lalc.set_xlim(opts.loss_plot_lims)
ax_lalc.set_ylabel("$\\alpha$")
ax_lalc.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_lalc.set_xlabel("$\\bar{{\\ell}}_{{\\alpha}}$")
plt.tight_layout()
fig.savefig(os.path.join(ofolder, f"{EXPNAME}-curves.pdf"))
logger.info("Done with level alpha loss curves")


#
# Make MC runs to compute empirical coverage
#

ks = np.ceil((ndata + 1) * (1 - confidence_levels)).astype(int)
if (ks > ndata).any():
    a = confidence_levels.min()
    logger.error(
        f"Some cutoffs is infinity. Choose confidence_level<=~{1/(ndata+1):.4f} or ndata>={(a+1) / a}"
    )

fig, ax_coverage = plt.subplots()

n_conf_levels = len(confidence_levels)
for domain, shift in [(1, False), (2, True)]:
    lots_of_X, lots_of_y = data_generator.gen_data_test(
        (ndata + 1) * opts.mc_size * n_conf_levels, rng=rng, shift=shift
    )
    all_losses = loss_fn(model, lots_of_X, lots_of_y).reshape(
        (opts.mc_size, n_conf_levels, ndata + 1)
    )
    level_alpha_losses = np.zeros((opts.mc_size, n_conf_levels))

    did_exceed = np.zeros((n_conf_levels))
    for k in range(opts.mc_size):
        for j in range(n_conf_levels):
            loss = all_losses[k, j, :-1]
            level_alpha_loss = np.inf if ks[j] > ndata else np.sort(loss)[ks[j] - 1]
            loss_new = all_losses[k, j, -1]
            did_exceed[j] += level_alpha_loss < loss_new
            level_alpha_losses[k, j] += level_alpha_loss

    empirical_levels = did_exceed / opts.mc_size
    ax_coverage.scatter(
        x=confidence_levels,
        y=empirical_levels,
        s=4,
        label=f"$p_{domain}$",
        color=f"C{domain-1}",
    )

    if (did_exceed == 0).any():
        logger.warn(
            "Some empirical levels == 0. Consider running with a larger test set?"
        )

zeroOne = np.array([0, 1])
ax_coverage.plot(zeroOne, zeroOne, color="black", linestyle="solid", alpha=0.3)
ax_coverage.plot(
    zeroOne, zeroOne - 1 / (1 + ndata), color="black", linestyle="dashed", alpha=0.3
)
ax_coverage.set_xlabel("$\\alpha$")
ax_coverage.set_ylabel("$\mathbb{P}[L_{n+1} > \\bar \\ell_{\\alpha}(\mathcal{D})]$")
ax_coverage.set_ylim([0, 1])
ax_coverage.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_coverage.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_coverage.set_xlim([0, 1])
fig.savefig(os.path.join(ofolder, "{EXPNAME}-coverage.pdf"))
logger.info("Completed MC for coverage plot")


#
# Make MC to compute convergence to quantile function
#
fig2, ax_converge = plt.subplots()
confidence_levels = np.linspace(0.05, 0.95, 20)
n_conf_levels = len(confidence_levels)
quantiles1 = np.quantile(
    loss_fn(model, *(data_generator.gen_data_test(100_000, rng=rng, shift=False))),
    1 - confidence_levels,
)
quantiles2 = np.quantile(
    loss_fn(model, *(data_generator.gen_data_test(100_000, rng=rng, shift=True))),
    1 - confidence_levels,
)
quantiles = [quantiles1, quantiles2]

ndatas = [30, 300, 3000]
norm = mcolors.LogNorm(vmin=np.min(ndatas), vmax=np.max(ndatas))
cmap = mcolors.LinearSegmentedColormap.from_list("01gradient", ["C0", "C1"])

domain = 2  # we only do the domain-shift situation
for ndata in ndatas:
    ks = np.ceil((ndata + 1) * (1 - confidence_levels)).astype(int)
    if (ks > ndata).any():
        a = confidence_levels.min()
        logger.error(
            f"Some cutoffs is infinity. Choose confidence_level>=~{1/(ndata+1):.4f} or ndata>={(a+1) / a}"
        )

    all_losses = loss_fn(
        model,
        *data_generator.gen_data_test(
            ndata * opts.mc_size * n_conf_levels, rng=rng, shift=shift
        ),
    ).reshape((opts.mc_size, n_conf_levels, ndata))

    level_alpha_losses = np.zeros((opts.mc_size, n_conf_levels))
    for k in range(opts.mc_size):
        for j in range(n_conf_levels):
            loss = all_losses[k, j]  # loss.shape == (ndata,)
            level_alpha_loss = np.inf if ks[j] > ndata else np.sort(loss)[ks[j] - 1]
            level_alpha_losses[k, j] = level_alpha_loss

    empirical_mean_level_alpha_losses = level_alpha_losses.mean(axis=0)
    ax_converge.plot(
        confidence_levels,
        empirical_mean_level_alpha_losses / quantiles[domain - 1],
        color=cmap(norm(ndata)),
        marker="o",
        markersize="3",
        linestyle="solid",
        label=f"$n=${ndata}",
    )


ax_converge.legend()
ax_converge.set_xlim([0, 1])
ax_converge.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_converge.set_xlabel("$\\alpha$")
ax_converge.set_ylabel("Expected Ratio")
fig2.savefig(os.path.join(ofolder, "{EXPNAME}-sizes.pdf"))


#
# Finalize
#
logger.info("Done!")
plt.show()
