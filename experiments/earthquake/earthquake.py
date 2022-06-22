"""
For a given dataset, 

fit two models.
compare their likliehood ddistributions.
The data example is inspired by 

Data Consistency Approach to Model Validation
Andreas Lindholm; Dave Zachariah; Petre Stoica; Thomas B. Schön
http://dx.doi.org/10.1109/ACCESS.2019.2915109

"""
import argparse
import datetime
import logging
import os
import sys

import lib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm

#
# Init logging
#
EXPNAME = "earthquakes"
ofolder = os.path.join(
    "output", EXPNAME, datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
)
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
# Config and init
#
logger.info(f"Loading config")
loss_fn = lambda f, z: -np.log(f.pmf(z))
opts = argparse.Namespace(loss_plot_lims=[0, 15], ndata_model=100, seed=987654321)
plt.rcParams.update(
    {
        "font.size": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3, 2),
        "figure.autolayout": True,
    }
)
np.random.seed(opts.seed)

#
# Generate data
#
Z_all = lib.quake_data().to_numpy()
ndata_all = len(Z_all)
idx = np.random.choice(ndata_all, size=opts.ndata_model, replace=False)
ndata = ndata_all - opts.ndata_model
logger.info(
    f"Using {opts.ndata_model} data points for model training, and {ndata} for evaluation"
)
Z_train, Z_test = Z_all[idx], np.delete(Z_all, idx)


#
# Train models
#
def fit_negbin(Z):
    """Fit a negative binomial after Z by maximum likliehood

    See
        https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf
            the formulas for maximum liklihood estimation of a NegBin model
            seems to be the same equation that statsmodels use
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html#scipy.stats.nbinom
            specifically the part where they introduce alpha and mu in relation to the n and p variables
    """
    res = sm.NegativeBinomial(Z, np.ones(len(Z))).fit(method="newton", disp=False)
    mu = np.exp(res.params[0])  # the mean of the neg-bin-distribution
    alpha = res.params[1]  # the dispersion parameter
    sigma2 = mu + mu * mu * alpha
    p = mu / sigma2
    n = mu * mu / (sigma2 - mu)
    return sps.nbinom(n, p)


def fit_poisson(Z):
    """Maximum likliehood poisson model"""
    return sps.poisson(Z.mean())


model1 = fit_poisson(Z_train)
model2 = fit_negbin(Z_train)


#
# Check the pdfs vs histogram
#
xs = np.arange(0, 400, 2)
fig, ax = plt.subplots()
ax.hist(
    Z_train,
    bins="auto",
    density=True,
    label=f"$\mathcal{{D}}_{{0}}$",
    alpha=0.3,
    color="C0",
)
ax.hist(
    Z_test, bins="auto", density=True, label=f"$\mathcal{{D}}$", alpha=0.3, color="C1"
)
ax.plot(xs, model1.pmf(xs), label="Poisson")
ax.plot(xs, model2.pmf(xs), label="NegBin")
ax.legend()
fig.savefig(os.path.join(ofolder, f"{EXPNAME}_data.pdf"))


#
# Compare models using level-alpha-loss-curves
#
fig, ax = plt.subplots()

for k, (label, model) in enumerate([("Poisson", model1), ("NegBin", model2)]):
    loss = loss_fn(model, Z_test)
    alphas = 1 - (np.arange(len(loss)) + 1) / (len(loss) + 1)
    ellbar = np.sort(loss)
    # Add data for correct plotting
    if opts.loss_plot_lims[-1] > loss.max():
        alphas = np.append(alphas, 0)
        ellbar = np.append(ellbar, opts.loss_plot_lims[-1] + 0.01)
    if opts.loss_plot_lims[0] < loss.min():
        alphas = np.insert(alphas, 0, 1)
        ellbar = np.insert(ellbar, 0, opts.loss_plot_lims[0] - 0.01)

    ax.step(ellbar, alphas, label=label, color=f"C{k}", ls="solid", where="post")

ax.set_ylim([0, 1])
ax.set_xlim(opts.loss_plot_lims)
ax.set_ylabel("$\\alpha$")
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("$\\bar{{\\ell}}_{{\\alpha}}(\\mathcal{{D}})$")

fig.savefig(os.path.join(ofolder, f"{EXPNAME}_losscurve.pdf"))

#
# Finalize
#
logger.info(f"finished output to {os.path.abspath(ofolder)}")
plt.show()
