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
import logging
import sys
import pathlib

import lib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm

#
# Init logging
#
ofolder = pathlib.Path(__file__).parent / "output"
ofolder.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
TARGETS = (
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(ofolder  / "transcript.log"),
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
logger.info(opts)

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
    model = sps.nbinom(n, p)
    aic = 2*1-2*model.logpmf(Z).sum()
    return model, aic


def fit_poisson(Z):
    """Maximum likliehood poisson model"""
    model:sps.rv_discrete = sps.poisson(Z.mean())
    aic = 2*1-2*model.logpmf(Z).sum()
    return model, aic


model1, aic1 = fit_poisson(Z_train)
model2, aic2 = fit_negbin(Z_train)

logger.info(f"AIC values for each model: poisson: {aic1}, {aic2}")

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
fig.savefig(ofolder / f"earthquake_data.pdf")


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
    logger.info(f"The average loss for model {label} on the calibration data is {loss.mean()}")

ax.set_ylim([0, 1])
ax.set_xlim(opts.loss_plot_lims)
ax.set_ylabel("$\\alpha$")
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("$\\ell_{{\\alpha}}^{{\\beta}}(\\mathcal{{D}})$")

fig.savefig(ofolder /  f"earthquake_losscurve.pdf")

#
# Finalize
#
logger.info(f"finished output to {ofolder.absolute()}")
plt.show()
