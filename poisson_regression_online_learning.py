import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from io import BytesIO
import pymc3 as pm
from pymc3.distributions import Interpolated
import theano
from scipy import stats
from tqdm import tqdm

def plot_traces(trcs, varnames=None):
    '''Plot traces with overlaid means and values'''

    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)

    ax = pm.traceplot(trcs, varnames=varnames, figsize=(12, nrows * 1.4),
                      lines={k: v['mean'] for k, v in
                             pm.df_summary(trcs, varnames=varnames).iterrows()})

    for i, mn in enumerate(pm.df_summary(trcs, varnames=varnames)['mean']):
        ax[i, 0].annotate('{:.2f}'.format(mn), xy=(mn, 0), xycoords='data',
                          xytext=(5, 10), textcoords='offset points', rotation=90,
                          va='bottom', fontsize='large', color='#AA0022')


def strip_derived_rvs(rvs):
    '''Remove PyMC3-generated RVs from a list'''

    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log', rv.name) or re.search('_interval', rv.name)):
            ret_rvs.append(rv)
    return ret_rvs


def generate_data():
    intercept = 2
    x1 = np.random.random_sample(1)[0]
    true_coeff = -2
    theta = np.exp(intercept + true_coeff * x1)
    output = np.random.poisson(theta)

    return {'intercept': intercept, 'x1': x1, 'true_coeff': true_coeff, 'output': output, 'theta': theta}


train = pd.DataFrame(data=[generate_data() for _ in range(100)], columns=generate_data().keys())

output_shared = theano.shared(train['output'].values)
x1_shared = theano.shared(train['x1'].values)
# train


with pm.Model() as model:
    alpha_0 = pm.Normal('alpha_0', mu=0, sd=10)
    alpha_1 = pm.Normal('alpha_1', mu=-1, sd=0.5)
    theta = (alpha_0 + alpha_1 * x1_shared)

    likelihood = pm.Poisson(
        'output',
        mu=np.exp(theta),
        observed=output_shared,
    )
    trace = pm.sample(1000)


def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)
    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)


traces = [trace]

for _ in tqdm(range(10)):
    new_data = pd.DataFrame(data=[generate_data() for _ in range(1)], columns=generate_data().keys())
    output_shared = theano.shared(new_data['output'].values)
    x1_shared = theano.shared(new_data['x1'].values)

    model = pm.Model()
    with model:
        # pm.glm.GLM.from_formula(formula=fml, data=new_data, family=pm.glm.families.NegativeBinomial())
        alpha_0 = from_posterior('alpha_0', trace['alpha_0'])
        alpha_1 = from_posterior('alpha_1', trace['alpha_1'])
        theta = (alpha_0 + alpha_1 * x1_shared)

        likelihood = pm.Poisson('output',
                                mu=np.exp(theta),
                                observed=output_shared
                                )
        trace = pm.sample(1000, init='adapt_diag', chains=2, cores=4, progressbar=False)
        traces.append(trace)

cmap = mpl.cm.autumn

for param in ['alpha_1']:
    for update_i, trace in enumerate(traces):
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
    plt.axvline({'alpha_1': -2}[param], c='k')
    plt.ylabel('Frequency')
    plt.title(param)
    plt.show()