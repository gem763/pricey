import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm_notebook

from scipy.stats import norm, laplace
from fbprophet import Prophet
from IPython.core.debugger import set_trace
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler


class Pricer:
    def __init__(self, p, scaler=None, reducer=None, fitscale=0.05, prob=0.7, howfar=250):
        self.p = p
        self.fitscale = fitscale
        self.prob = prob
        self.howfar = howfar
        self.log_p = np.log(p)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.reducer = reducer if reducer is not None else PCA(n_components=0.9, random_state=0)
        self.comp = self._reduce()
        self.comp_forecast = self._comp_forecast()
        self.p_forecast = self._p_forecast()
        
        
    def _reduce(self):
        log_p_scaled = self.scaler.fit_transform(self.log_p)
        comp = self.reducer.fit_transform(log_p_scaled)
        comp_names = ['c'+str(i+1) for i in range(comp.shape[1])]
        return pd.DataFrame(comp, index=self.p.index, columns=comp_names)
    
    def _remodel(self, comp):
        # 참고: reducer.inverse_transform(comp) = comp.dot(reducer.components_)
        return self.scaler.inverse_transform(self.reducer.inverse_transform(comp))
    
    def plot_p(self, symbols=[], logy=True, **kwarg):
        if logy:
            data = self.log_p - self.log_p.iloc[0]
        else:
            data = self.p / self.p.iloc[0]
            
        if len(symbols)!=0:
            data = data[symbols]
            
        ax = data.plot(**kwarg)
        ax.set_xlabel('')
        
    def plot_comp(self, **kwarg):
        ax = self.comp.plot(**kwarg)
        ax.set_xlabel('')
        
    def plot_comp_forecast(self):
        ncomp = self.comp.shape[1]
        fig, axes = plt.subplots(nrows=1, ncols=ncomp, figsize=(3*ncomp, 3), sharey=True)

        for i, ax in enumerate(axes):
            title = 'c' + str(i+1)
            df = self.comp_forecast[title]
            df.p.plot(ax=ax, color='k', lw=1)
            ax.fill_between(df.index, df.pmin, df.pmax, color='lightblue', alpha=1, lw=0)
            ax.set_xlabel('')
            ax.set_title(title)
            ax.autoscale(enable=True, axis='x', tight=True)

            
    def plot_p_forecast(self, *symbols):
        n = len(symbols)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(3*n, 3), sharey=False)
        p_forecast = self.p_forecast.unstack()

        if n==1:
            axes = [axes]

        for i, sym in enumerate(symbols):
            df = np.exp(p_forecast.xs(sym, level=1, axis=1))
            df.p.plot(ax=axes[i], color='k', lw=1)
            axes[i].fill_between(df.index, df.pmin, df.pmax, color='pink', lw=0, alpha=1)
            axes[i].autoscale(enable=True, axis='x', tight=True)
            axes[i].set_xlabel('')
            axes[i].set_title(sym)

        fig.tight_layout()

        
    def _comp_forecast(self):
        comp_forecast = pd.DataFrame()
        width = self.prob**(1/self.comp.shape[1])
        for _cname, _comp in tqdm_notebook(list(self.comp.items())):
            df = _comp.reset_index()
            df.columns = ['ds', 'y']
            comp_forecast[_cname] = _prophet(df, fitscale=self.fitscale, width=width, howfar=self.howfar)
        return comp_forecast.unstack()
    
    def _p_forecast(self):
        #items = [[('pc1','pmin'), ('pc1','pmax')], [('pc2','pmin'), ('pc2','pmax')], [('pc3','pmin'), ('pc3','pmax')]]
        items = [[(c,_p) for _p in ('pmin','pmax')] for c in self.comp_forecast.columns.get_level_values(0).unique()]
        cases = list(product(*items))
        p_forecast = pd.DataFrame()

        for i, case in enumerate(cases):
            proj = self._remodel(self.comp_forecast[list(case)])
            p_forecast[i] = pd.DataFrame(proj, index=self.comp_forecast.index, columns=self.p.columns).stack()

        pmax = p_forecast.max(axis=1)
        pmin = p_forecast.min(axis=1)

        _pfair = self.comp_forecast.xs('pfair', level=1, axis=1)
        proj = self._remodel(_pfair)
        pfair = pd.DataFrame(proj, index=self.comp_forecast.index, columns=self.p.columns).stack()
        p = self.log_p.stack()

        p_forecast['p'] = p
        p_forecast['pmin'] = pmin
        p_forecast['pfair'] = pfair
        p_forecast['pmax'] = pmax
        
        return p_forecast[['p','pmin','pfair','pmax']]
    
    
    def _pbands_stats(self, asis, nfwd):
        pbands = self.p_forecast.loc[asis:][['pmin','pmax','pfair']].unstack() - list(self.p_forecast.loc[asis].p.values)*3
        pbands_nfwd = pbands.iloc[nfwd].unstack()
        std = -(pbands_nfwd.loc['pmax']-pbands_nfwd.loc['pmin']) / 2 / norm.ppf((1-self.prob)/2)
        mean = pbands_nfwd.loc['pfair']
        return mean, std

    def _sample_stats(self, asis, nfwd):
        samples_bf = self.p_forecast.loc[:asis, 'p'].unstack().iloc[-nfwd-1:].diff().iloc[1:]
        samples_af = self.p_forecast.loc[asis:, 'pfair'].unstack().iloc[:nfwd+1].diff().iloc[1:]
        samples = pd.concat([samples_bf, samples_af])

        mean, std = samples.mean(), samples.std()
        mean_nfwd, std_nfwd = (mean - 0.5*(std**2)) * nfwd, std * (nfwd**0.5)
        return mean_nfwd, std_nfwd

    def stats(self, nfwd, up_thres=0.01):
        asis = self.p.index[-1]
        pbands_mean, pbands_std = self._pbands_stats(asis, nfwd)
        sample_mean, sample_std = self._sample_stats(asis, nfwd)

        # 두 분포를 mix 할 때에는 우선 정규분포를 가정한다
        # 두 laplace 분포를 mix 하는 방법이 있을까? 하지만 우선 정규분포로... 2020.05.10
        mean_mixed = 0.5*pbands_mean + 0.5*sample_mean
        std_mixed = ( 0.5*(pbands_std**2) + 0.5*(sample_std**2) )**0.5

        dist = _estimate_dist(mean_mixed, std_mixed, dist=laplace)

        _stats = pd.DataFrame(index=self.p.columns)
        _stats['rband_fair'] = np.exp(dist.mean()) - 1
        _stats['rband_min'], _stats['rband_max'] = np.exp(dist.interval(self.prob)) - 1
        _stats['std'] = dist.std() * ((250/nfwd)**0.5)
        _stats['shband_min'] = _stats.rband_min / dist.std()
        _stats['shband_fair'] = _stats.rband_fair / dist.std()
        _stats['shband_max'] = _stats.rband_max / dist.std()
        _stats['up_prob'] = dist.sf(up_thres)

        return _stats
    
    
    def plot_stats(self, nfwd, up_thres=0.01, sortby='up_prob', excludes=[], h=0.25, title=True):
        _stats = self.stats(nfwd, up_thres=up_thres)
        _stats = _stats.drop(excludes, axis=0).sort_values(by=sortby, axis=0)

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, len(_stats)*h), sharey=True)
        
        if title:
            fig.suptitle(str(nfwd) + ' days forward returns forecast', fontsize=15, fontweight='bold');
            
        opt = {'linewidth':1, 'edgecolor':'k', 'zorder':2, 'height':0.8, 'alpha':1}#, 'color':'lightblue'}

        axes[0].barh(_stats.index, _stats['rband_fair'], **opt)
        axes[0].axvline(x=0, linewidth=1, color='k', zorder=1, ls='-')
        axes[0].set_title('Upside to \nExpected fair price')
        axes[0].set_xticklabels(['{:,.0%}'.format(x) for x in axes[0].get_xticks()]);
        axes[0].margins(0.05, 0.005)

        axes[1].barh(_stats.index, _stats['std'], **opt)
        axes[1].set_title('Expected \nAnnual volatility')
        axes[1].set_xticklabels(['{:,.0%}'.format(x) for x in axes[1].get_xticks()]);
        axes[1].margins(0.05, 0)

        axes[2].barh(_stats.index, _stats['rband_max']-_stats['rband_fair'], left=_stats['rband_fair'], **opt)
        axes[2].barh(_stats.index, _stats['rband_min']-_stats['rband_fair'], left=_stats['rband_fair'], **opt)
        axes[2].axvline(x=0, linewidth=1, color='k', zorder=1, ls='-')
        axes[2].set_xticklabels(['{:,.0%}'.format(x) for x in axes[2].get_xticks()]);
        axes[2].set_title('Return bands \n{:,.0%} prob'.format(self.prob))
        axes[2].margins(0.05, 0)

        axes[3].barh(_stats.index, _stats['shband_max']-_stats['shband_fair'], left=_stats['shband_fair'], **opt)
        axes[3].barh(_stats.index, _stats['shband_min']-_stats['shband_fair'], left=_stats['shband_fair'], **opt)
        axes[3].axvline(x=0, linewidth=1, color='k', zorder=1, ls='-')
        axes[3].set_title('Sharpe bands \n{:,.0%} prob'.format(self.prob))
        axes[3].margins(0.05, 0)

        axes[4].barh(_stats.index, _stats['up_prob'], **opt)
        axes[4].set_title('Upside \nLikelihood')
        axes[4].set_xticklabels(['{:,.0%}'.format(x) for x in axes[4].get_xticks()]);
        axes[4].axvline(x=0.5, linewidth=1, color='k', zorder=1, ls='-')
        axes[4].margins(0.05, 0)
        axes[4].set_xlim(0, 1);

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.91)
    
    
def _prophet(df, fitscale=None, width=None, howfar=None):
    np.random.seed(0)
    m = Prophet(changepoint_prior_scale=fitscale, interval_width=width)
    m.fit(df)
    future = m.make_future_dataframe(periods=howfar)
    forecast = m.predict(future)

    fc = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
    fc.columns = ['pfair', 'pmin', 'pmax']
    fc['p'] = df.set_index('ds').y

    return fc.stack()


def _estimate_dist(mean, std, dist=laplace):
    # 분포의 std = std(param) 가 되도록 scaling 한다
    # 예를들어, laplace(loc=mean, scale=scale(?)).std() = std
    scale = std / dist.std(loc=0, scale=1)
    return dist(loc=mean, scale=scale)