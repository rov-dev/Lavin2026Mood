import os, sys, glob, itertools, tqdm, logging, warnings, time, math, re, json, math, csv, rpy2, xlsxwriter
from datetime import datetime
warnings.simplefilter("ignore")
if not sys.warnoptions: os.environ["PYTHONWARNINGS"] = "ignore" 
import numpy as np
import pandas as pd
import scipy.io as spio
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import gamma as Gamma
from scipy.stats import beta as Beta
from scipy.stats import norm as Normal
import matplotlib.pyplot as plt
import pingouin as pg
from IPython.core.display import display, HTML
import rpy2
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymer4.models import Lmer
import seaborn as sns
from contextlib import contextmanager
import sys, os
from pymer4.models import Lmer, Lm, Lm2
from pymer4.bridge import pandas2R, R2pandas
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
from os import listdir
from os.path import join

import pingouin as pg
from pingouin import ttest
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter

#==============================================================================
# RSYSTEM
#==============================================================================

# install.packages("lme4")
# install.packages("dplyr")
# install.packages("easystats")
# install.packages("parameters")
# install.packages("optimx")
# install.packages("sjPlot")
# install.packages("devtools")
# install.packages("logistf")
# install.packages("blme")
# install.packages("multcomp")
# install.packages("emmeans")
# install.packages("lsmeans")
# devtools::install_github("runehaubo/lmerTestR")
# install.packages("glmmTMB")
# devtools::install_github('ontogenerator/cpdetectoR')

utils = importr("utils")
base = importr("base")
stats = importr("stats")
lme4 = importr("lme4")
optimx = importr("optimx") 
parameters = importr("parameters")
performance = importr("performance")
sjplot = importr("sjPlot")
emmeans = importr("emmeans")
ggeffects = importr("ggeffects")
report = importr("report")
broommixed = importr("broom.mixed")
logistf = importr("logistf")
multcomp = importr("multcomp")

dll = base.__dict__["$"]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def pd_to_r(pd_df):
    with (ro.default_converter + pandas2ri.converter).context():
        r_df = conversion.get_conversion().py2rpy(pd_df)
    return r_df

def r_to_pd(r_df):
    with (rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter).context(): 
         pd_df = rpy2.robjects.conversion.get_conversion().rpy2py(r_df)
    return pd_df

def pandas2R(df):
    """Local conversion of pandas dataframe to R dataframe as recommended by rpy2"""
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.get_conversion().py2rpy(df)
    return data

def R2pandas(rdf):
    """Local conversion of R dataframe to pandas as recommended by rpy2"""
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.get_conversion().rpy2py(rdf)
    return data

def save_r(obj, fname):
    import rpy2.robjects as ro
    saveRDS = ro.r['saveRDS']
    saveRDS(obj, fname)

def read_r(fname):
    import rpy2.robjects as ro
    readRDS = ro.r['readRDS']
    obj = readRDS(fname)
    return obj

#==============================================================================
# TIMESERIE
#==============================================================================

def learning_trial(data, test, crit):
    import rpy2.robjects as ro
    # test = 'ttest'
    # test = 'chisquare'
    # test = 'binomial'
    rstring = (
        """
        function(data){
        suppressMessages(library(cpdetectoR))
        out <- suppressMessages(suppressWarnings(
            cpdetectoR::cp_wrapper(data, TRUE, '"""
            + test 
            + """' ,"""
            + str(crit) 
            + """)
            ))
        out
        }"""
    )
    data = pd_to_r(data)
    func = ro.r(rstring)
    res = func(data)
    res = r_to_pd(res)
    return res

def clusterperm_glmer(formula, data, nperm):
    import rpy2.robjects as ro
    rstring = (
        """
        function(data){
        suppressMessages(library(permutes))
        out <- suppressMessages(suppressWarnings(
            clusterperm.glmer(as.formula( """
        + formula
        +"""), data"""
        +""",series.var=~Trial, nperm = """
        +str(nperm)
        +""", type='anova' """
        +""", buildmerControl = list(direction = 'order', crit = 'LL', quiet = TRUE, ddf = 'Wald') """
        +""")))
        out
        }"""
    )
    data = pd_to_r(data)
    func = ro.r(rstring)
    res = func(data)
    res = r_to_pd(res)
    return res

def clusterperm_lmer(formula, data, nperm):
    import rpy2.robjects as ro
    rstring = (
        """
        function(data){
        suppressMessages(library(permutes))
        out <- suppressMessages(suppressWarnings(
            clusterperm.lmer(as.formula( """
        + formula
        +"""), data"""
        +""",series.var=~Trial, nperm = """
        +str(nperm)
        +""", type='anova' """
        +""", buildmerControl = list(direction = 'order', crit = 'LL', quiet = TRUE, ddf = 'Wald') """
        +""")))
        out
        }"""
    )
    data = pd_to_r(data)
    func = ro.r(rstring)
    res = func(data)
    res = r_to_pd(res)
    return res

def plot_timeseries_cluster(x, DV, mask, ax=None):
    kargs = dict(markersize=4, linewidth=0.6)
    col = {'y0': '#A82D13', 'y1': '#1E6DA0'} #red, blue
    lbs = {'y0': 'High', 'y1': 'Low',}

    d = timeseries_data(x, DV, 'mean')
    er = timeseries_data(x, DV, 'sem')
    for yn, y in d.items():
        y = np.array(y)
        yerr = np.array(er[yn])
        if DV=='Stay':

            y = np.append(np.append(y[:14], np.nan), np.append(y[14:], np.nan))
            yerr = np.append(np.append(yerr[:14], np.nan), np.append(yerr[14:], np.nan))

        x = np.arange(1, len(y)+1)
        kargs = dict(capsize=2, marker="o", linestyle="-",markersize=3, linewidth=0.4) 
        ax.errorbar(x, y, color = col[yn], ecolor = col[yn], label=lbs[yn], **kargs, 
                    yerr=yerr, 
                    )
        ax.fill_between(x, y-yerr, y+yerr,
            alpha=0.3, 
            facecolor= col[yn],
            linewidth=0.7, 
            edgecolor= col[yn], 
            antialiased=True,
            )

    YLim, DVt = (0.1, 1.0), DV
    if DV=='Stay':  YLim, DVt, DVy = (0.3, 1.05), 'stay choice',  'Stay'
    if DV=='Choice': YLim, DVt, DVy = (0.05, 1.05), 'choice accuracy',  'Choice'

    XLim = (0, 30)
    XLim = (0, 31)
    ax.set_ylim(YLim)
    ax.set_xlim(XLim)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.7)

    ax.set_ylabel(DVy, fontsize=9)
    ax.set_xlabel('Trials', fontsize=10)
    ax.set_title('Evolution of average \n'+DVt+' across trials', fontsize=10)

    ax.tick_params(axis='y',labelsize=9)
    ax.tick_params(axis='x',labelsize=8)

    kargs = dict(horizontalalignment='left', verticalalignment='top', fontsize=9,)
    ax.text(0.03, 0.1, 'Learning', transform=ax.transAxes,**kargs)
    ax.text(0.51, 0.1, 'Reversal', transform=ax.transAxes,**kargs)

    leg = ax.legend(title='State', bbox_to_anchor=(1.0, 1), loc='upper left',fontsize=7)
    leg.set_title('Reward\n  State',prop={'size':9})

    ax.fill_between(np.arange(1, 31), 0, 1, where=mask, color='orange', alpha=0.3, transform=ax.get_xaxis_transform())

    return ax

#==============================================================================
# MODEL
#==============================================================================
def lme4_lmer(formula, data):
    import rpy2.robjects as ro
    rstring = (
        """
        function(formula, data){
        suppressMessages(library(lme4))
        out <- suppressMessages(suppressWarnings(
            lmer(formula, data"""
        +""")))
        out
        }"""
    )
    data = pd_to_r(data)
    func = ro.r(rstring)
    res = func(formula, data)
    return res

def lme4_glmer(formula, data):
    import rpy2.robjects as ro
    # Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmerMod]
    family = 'binomial'
    rstring = (
        """
        function(formula, data){
        suppressMessages(library(lme4))
        out <- suppressMessages(suppressWarnings(
            glmer(formula, data"""
        + """,family="""
        + str(family)
        + """,control = glmerControl(optimizer = "bobyqa")"""
        +""")))
        out
        }"""
    )
    data = pd_to_r(data)
    func = ro.r(rstring)
    res = func(formula, data)
    return res

def stats_anova(model0, model1):
    import rpy2.robjects as ro
    rstring = (
        """
        function(model0, model1){
        suppressMessages(library(performance))
        out <- suppressMessages(suppressWarnings(
            anova(model0, model1"""
        +""")))
        out
        }"""
    )
    func = ro.r(rstring)
    res = func(model0, model1)
    res = r_to_pd(res)
    res = res.round(4)
    return res

def parameters_model_parameters_lmm(model):
    hm_df = r_to_pd(parameters.model_parameters(model))
    hm_df = hm_df.loc[(hm_df.Effects == 'fixed'),]
    keep = ['Parameter', 'Coefficient',]
    keep += ['SE', 'CI_low', 'CI_high',]
    keep += ['t', 'p']
    m = hm_df[keep].dropna().round(4).reset_index(drop=True) 
    m['Parameter'] = m['Parameter'].replace('(Intercept)', '')
    m['Parameter'] = m['Parameter'].str.replace(':', ' × ')
    return m

def parameters_model_parameters_glmm(model):
    hm_df = r_to_pd(parameters.model_parameters(model))
    keep = ['Parameter', 'Coefficient',]
    keep += ['SE', 'CI_low', 'CI_high',]
    keep += ['z', 'p']
    hm_df = hm_df.loc[(hm_df.Effects == 'fixed'),]
    m = hm_df[keep].dropna().round(4).reset_index(drop=True) 
    m['Parameter'] = m['Parameter'].replace('(Intercept)', '')
    m['Parameter'] = m['Parameter'].str.replace(':', ' × ')
    m['Odds Ratios'] = np.exp(m['Coefficient'])
    m['OR_CI_low'] = np.exp(m['CI_low'])
    m['OR_CI_high'] = np.exp(m['CI_high'])
    return m

def emmeans_joint_test(model):
    res = r_to_pd(emmeans.joint_tests(model))
    return res

def performance_model_performance_glmm(model):
    return r_to_pd(performance.model_performance(model))

def performance_check_singularity(model):
    out = performance.check_singularity(model)
    return bool(out[0])

def emmeans_pairwise(model, p_adjust):
    import rpy2.robjects as ro
    # p_adjust="tukey"
    # p_adjust="holm"
    rstring = (
        """
        function(model){
        suppressMessages(library(emmeans))
        out <- emmeans(model,pairwise ~ State | Phase """
        + """, adjust='"""
        + p_adjust
        + """',lmer.df='satterthwaite',lmerTest.limit='"""
        + str(99999)
        + """')
        out
        }"""
    )

    func = ro.r(rstring)
    res = func(model)
    res = emmeans.summary_emm_list(res)[1]
    res = r_to_pd(res)
    return res

def sjplot_plot_model_fixed_glmm(model, ivs):
    import rpy2.robjects as ro
    terms = """,terms="""+ 'c(' + ', '.join([ '"'+str(i)+'"' for i in ivs]) + ')'
    transform = ""
    # transform = """,transform = NULL"""
    transform = """,transform = 'exp'"""
    types = "pred"

    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- suppressMessages(suppressWarnings(
            plot_model(model"""
        +""",type="""
        + """'"""
        + types
        + """'"""
        + terms
        + transform
        + """,  pred.type = 'fe'"""
        +""")))"""
        +"""
        out$data
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    res = r_to_pd(res)
    res = res.sort_values(by=['group']).reset_index(drop=True)
    return res

def sjplot_plot_model_fixed_lmm(model, ivs):
    import rpy2.robjects as ro
    terms = """,terms="""+ 'c(' + ', '.join([ '"'+str(i)+'"' for i in ivs]) + ')'
    transform = ""
    transform = """,transform = NULL"""
    # transform = """,transform = 'exp'"""
    types = "pred"

    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- suppressMessages(suppressWarnings(
            plot_model(model"""
        +""",type="""
        + """'"""
        + types
        + """'"""
        + terms
        + transform
        + """,  pred.type = 'fe'"""
        +""")))"""
        +"""
        out$data
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    res = r_to_pd(res)
    res = res.sort_values(by=['group']).reset_index(drop=True)
    return res

def sjplot_pred_fixed_glmm(model, ivs):
    import rpy2.robjects as ro
    terms = """,terms="""+ 'c(' + ', '.join([ '"'+str(i)+'"' for i in ivs]) + ')'
    transform = ""
    # transform = """,transform = NULL"""
    transform = """,transform = 'exp'"""
    types = "pred"

    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- suppressMessages(suppressWarnings(
            plot_model(model"""
        +""",type="""
        + """'"""
        + types
        + """'"""
        + terms
        + transform
        + """,  pred.type = 'fe'"""
        +""")))"""
        +"""
        out$data
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    res = r_to_pd(res)
    res = res.sort_values(by=['group']).reset_index(drop=True)
    return res

def emmeans_pred_probs(model, ivs):
    import rpy2.robjects as ro
    formula = f'pairwise ~ {ivs[1]} | {ivs[0]}'
    rstring = f"""
    function(model){{
    suppressMessages(library(emmeans))
    posthoc <- suppressMessages(suppressWarnings(emmeans(model, {formula}, adjust="holm", lmer.df="satterthwaite", lmerTest.limit=99999)))
    res <- as.data.frame(summary(posthoc, infer=TRUE, type="response")$emmeans)
    class(res) <- "data.frame"
    res
    }}
    """
    func = ro.r(rstring)
    res = r_to_pd(func(model))
    return res.reset_index(drop=True)

def sjplot_estimates_fixed_glmm(model, transform="NULL"):
    import rpy2.robjects as ro
    if transform=="NULL": transform_ = f",transform = NULL"
    else: transform_ = f",transform = '{transform}'"
    types_ = f",type='est'"
    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- suppressMessages(suppressWarnings(
            plot_model(model"""
        + types_
        + transform_
        +""")))"""
        +"""
        out$data$DV <- out$labels$title
        out$data
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    res = r_to_pd(res)
    if transform=="NULL": title = 'Beta Estimates'
    if transform=="exp": title = 'Odds Ratios'
    if transform=="plogis": title = 'Probabilities'
    res['title'] = title

    return res

def sjPlot_tab_model(model, fname):
    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- tab_model(model)
        out$page.content
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    html_tab = np.asarray(res)[0]
    with open('data/out/'+fname+".html", "w") as file: file.write(html_tab)
    display(HTML(html_tab))

def sjplot_estimates_random_glmm(model, ivs):

    terms = """,terms="""+ 'c(' + ', '.join([ '"'+str(i)+'"' for i in ivs]) + ')'
    transform = """,transform = NULL"""
    types = "re"

    rstring = (
        """
        function(model){
        suppressMessages(library(sjPlot))
        out <- suppressMessages(suppressWarnings(
            plot_model(model"""
        +""",type="""
        + """'"""
        + types
        + """'"""
        + terms
        + transform
        +""")))"""
        +"""
        out$data
        }"""
    )
    func = ro.r(rstring)
    res = func(model)
    res = r_to_pd(res)

    keep = ['estimate', 'conf.low', 'conf.high', 'facet', 'term', 'group']
    res = res[keep]
    res['err'] = np.abs((res['conf.low'] - res['conf.high'])/2)
    return res
