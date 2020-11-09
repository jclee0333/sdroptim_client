# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.offline import plot as offplot
import numpy as np
import os
import optuna, base64, json, argparse, easydict
############################
### render config. for jupyter lab
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'colab'
#from plotly.offline import init_notebook_mode, iplot
############################
def history_plot(study, direction, chart_y_label="Optimization Score"):
    df=study.trials_dataframe()
    layout = go.Layout(
            title="Optimization History by time",
            xaxis={"title": "# seconds"},
            yaxis={"title": chart_y_label + "("+("high" if direction=='maximize' else "low")+"er is better)"},
        )
    def get_traces(df, direction):
        df['cum_time_to_sec'] = df['datetime_complete'].apply(lambda x: (x - df['datetime_start'].iloc[0]).seconds)
        df=df[df['state']=='COMPLETE']
        
        best_values=[]
        df=df.sort_values(by=['cum_time_to_sec'])
        if direction == 'maximize':
            cur_max = -float("inf")
            for i in range(len(df)):
                cur_max = max(cur_max, df['value'].iloc[i])
                best_values.append(cur_max)
            best_idx = best_values.index(max(best_values))
        elif direction == 'minimize':
            cur_min = float("inf")
            for i in range(len(df)):
                cur_min = min(cur_min, df['value'].iloc[i])
                best_values.append(cur_min)
            best_idx = best_values.index(min(best_values))
        #
        traces = [
            go.Scatter(
                x=df['cum_time_to_sec'],
                y=df['value'],
                mode="markers",
                marker=dict(size=3),
                name="Score",
                #hovertext=df['number'].map(lambda x :"idx: "+str(x))
            ),
            go.Scatter(x=df['cum_time_to_sec'], y=best_values, name="Best Score", mode='lines+markers',
                      hovertext=df['number'].map(lambda x :"idx: "+str(x)))
        ]
        #print(len(df),i)
        #print(best_values)
        return traces, df['cum_time_to_sec'].iloc[best_idx], best_values[best_idx]
    t1, best_time, best_value=get_traces(df, direction)
    #t2=get_traces(df2, "optuna-base")
    #return t1,t2
    figure = go.Figure(data=t1, layout=layout)
    figure.update_layout(showlegend=False,
                        annotations=[
                            dict(
        x=best_time,
        y=best_value,
        xref="x",
        yref="y",
        text=str(best_time)+"s, "+str(best_value),
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)
                        ])
    return figure

### supporting multiple algorithms for plotting param importances (2020-10-12) by jclee
def get_params_list_from_df(df, algo_name):
    df = df.dropna(axis='columns')
    each_params=[x for x in df.columns if x.startswith("params_"+algo_name)]
    #### remove columns containing a unique value only.
    removal_params=[]
    for each_p in each_params:
        if len(df[each_p].unique())==1:
            removal_params.append(each_p)
    ####
    res = [x.replace('params_','') for x in each_params if x not in removal_params]
    return res

def get_params_list_from_multiple_algorithms_study(study):
    res=[]
    df=study.trials_dataframe()
    if 'user_attrs_algorithm_name' in df.columns:
        algorithm_list = df['user_attrs_algorithm_name'].unique()
        for each in algorithm_list:
            each_algo_df=df[df['user_attrs_algorithm_name']==each]
            ### divide CNN/FNN in DL_Pytorch
            if each == 'DL_Pytorch':
                DL_modeltype=each_algo_df['params_DL_Pytorch_model'].unique()
                for each_DL_modeltype in DL_modeltype:
                    sampled_df=each_algo_df[each_algo_df['params_DL_Pytorch_model']==each_DL_modeltype]
                    each_params=get_params_list_from_df(sampled_df, each)
                    res.append((each+"_"+each_DL_modeltype, each_params))
                    #optuna.visualization.plot_param_importances(study, params=each_params)
            ####    
            each_params=get_params_list_from_df(each_algo_df, each)
            res.append((each,each_params))
            #optuna.visualization.plot_param_importances(study, params=each_params)
    return res

def plot_param_importances(study):
    figure = optuna.visualization.plot_param_importances(study)
    return figure

def get_study(json_file_name):
    import warnings
    warnings.filterwarnings(action='ignore')
    with open(json_file_name) as data_file:
        gui_params = json.load(data_file)
    key = str(base64.b64decode('cG9zdGdyZXNxbDovL3Bvc3RncmVzOnBvc3RncmVzQDE1MC4xODMuMjQ3LjI0NDo1NDMyLw=='))[2:-1]
    url = key + gui_params['hpo_system_attr']['user_name']
    study_name = gui_params['hpo_system_attr']['study_name']
    try:
        rdbs = optuna.storages.RDBStorage(url)
        study_id = rdbs.get_study_id_from_name(study_name)
    except:
        raise ValueError(
                "Cannot find the study_name {}".format(
                    study_name#, args.storage_url
                ))
    try:
        direction='minimize'
        s = optuna.create_study(load_if_exists=True, study_name=study_name, storage=url, direction=direction)
    except:
        direction='maximize'
        s = optuna.create_study(load_if_exists=True, study_name=study_name, storage=url, direction=direction)
    warnings.filterwarnings(action='default')
    return s, study_name, direction


#def get_chart_html(args, with_df_csv=False):
#    study, study_name, direction = get_study(args.json_file_name)
#    df = study.trials_dataframe()
#    with open(args.json_file_name, 'r') as data_file:
#        gui_params = json.load(data_file)
#    chart_y_label = "Optimization Score"
#    if 'job_from' in gui_params['hpo_system_attr']:
#        if gui_params['hpo_system_attr']['job_from']=='webgui':
#            webgui= True
#            jupyterlab = False
#            if gui_params['task']=='Classification':
#                chart_y_label = 'avg. F1 score'
#            elif gui_params['task']=='Regression':
#                chart_y_label = 'R2 score'
#        elif gui_params['hpo_system_attr']['job_from']=='jupyterlab':
#            webgui = False
#            jupyterlab = True
#    #
#    if not args.study_csv:
#        args.study_csv = study_name+"_df.csv"
#    if args.output_dir: # not None
#        args.output_dir = args.output_dir + (os.sep if args.output_dir[-1]!=os.sep else "")
#    if with_df_csv:
#        df.to_csv(args.output_dir+args.study_csv)
#    #
#    history_figure = history_plot(study, direction, chart_y_label)
#    offplot(history_figure, filename = args.output_dir+args.optimhist_html, auto_open=False)
#    #
#    if webgui:
#        params_in_all_algorithms = get_params_list_from_multiple_algorithms_study(study)
#        if len(params_in_all_algorithms)>1:
#            for each_params in params_in_all_algorithms: # each_params[0] = algo_name, each_params[1] = params list
#                paramimportance_figure = mod_plot_param_importances(study, title=each_params[0], params=each_params[1])
#                offplot(paramimportance_figure, filename = args.output_dir+each_params[0]+"_"+args.paramimpo_html, auto_open=False)
#    else:
#        paramimportance_figure = plot_param_importances(study)
#        offplot(paramimportance_figure, filename = args.output_dir+args.paramimpo_html, auto_open=False)

def get_chart_html(args, with_df_csv=False, history="True", paramimpo="False"):
    study, study_name, direction = get_study(args.json_file_name)
    df = study.trials_dataframe()
    with open(args.json_file_name, 'r') as data_file:
        gui_params = json.load(data_file)
    chart_y_label = "Optimization Score"
    if 'job_from' in gui_params['hpo_system_attr']:
        if gui_params['hpo_system_attr']['job_from']=='webgui':
            webgui= True
            jupyterlab = False
            if gui_params['task']=='Classification':
                chart_y_label = 'avg. F1 score'
            elif gui_params['task']=='Regression':
                chart_y_label = 'R2 score'
        elif gui_params['hpo_system_attr']['job_from']=='jupyterlab':
            webgui = False
            jupyterlab = True
    #
    if not args.study_csv:
        args.study_csv = study_name+"_df.csv"
    if args.output_dir: # not None
        args.output_dir = args.output_dir + (os.sep if args.output_dir[-1]!=os.sep else "")
    if with_df_csv:
        df.to_csv(args.output_dir+args.study_csv)
        os.chmod(args.output_dir+args.study_csv, 0o770) # add permission 201012
    #
    if history == "True":
        history_figure = history_plot(study, direction, chart_y_label)
        offplot(history_figure, filename = args.output_dir+args.optimhist_html, auto_open=False)
        os.chmod(args.output_dir+args.optimhist_html, 0o770) # add permission 201012
    #
    if paramimpo == "True":
        if webgui:
            params_in_all_algorithms = get_params_list_from_multiple_algorithms_study(study)
            if len(params_in_all_algorithms)>=1:
                for each_params in params_in_all_algorithms: # each_params[0] = algo_name, each_params[1] = params list
                    try:
                        paramimportance_figure = mod_plot_param_importances(study, title=each_params[0], params=each_params[1])
                        offplot(paramimportance_figure, filename = args.output_dir+each_params[0]+"_"+args.paramimpo_html, auto_open=False)
                        os.chmod(args.output_dir+each_params[0]+"_"+args.paramimpo_html, 0o770) # add permission 201012
                    except:
                        pass
        else:
            paramimportance_figure = plot_param_importances(study)
            offplot(paramimportance_figure, filename = args.output_dir+args.paramimpo_html, auto_open=False)
            os.chmod(args.output_dir+args.paramimpo_html, 0o770) # add permission 201012



def get_default_args():
    args = easydict.EasyDict({
            #"user_name":"",
            "json_file_name":"metadata.json",
            "output_dir":"",
            "study_csv":"",
            "optimhist_html":"history.html",
            "paramimpo_html":"paramimpo.html"
    })
    return args  

def get_default_chart_html(json_file_name="", output_dir = "", study_csv="", optimhist_html="", paramimpo_html=""):
    args=get_default_args()
    if json_file_name:
        args.json_file_name = json_file_name
    if output_dir:
        args.output_dir = output_dir
    if study_csv:
        args.study_csv = study_csv
    if optimhist_html:
        args.optimhist_html = optimhist_html
    if paramimpo_html:
        args.paramimpo_html = paramimpo_html
    get_chart_html(args, with_df_csv=True)

def get_all_chart_html(json_file_name="metadata.json", output_dir = "./", study_csv="", optimhist_html="", paramimpo_html=""):
    args=get_default_args()
    if json_file_name:
        args.json_file_name = json_file_name
    if output_dir:
        args.output_dir = output_dir
    if study_csv:
        args.study_csv = study_csv
    if optimhist_html:
        args.optimhist_html = optimhist_html
    if paramimpo_html:
        args.paramimpo_html = paramimpo_html
    get_chart_html(args, with_df_csv=True, history="True", paramimpo="True")

####
from collections import OrderedDict
from typing import List
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports

if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    import plotly
    Blues = plotly.colors.sequential.Blues
    _distribution_colors = {
        UniformDistribution: Blues[-1],
        LogUniformDistribution: Blues[-1],
        DiscreteUniformDistribution: Blues[-1],
        IntUniformDistribution: Blues[-2],
        IntLogUniformDistribution: Blues[-2],
        CategoricalDistribution: Blues[-4],
    }

logger = get_logger(__name__)


def mod_plot_param_importances(
    study: Study, evaluator: BaseImportanceEvaluator = None, title = "", params: Optional[List[str]] = None
) -> "go.Figure":
    base_title="Hyperparameter Importances"
    if title:
        final_title=title+" "+base_title
    layout = go.Layout(
        title=final_title,
        xaxis={"title": "Importance"},
        yaxis={"title": "Hyperparameter"},
        showlegend=False,
    )
    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return go.Figure(data=[], layout=layout)
    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params
    )
    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())
    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_values,
                y=param_names,
                text=importance_values,
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=[
                    _make_hovertext(param_name, importance, study)
                    for param_name, importance in importances.items()
                ],
                marker_color=[_get_color(param_name, study) for param_name in param_names],
                orientation="h",
            )
        ],
        layout=layout,
    )
    return fig

def _get_distribution(param_name: str, study: Study) -> BaseDistribution:
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False

def _get_color(param_name: str, study: Study) -> str:
    return _distribution_colors[type(_get_distribution(param_name, study))]

def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )
####

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', help="json_file_name", type=str, default = '') # json_file_name
    parser.add_argument('--output_dir', help="output directory", type=str, default= '')
    parser.add_argument('--study_csv', help="output study dataframe filename", type=str)
    parser.add_argument('--optimhist_html', help="output optimization history html filename", type=str, default='history.html')
    parser.add_argument('--paramimpo_html', help="output parameter importance html filename", type=str, default='paramimpo.html')
    parser.add_argument('--history', help='get history html True/False', type=str, default='True')
    parser.add_argument('--paramimpo', help='get paramimpo html True/False', type=str, default='False')
    #
    args = parser.parse_args()
    get_chart_html(args, with_df_csv=True, history=args.history, paramimpo=args.paramimpo)
