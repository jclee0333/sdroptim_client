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

def plot_param_importances(study):
    figure = optuna.visualization.plot_param_importances(study)
    return figure
#def plot_intermediate_value(self):
#    _plot_config_for_jupyterlab()
#    figure = optuna.visualization.plot_intermediate_values(self.get_study())
#    return figure


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


def get_chart_html(args, with_df_csv=False):
    study, study_name, direction = get_study(args.json_file_name)
    df = study.trials_dataframe()
    with open(args.json_file_name, 'r') as data_file:
        gui_params = json.load(data_file)
    chart_y_label = "Optimization Score"
    if 'job_from' in gui_params['hpo_system_attr']:
        if gui_params['hpo_system_attr']['job_from']=='webgui':
            if gui_params['task']=='Classification':
                chart_y_label = 'avg. F1 score'
            elif gui_params['task']=='Regression':
                chart_y_label = 'R2 score'
        elif gui_params['hpo_system_attr']['job_from']=='jupyterlab':
            pass
    #
    if not args.study_csv:
        args.study_csv = study_name+"_df.csv"
    if args.output_dir: # not None
        args.output_dir = args.output_dir + (os.sep if args.output_dir[-1]!=os.sep else "")
    if with_df_csv:
        df.to_csv(args.output_dir+args.study_csv)
    #
    history_figure = history_plot(study, direction, chart_y_label)
    offplot(history_figure, filename = args.output_dir+args.optimhist_html, auto_open=False)
    #
    paramimportance_figure = plot_param_importances(study)
    offplot(paramimportance_figure, filename = args.output_dir+args.paramimpo_html, auto_open=False)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', help="json_file_name", type=str, default = '') # json_file_name
    parser.add_argument('--output_dir', help="output directory", type=str, default= '')
    parser.add_argument('--study_csv', help="output study dataframe filename", type=str)
    parser.add_argument('--optimhist_html', help="output optimization history html filename", type=str, default='history.html')
    parser.add_argument('--paramimpo_html', help="output parameter importance html filename", type=str, default='paramimpo.html')
    #
    args = parser.parse_args()
    get_chart_html(args, with_df_csv=True)
