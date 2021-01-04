# Korea Institute of Science Technology and Information (KISTI)
SDROptim: Science Data Repository (SDR) hyperparameter Optimizer

Create your programming code by using simple metadata (json) only!
Current version of this module can run under the sdr.edison.re.kr portal.

An example of a metadata(json)
{"ml_file_path":"./","kernel":"Python","ml_file_name":"feature.csv","analysis":[],"testing_frame_rate":0.2,"input_columns_index_and_name":{"1":"x"},"output_columns_index_and_name":{"2":"y"},"datatype_of_columns":{"1":"Numeric","2":"Numeric"},"task":"Regression","testing_frame_extract_method":"basic","perf_eval":[],"whole_columns_index_and_name":{"1":"x","2":"y"},"hpo_system_attr":{"study_name":"test_under_10_dataset","stepwise":0,"user_name":"enoz","groupId":7094303,"dejob_id":8954,"n_nodes":1,"userId":7094301,"companyId":20154,"env_name":"","job_name":"test_under_10_datasetstudy_in_enoz_db","job_id":"ca146c2b-e61b-4e4a-9276-96fea968bf4d","time_deadline_sec":100,"greedy":1,"job_from":"webgui","workspace_name":"Demo","job_directory":"automl-20201204094443"},"ml_file_info":{"info_type":"","data_type":"","image_color":""},"hparams":{},"algorithm":["MLR","BT","RF","XGBoost","LightGBM","DL_Pytorch","SVM"]}

--------------------------------
(Usage:)
> python
Python 3.7.6 | packaged by conda-forge | (default, Jun  1 2020, 18:57:50) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>> import sdroptim_client as sc
>> sc.get_default_generatedpy()

--------------------------------
(results: generated.py)
json_file_name = 'metadata.json'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
def objective_gpu(trial):
    train_data, test_data, features, target = load_data()
    algorithm_name = trial.suggest_categorical("algorithm_name_gpu", ['XGBoost', 'LightGBM', 'DL_Pytorch'])
    if algorithm_name == 'XGBoost':
        XGBoost_cv = trial.suggest_int("XGBoost_cv", 5, 5)
        XGBoost_eval_metric = trial.suggest_categorical("XGBoost_eval_metric", ['rmse'])
        XGBoost_num_boost_round = trial.suggest_categorical("XGBoost_num_boost_round", [100, 500, 1000, 2000])
        XGBoost_booster = trial.suggest_categorical("XGBoost_booster", ['gbtree', 'dart'])
        XGBoost_objective = trial.suggest_categorical("XGBoost_objective", ['reg:squarederror'])
--------------------------------

How to Install:
pip install git+https://github.com/jclee0333/sdroptim_client.git

