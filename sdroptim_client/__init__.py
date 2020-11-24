# -*- coding: utf-8 -*-
from sdroptim_client.from_json_to_scripts import FullscriptsGenerator, get_default_generatedpy
from sdroptim_client.hpo_job_submit import SubmitHPOjob, get_params, set_params
from sdroptim_client.hpo_job_submit import create_hpojob, load_hpojob
from sdroptim_client.visualization import get_default_chart_html
def get_portal_address(configure_json='run_conf.json'):
    import os, json, inspect
    with open(os.path.dirname(inspect.getfile(sdroptim_client))+os.sep+configure_json, 'r') as f:
        res = json.load(f)['gid']
    GID = os.getenv('NB_GID')
    for k,v in res.items():
        if GID == k:
            return v
        elif GID == 'default':
            default_v = v
    return v