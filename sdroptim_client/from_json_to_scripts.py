# -*- coding: utf-8 -*-
import json, os
import sdroptim_client.RCodeGenerator as RGen
import sdroptim_client.PythonCodeModulator as PyMod

def FullscriptsGenerator(json_file_name):
    # GUI parameters loading part
    json_file_path = "./"
    json_file_number = ""
    with open(json_file_name) as data_file:
        gui_params = json.load(data_file)
    #
    #prefix_generated_code = "json_file_name = "+"'"+json_file_path+json_file_number+json_file_name+"'\n"
    prefix_generated_code = "json_file_name = "+"'"+json_file_name+"'\n"
    #
    temp = gui_params.copy()
    #############################
    ## make python script (.py)
    #############################
    if type(gui_params['algorithm']) is list:
        jobpath, (uname, sname, job_title, wsname, job_directory) = PyMod.get_jobpath_with_attr(gui_params)
        generated_code = PyMod.from_gui_to_code(gui_params)        
        with open(jobpath+os.sep+job_title+'_generated.py', 'w') as f:
            f.write(prefix_generated_code+generated_code)
            os.chmod(jobpath+os.sep+job_title+'_generated.py', 0o666) # add permission 201012
    #############################
        #############################
        ## make job script(sbatch)
        #############################
        jobscripts = PyMod.get_batch_script(gui_params)
        with open(jobpath+os.sep+'job.sh', 'w') as f:
            f.write(jobscripts)
            os.chmod(jobpath+os.sep+'job.sh', 0o777) # add permission 201012
        #############################
    else: # if not hpo
        if gui_params['kernel'] == 'R':
            generated_code = RGen.from_gui_to_code(gui_params)
        elif gui_params['kernel'] == 'Python':
            generated_code = PyMod.from_gui_to_code(gui_params)
            generated_code = prefix_generated_code + generated_code
        else:
            generated_code = '[ERR] Empty kernel!'
        with open('generated.py', 'w') as f:
            f.write(generated_code)
            os.chmod('generated.py', 0o666) # add permission 201012
####

def get_default_generatedpy():
	json_file_name = "metadata.json"
	FullscriptsGenerator(json_file_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_name', help="name of jsonfile generated by GUI interfaces", default="aml_pytorch-classification-indirect.json")
    args=parser.parse_args()
    FullscriptsGenerator(args.json_file_name)
