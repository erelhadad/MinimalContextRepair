import os
import pathlib
from datetime import datetime

path = "/data/home/erel.hadad/MinimalContextRepair/microsoft_hotpot/examples/ex0003/models/microsoft__Phi-3-mini-4k-instruct/methods/attention"
st = os.stat(path)

# greate for boolq examples lets cheack on minstral

#./outputs/runs/time_run/microsoft_hotpot

print("atime:", datetime.fromtimestamp(st.st_atime))

def modification_time(path):
    st = os.stat(path)

    return datetime.fromtimestamp(st.st_mtime)

def find_pipeline(methods_dir_path):
    '''needs to be the first one that was generated'''
    pass

methods_list=["attention","random","context_cite","at2","recompute_attention","recompute_context_cite","recompute_at2"]

def create_methods_times(methods_dir_path, outside_dir_path):
    methods_time={}
    next_creation_time= 0.0
    first_creation_time=modification_time(methods_dir_path)
    for i,method in enumerate(methods_list[-1]):
        current_method_path = os.path.join(methods_dir_path, method)
        next_method_path = os.path.join(methods_dir_path,methods_list[i+1])
        first_creation_time=modification_time(current_method_path)
        next_creation_time=modification_time(next_method_path)
        methods_time[method] = next_creation_time-first_creation_time

    #handle last case
    pipeline_path =find_pipeline(outside_dir_path)
    pipeline_time=modification_time(pipeline_path)
    last_creation_time=next_creation_time
    methods_time[methods_list[-1]]= pipeline_time-last_creation_time

    return methods_time

def model_examples_iteration():
    ''' returns: time avregatse for each methods on the models specific examples '''
    pass

def all_examples_iteration_boolq():
    '''call model_examples_iteration for each model '''
    pass


def all_examples_iteration_hotpotqa(model_dir_path, model_dir_name):
    '''call model_examples_iteration for each model '''
    examples_dir =os.path.join(model_dir_path, "examples")

    for i, ex in enumerate(os.listdir(examples_dir)):
        #inside of this folder there is the method folder and the pipeline stuff
        model_example_inside_path = os.path.join(examples_dir, ex, "models", model_dir_name)
        methods_dir_path = os.path.join(model_example_inside_path, "methods")
        create_methods_times(methods_dir_path, model_example_inside_path)







