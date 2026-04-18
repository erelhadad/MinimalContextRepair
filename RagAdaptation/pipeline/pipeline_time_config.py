import os
from datetime import datetime

path = "/data/home/erel.hadad/MinimalContextRepair/microsoft_hotpot/examples/ex0003/models/microsoft__Phi-3-mini-4k-instruct/methods/attention"
st = os.stat(path)

print("atime:", datetime.fromtimestamp(st.st_atime))


def print_methods_time(methods_dir_path):
    methods_dict={}
    for method in os.listdir(methods_dir_path):
        method_path= os.path.join(methods_dir_path, method)
        st = os.stat(method_path)
        methods_dict[method]=datetime.fromtimestamp(st.st_atime)

    methods_time={}
    sorted_dict = dict(sorted(methods_dict.items(), key=lambda item: item[1]))
    for key,i in enumerate(sorted_dict):
        methods_time[key]=sorted_dict[i]-sorted_dict[i+1]