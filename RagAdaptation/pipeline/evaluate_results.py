
import argparse
from pathlib import Path
import json




def scan_dir(dir_path,models):

    roor= Path(dir_path)

    for example_dir in roor.iterdir():
        if example_dir.is_dir():
            models_dir = example_dir / 'models'
            context_length =""
            if models_dir.is_dir():
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name in models:
                        json_path= models_dir.rglob("pipeline_result_methods_*.json")
                        if json_path:
                            with json_path.open("r", encoding="utf-8") as f:
                                data = json.load(f)






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_main_dir", required=True, type=str)
    parser.add_argument("--output_main_dir", required=True, type=str)
    parser.add_argument("--models", nargs="+",  default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"])
    args = parser.parse_args()







