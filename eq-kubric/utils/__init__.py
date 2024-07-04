import os
import shutil
import json

def setup_output_files(dataset_type: str) -> None:
    """
    Sets up the necessary directories and files for the script to run.
    """
    setup_logs()
    start_output_file(dataset_type)

def setup_logs(log_dir="logs") -> None:
    """
    Sets up the logging directory by removing it if it exists and then recreating it.
    
    Args:
    log_dir (str): The path to the log directory to set up.
    """
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

def start_output_file(dataset_type: str) -> None:  
    output_json = f"output/{dataset_type}/eq_kubric_{dataset_type}.json"
    os.makedirs(f"output/{dataset_type}", exist_ok=True)
    with open(output_json, "w") as response_file:
        response_file.write("[\n")

def end_output_file(dataset_type: str) -> None:  
    output_json = f"output/{dataset_type}/eq_kubric_{dataset_type}.json"
    with open(output_json, "ab+") as response_file:
        response_file.seek(-2, os.SEEK_END)
        response_file.truncate()
        response_file.write(b"\n]")

def dataset_dir(dataset_type: str) -> str:
    return f"../../change_descriptions/eqmod/{dataset_type}/"

def save_scene_instruction(output_file_path: str, entries: list, dataset_type: str, index: int, log_file="logs/error_log.txt") -> None:
    try:
        for entry in entries:
            with open(output_file_path, "a") as file:
                json_string = json.dumps(entry, indent=4)
                indented_json_string = "\n    ".join(json_string.splitlines())
                file.write(f"    {indented_json_string},\n")
    except Exception as e:
        error_message = f"{e}\nFailed to save a scene text: Index: {index+1}, Dataset Type: '{dataset_type}'."
        with open(log_file, "a") as file:
            file.write(f"{error_message}\n")
