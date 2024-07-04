from tqdm import tqdm
from utils import setup_output_files, end_output_file
from create_scene import create_scene
from joblib import Parallel, delayed

def update_progress_bar(bar):
    def update(_):
        bar.update()
    return update

def create_dataset(dataset_length, dataset_type="location", max_num_objects=4, n_jobs=1, verbose=False):
    dataset_length = dataset_length//2
    setup_output_files(dataset_type)
    
    # Parallelize scene creation
    _results = [
        result for result in tqdm(Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(create_scene)(i, dataset_type, max_num_objects, verbose=verbose) for i in range(dataset_length)
        ),
        total=dataset_length, desc=f"Creating {dataset_type} dataset")
    ]

    end_output_file(dataset_type)

# create_dataset(10, "counting", n_jobs=2)