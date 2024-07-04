import argparse
from create_dataset import create_dataset

def main():
    parser = argparse.ArgumentParser(description='Create a dataset.')
    parser.add_argument('dataset_length', type=int, help='The length of the dataset')
    parser.add_argument('--dataset_type', type=str, default='location', help='Type of dataset to create (default: location)')
    parser.add_argument('--max_num_objects', type=int, default=4, help='Maximum number of objects in the scene (default: 4)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to run (default: 1)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()
    create_dataset(args.dataset_length, args.dataset_type, args.max_num_objects, args.n_jobs, args.verbose)

if __name__ == '__main__':
    main()
