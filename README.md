# AURORA
Code and data for the paper: Learning Action and Reasoning-Centric Image Editing from Videos and Simulation


## TODOs
- [ ] Training ataset access
- [ ] Benchmark access
- [ ] Human ratings
- [ ] Push code for inference
- [ ] Push clean code for reproducing training and evaluation
- [ ] Create a demo of our model
- [ ] Acknowledgements: Something Something, AG, Kubric, EQBEN, MagicBrush (!!)

## Data

On the data side, we release three artifacts:
1. The training dataset (AURORA)
2. A benchmark for testing diverse editing skills (AURORA-Bench): object-centric, action-centric, reasoning-centric, and global edits
3. Human ratings on AURORA-Bench, i.e. for other researchers working image editing metrics


For the training data, use the json files under `data/` as well as the image folders on Zenodo for easy `wget` downloading:

- **MagicBrush**: either use the [original instructions](Link) (via Huggingface) or ours [Zenodo](URL). Download via: `wget X`
- **Action-Genome**: [Zenodo](Link)
- **Something-Something**: You unfortunately need to go to the [original source](https://developer.qualcomm.com/software/ai-datasets/something-something) and download all the zip files and put all the videos in a folder named `videos/`. Then run `data/something/extract_frames.py`.
- **Kubric**: [Zenodo](Link)



