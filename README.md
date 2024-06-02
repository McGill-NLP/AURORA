# AURORA
Code and data for the paper: Learning Action and Reasoning-Centric Image Editing from Videos and Simulation

## Data Setup

For the training data, use the json files under `data/` as well as the image folders on Zenodo for easy `wget` downloading:

- **MagicBrush**: either use the [original instructions](Link) (via Huggingface) or ours [Zenodo](URL). Download via: `wget X`
- **Action-Genome**: [Zenodo](Link)
- **Something-Something**: You unfortunately need to go to the [original source](https://developer.qualcomm.com/software/ai-datasets/something-something) and download all the zip files and put all the videos in a folder named `videos/`. Then run `data/something/extract_frames.py`.
- **Kubric**: [Zenodo](Link)

## TODOs
- [ ] Dataset access
- [ ] Push code for running our model
- [ ] Push clean code for reproducing training and evaluation
- [ ] Create a demo of our model
