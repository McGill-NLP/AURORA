import cv2
import json
import os
from tqdm import tqdm

def extract_first_and_last_frames(video_file):
    # Create base directory path
    base_dir = 'frames/' + os.path.splitext(os.path.basename(video_file))[0]
    
    # Check if both 'first.jpg' and 'last.jpg' already exist
    first_frame_path = os.path.join(base_dir, 'first.jpg')
    last_frame_path = os.path.join(base_dir, 'last.jpg')
    if os.path.exists(first_frame_path) and os.path.exists(last_frame_path):
        print(f"Skipping {video_file} as frames already exist.")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 1:
        print(f"Warning: Video {video_file} has only 1 frame.")

    # Extract and save first frame
    extract_and_save_frame(cap, video_file, 0, 'first', base_dir)
    
    # Extract and save last frame
    if total_frames > 1:
        extract_and_save_frame(cap, video_file, total_frames - 1, 'last', base_dir)

    cap.release()

def extract_and_save_frame(cap, video_file, frame_number, frame_type, base_dir):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    attempts = 0
    while not ret and attempts < 10:
        frame_number -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        attempts += 1

    if not ret:
        print(f"Error: Could not read frame after several attempts at {frame_number} in {video_file}.")
        return

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    frame_file_name = f"{base_dir}/{frame_type}.jpg"
    cv2.imwrite(frame_file_name, frame)
    print(f"Saved {frame_file_name}")

# Read JSON file
json_file = 'valid.json'
with open(json_file, 'r') as file:
    videos = json.load(file)

# Process each video file in the JSON
for video in tqdm(videos):
    video_file_name = f"{video['id']}.webm"  # Assuming webm format; adjust if needed
    video_file_path = os.path.join('./videos', video_file_name)
    extract_first_and_last_frames(video_file_path)
