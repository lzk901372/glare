import argparse
import os
from generate import InferenceOptions, InferenceAgent
from tqdm import tqdm
from pathlib import Path

"""
Usage:

VIDEO_LIST: Text file containing video IDs (one per line without extension). Another situation is that each line contains video_path (Original path to the video file), audio_path (Original path to the audio file), ref_path (Original path to the first-frame reference image) separated by commas.
FRAMES_DIR: Directory containing video folders where each folder contains reference images (video_id/0000.png)
AUDIO_DIR: Directory containing audio files (video_id.wav)
OUTPUT_DIR: Directory to save generated videos while preserving the folder structure of FRAMES_DIR
CHECKPOINT_PATH: Path to the FLOAT checkpoint file
a_cfg_scale: Control the strength of the audio conditioning
e_cfg_scale: Control the strength of the emotion conditioning
r_cfg_scale: Control the strength of the reference conditioning
s_cfg_scale: Control the strength of the style conditioning
ref_cfg_scale: Control the strength of the reference conditioning
i_cfg_scale: Control the strength of the identity conditioning

All other parameters similar to generate.py
"""


def get_relative_path(path: str, base_path: str) -> str:
    return Path(path).relative_to(base_path).as_posix()


if __name__ == '__main__':

    # Create InferenceOptions and add our custom arguments
    opt_parser = InferenceOptions()
    parser = opt_parser.initialize(argparse.ArgumentParser())
    
    # Add batch processing arguments
    parser.add_argument("--video_list", type=str, help="Text file containing video IDs (one per line)")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--ref_dir", type=str, default="./crops", help="Directory containing reference images")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    
    # Parse arguments
    opt = parser.parse_args()
    
    # Set required attributes
    opt.rank, opt.ngpus = 0, 1
    
    # Verify inputs
    assert os.path.isfile(opt.video_list), "Video list file not found"
    assert os.path.isdir(opt.ref_dir), "Reference directory not found"
    assert os.path.isdir(opt.audio_dir), "Audio directory not found"
    
    # Create output directory if it doesn't exist
    os.makedirs(opt.res_dir, exist_ok=True)
    
    # Initialize the inference agent
    agent = InferenceAgent(opt)

    # Read video IDs from file
    with open(opt.video_list, 'r') as f:
        # video_ids = [line.strip() for line in f.readlines() if line.strip()]
        video_ids = []
        lines = f.readlines()
        for line in lines:
            if "," not in line and line.strip():
                video_ids.append(line.strip())
            else:
                video_path, audio_path, ref_path = line.strip().split(",")
                video_ids.append((video_path, audio_path, ref_path))

    print(f"Processing {len(video_ids)} videos from {opt.video_list}")
    print(f"Reference images directory: {opt.ref_dir}")
    print(f"Audio files directory: {opt.audio_dir}")
    print(f"Results directory: {opt.res_dir}")
    print("=" * 50)
        
    # Process each video ID
    successful = 0
    failed = 0
    
    for i, video_id in enumerate(video_ids):
        if isinstance(video_id, str):

            print(f"\n[{i+1}/{len(video_ids)}] Processing video ID: {video_id}")
        
            # Construct file paths for this video ID
            ref_path = os.path.join(opt.ref_dir, f"{video_id}/0000.png")
            aud_path = os.path.join(opt.audio_dir, f"{video_id}.wav")
            
            # Check if files exist
            if not os.path.exists(ref_path):
                print(f"  WARNING: Reference image not found: {ref_path}")
                failed += 1
                continue
            
            if not os.path.exists(aud_path):
                print(f"  WARNING: Audio file not found: {aud_path}")
                failed += 1
                continue
            
            # Generate output video path
            res_video_path = os.path.join(opt.res_dir, f"{video_id}.mp4")
            os.makedirs(os.path.dirname(res_video_path), exist_ok=True)
            
            print(f"  Reference: {ref_path}")
            print(f"  Audio: {aud_path}")
            print(f"  Output: {res_video_path}")

            try:
                # Run inference for this video
                agent.run_inference(
                    res_video_path,
                    ref_path,
                    aud_path,
                    ref_r_s_path = opt.ref_r_s_path,
                    prev_r_s_path = opt.prev_r_s_path,
                    prev_video_path = opt.prev_video_path,
                    a_cfg_scale = opt.a_cfg_scale,
                    e_cfg_scale = opt.e_cfg_scale,
                    r_cfg_scale = opt.r_cfg_scale,
                    s_cfg_scale = opt.s_cfg_scale,
                    ref_cfg_scale = opt.ref_cfg_scale,
                    i_cfg_scale = opt.i_cfg_scale,
                    emo         = opt.emo,
                    nfe         = opt.nfe,
                    no_crop     = opt.no_crop,
                    seed        = opt.seed,
                    verbose     = True,
                    save_latent = opt.save_latent
                )
                print(f"  ✓ Successfully generated: {res_video_path}")
                successful += 1
            except Exception as e:
                print(f"  ✗ Error processing {video_id}: {str(e)}")
                failed += 1
                continue
        
        else:
            video_path, audio_path, ref_path = video_id
            video_id = get_relative_path(video_path, opt.video_dir)
            print(f"\n[{i+1}/{len(video_ids)}] Processing video ID: {video_id}")

            # Check if files exist
            if not os.path.exists(ref_path):
                print(f"  WARNING: Reference image not found: {ref_path}")
                failed += 1
                continue
            
            if not os.path.exists(audio_path):
                print(f"  WARNING: Audio file not found: {audio_path}")
                failed += 1
                continue

            # Generate output video path
            res_video_path = os.path.join(opt.res_dir, f"{video_id}.mp4")
            os.makedirs(os.path.dirname(res_video_path), exist_ok=True)
            
            print(f"  Reference: {ref_path}")
            print(f"  Audio: {audio_path}")
            print(f"  Output: {res_video_path}")

            try:
                # Run inference for this video
                agent.run_inference(
                    res_video_path,
                    ref_path,
                    audio_path,
                    ref_r_s_path = opt.ref_r_s_path,
                    prev_r_s_path = opt.prev_r_s_path,
                    prev_video_path = opt.prev_video_path,
                    a_cfg_scale = opt.a_cfg_scale,
                    e_cfg_scale = opt.e_cfg_scale,
                    r_cfg_scale = opt.r_cfg_scale,
                    s_cfg_scale = opt.s_cfg_scale,
                    ref_cfg_scale = opt.ref_cfg_scale,
                    i_cfg_scale = opt.i_cfg_scale,
                    emo         = opt.emo,
                    nfe         = opt.nfe,
                    no_crop     = opt.no_crop,
                    seed        = opt.seed,
                    verbose     = True,
                    save_latent = opt.save_latent
                )
                print(f"  ✓ Successfully generated: {res_video_path}")
                successful += 1
            except Exception as e:
                print(f"  ✗ Error processing {video_id}: {str(e)}")
                failed += 1
                continue
    
    
    
    print(f"\n" + "=" * 50)
    print("Batch processing completed!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(video_ids)}")