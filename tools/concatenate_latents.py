#!/usr/bin/env python3

import torch
import argparse
import os
from pathlib import Path
import pandas as pd

def concatenate_latents(output_name: str, index_csv: str, num_gpus: int, cleanup: bool = True):
    """
    Concatenate latent tensors from multiple GPU ranks in correct order.

    Args:
        output_name (str): Base output filename
        num_gpus (int): Number of GPUs used for processing
        index_csv (str): Path to index CSV file
        cleanup (bool): Whether to delete individual rank files after concatenation
    """

    # Get base name and extension
    output_base, output_ext = os.path.splitext(output_name)

    # Collect all rank files
    rank_files = []
    rank_tensors = []

    for rank in range(num_gpus):
        rank_file = f"{output_base}_rank{rank}{output_ext}"
        if not os.path.exists(rank_file):
            raise FileNotFoundError(f"Rank file {rank_file} not found!")

        print(f"Loading {rank_file}...")
        tensor = torch.load(rank_file, map_location='cpu')
        print(f"  Shape: {tensor.shape}")

        rank_files.append(rank_file)
        rank_tensors.append(tensor)

    # Concatenate tensors in rank order
    print("\nConcatenating tensors...")
    concatenated = torch.cat(rank_tensors, dim=0)
    print(f"Final concatenated shape: {concatenated.shape}")

    # Save the concatenated result
    print(f"Saving concatenated result to {output_name}...")
    torch.save(concatenated, output_name)

    # Verify the save was successful
    if os.path.exists(output_name):
        # Quick verification by loading and checking shape
        verification = torch.load(output_name, map_location='cpu')
        print(f"Verification: Saved tensor shape: {verification.shape}")
        del verification  # Free memory

        print(f"✓ Successfully saved concatenated latents to {output_name}")
    else:
        raise RuntimeError(f"Failed to save concatenated result to {output_name}")
    
    # Combine index csvs
    if index_csv is not None:
        rank_start_idx = 0
        for rank in range(num_gpus):
            index_csv_rank = index_csv.replace(".csv", f"_rank{rank}.csv")
            if not os.path.exists(index_csv_rank):
                raise FileNotFoundError(f"Index csv {index_csv_rank} not found!")
            index_df = pd.read_csv(index_csv_rank)
            index_df['StartIdx'] += rank_start_idx
            index_df['EndIdx'] += rank_start_idx
            rank_start_idx += index_df['EndIdx'].iloc[-1]
        index_df.to_csv(index_csv, index=False)
        print(f"✓ Successfully combined index csvs to {index_csv}")

    # Cleanup individual rank files if requested
    if cleanup:
        print("\nCleaning up individual rank files...")
        for rank_file in rank_files:
            try:
                os.remove(rank_file)
                print(f"  Removed {rank_file}")
            except OSError as e:
                print(f"  Warning: Could not remove {rank_file}: {e}")

    print("\n✓ Concatenation completed successfully!")

    # Print summary statistics
    total_samples = sum(tensor.shape[0] for tensor in rank_tensors)
    print(f"\nSummary:")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Final tensor shape: {concatenated.shape}")
    print(f"  Output file: {output_name}")

def main():
    parser = argparse.ArgumentParser(description='Concatenate latent tensors from parallel preprocessing')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Base output filename (e.g., hallo3_latents.pth)')
    parser.add_argument('--num_gpus', type=int, required=True,
                        help='Number of GPUs used for processing')
    parser.add_argument('--no_cleanup', action='store_true',
                        help='Keep individual rank files after concatenation')
    parser.add_argument('--index_csv', type=str, required=True,
                        help='Path to index CSV file')

    args = parser.parse_args()

    try:
        concatenate_latents(
            output_name=args.output_name,
            index_csv=args.index_csv,
            num_gpus=args.num_gpus,
            cleanup=not args.no_cleanup
        )
    except Exception as e:
        print(f"Error during concatenation: {e}")
        exit(1)

if __name__ == "__main__":
    main()
