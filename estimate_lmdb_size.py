#!/usr/bin/env python3
"""
Estimate LMDB storage size from a slide index.

Usage:
    python estimate_lmdb_size.py --index_file index_level_0.pkl
    python estimate_lmdb_size.py --index_file index_level_0.pkl --balance --normal_ratio 1.0
"""
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate LMDB size from slide index")
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--normal_ratio', type=float, default=1.0)
    parser.add_argument('--compression_ratio', type=float, default=0.2,
                        help="Estimated PNG compression ratio (0.1-0.3 typical for histopath)")
    return parser.parse_args()


def count_patches(slide_index, balance=False, normal_ratio=1.0):
    """Count tumor and normal patches."""
    tumor_count = 0
    normal_count = 0
    
    for slide_id, info in slide_index.items():
        for (y, x, is_tumor) in info['patches']:
            if is_tumor:
                tumor_count += 1
            else:
                normal_count += 1
    
    if balance:
        normal_selected = min(normal_count, int(tumor_count * normal_ratio))
        return tumor_count, normal_selected, normal_count
    
    return tumor_count, normal_count, normal_count


def format_size(bytes_size):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"


def main():
    args = parse_args()
    
    with open(args.index_file, 'rb') as f:
        slide_index = pickle.load(f)
    
    tumor_count, normal_selected, normal_total = count_patches(
        slide_index, args.balance, args.normal_ratio
    )
    
    total_patches = tumor_count + normal_selected
    
    # Size calculations
    raw_size_per_patch = args.patch_size * args.patch_size * 3  # RGB
    png_size_per_patch = raw_size_per_patch * args.compression_ratio
    
    # LMDB overhead (~10-20% for keys, B-tree, etc.)
    lmdb_overhead = 1.15
    
    estimated_size = total_patches * png_size_per_patch * lmdb_overhead
    
    # Also estimate raw (HDF5-style) storage
    raw_total = total_patches * raw_size_per_patch
    
    print("=" * 50)
    print("PATCH COUNTS")
    print("=" * 50)
    print(f"Tumor patches:       {tumor_count:,}")
    print(f"Normal patches:      {normal_total:,} total")
    if args.balance:
        print(f"Normal selected:     {normal_selected:,} (ratio={args.normal_ratio})")
    print(f"Total to extract:    {total_patches:,}")
    print()
    print("=" * 50)
    print("SIZE ESTIMATES")
    print("=" * 50)
    print(f"Patch dimensions:    {args.patch_size}x{args.patch_size}x3")
    print(f"Raw size per patch:  {format_size(raw_size_per_patch)}")
    print(f"PNG size per patch:  ~{format_size(png_size_per_patch)} (compression={args.compression_ratio})")
    print()
    print(f"LMDB estimate:       {format_size(estimated_size)}")
    print(f"HDF5 estimate (raw): {format_size(raw_total)}")
    print()
    print("=" * 50)
    print("ESTIMATES AT DIFFERENT COMPRESSION RATIOS")
    print("=" * 50)
    for ratio in [0.10, 0.15, 0.20, 0.25, 0.30]:
        est = total_patches * raw_size_per_patch * ratio * lmdb_overhead
        print(f"  Compression {ratio:.0%}:    {format_size(est)}")


if __name__ == '__main__':
    main()
