#!/usr/bin/env python3
import argparse
import pickle
import random
import math


def build_balanced_index(full_index, seed=42):
    random.seed(seed)

    # --- 1. Count tumour patches across all slides ---
    tumour_entries = []          # list of (slide_id, patch)
    normals_by_slide = {}        # slide_id -> list[(slide_id, patch)]

    for slide_id, info in full_index.items():
        tumour_entries.extend(
            (slide_id, p) for p in info["patches"] if p[2]
        )
        normals_by_slide[slide_id] = [
            (slide_id, p) for p in info["patches"] if not p[2]
        ]

    Nt = len(tumour_entries)
    slides = list(normals_by_slide.keys())
    N_slides = len(slides)

    print(f"Total slides: {N_slides}")
    print(f"Total tumour patches: {Nt}")

    # --- 2. Number of normals to sample per slide ---
    normal_per_slide = Nt / N_slides
    normal_per_slide = int(math.ceil(normal_per_slide))

    print(f"Sampling {normal_per_slide} normal patches from EACH slide "
          f"(≈ {normal_per_slide * N_slides} total normals)")

    # --- 3. Sample normals equally per slide ---
    sampled_normals = []

    for slide_id in slides:
        normals = normals_by_slide[slide_id]

        if len(normals) >= normal_per_slide:
            sampled = random.sample(normals, normal_per_slide)
        else:
            # oversample with replacement
            sampled = random.choices(normals, k=normal_per_slide)

        sampled_normals.extend(sampled)

        print(f"{slide_id}: total_norm={len(normals)}, sampled={len(sampled)}")

    print(f"Total sampled normals: {len(sampled_normals)}")
    print(f"Expected final total: {Nt + len(sampled_normals)}")

    # --- 4. Build final index ---
    balanced_index = {}

    for slide_id, info in full_index.items():
        balanced_index[slide_id] = {k: v for k, v in info.items() if k != "patches"}
        balanced_index[slide_id]["patches"] = []

    # Add tumour patches
    for slide_id, patch in tumour_entries:
        balanced_index[slide_id]["patches"].append(patch)

    # Add sampled normals
    for slide_id, patch in sampled_normals:
        balanced_index[slide_id]["patches"].append(patch)

    # Shuffle patches per slide
    for slide_id in balanced_index:
        random.shuffle(balanced_index[slide_id]["patches"])

    return balanced_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_index", required=True)
    parser.add_argument("--out_index", required=True)
    args = parser.parse_args()

    with open(args.full_index, "rb") as f:
        full_index = pickle.load(f)

    balanced = build_balanced_index(full_index)

    with open(args.out_index, "wb") as f:
        pickle.dump(balanced, f)

    print(f"Saved balanced index to {args.out_index}")


if __name__ == "__main__":
    main()


