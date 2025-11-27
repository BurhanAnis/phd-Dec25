#!/usr/bin/env python3
import argparse
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import openslide
from tqdm import tqdm


def load_model(checkpoint_path, device):
    # build model architecture
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    # load checkpoint
    state = torch.load(checkpoint_path, map_location=device)

    # ---- FIX: strip 'module.' if present ----
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "")  # remove DataParallel prefix
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=True)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference on new slides.")
    parser.add_argument("--index", required=True, help="Index pkl for new slides.")
    parser.add_argument("--model", required=True, help="Path to trained model .pt.")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--out", default="inference_results.pkl")
    args = parser.parse_args()

    # device
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # load model
    model = load_model(args.model, device)

    # load index
    with open(args.index, 'rb') as f:
        slide_index = pickle.load(f)

    # transforms
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    slide_results = {}

    for slide_id, info in slide_index.items():
        slide_path = info["slide_path"]
        level = info["level"]
        patches = info["patches"]

        print(f"\nProcessing slide: {slide_id} ({len(patches)} patches)")

        slide = openslide.OpenSlide(slide_path)
        downsample = slide.level_downsamples[level]

        probs_list = []
        preds_list = []

        for (y, x, _) in tqdm(patches, desc=slide_id):
            loc = (int(x * downsample), int(y * downsample))
            patch = slide.read_region(loc, level,
                                      (args.patch_size, args.patch_size)).convert("RGB")

            tensor = base_tf(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                prob = F.softmax(logits, dim=1)[0, 1].item()
                pred = int(prob > 0.5)

            probs_list.append(prob)
            preds_list.append(pred)

        slide.close()

        slide_results[slide_id] = {
            "probs": probs_list,
            "preds": preds_list,
            **{k: v for k, v in info.items() if k not in ["patches"]}
        }

    # save results
    with open(args.out, "wb") as f:
        pickle.dump(slide_results, f)

    print(f"\nInference complete. Saved results to {args.out}")


if __name__ == "__main__":
    main()
