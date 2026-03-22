import os
from PIL import Image
from torchvision.transforms import v2
import torch
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def stylegan_preprocess_cityscapes(root='./CityScapes/data',
                                 output_root='./preprocessed_data',
                                 size=512,
                                 preprocess_labels=True,
                                 output_activation="tanh"):
    """
    Args:
        root (str): Path to Cityscapes dataset root directory
        output_root (str): Path to save preprocessed tensors
        size (int): Target dimension to resize square images to (size x size)
        preprocess_labels (bool): Whether to also preprocess label maps
        output_activation (str): Output scaling ('sigmoid' for [0,1], 'tanh' for [-1,1])
    """

    if output_activation not in ["sigmoid", "tanh"]:
        raise ValueError("output_activation must be 'sigmoid' or 'tanh'")

    if not isinstance(size, int):
        raise ValueError("size must be an integer")

    # ---- Build transforms for the specific size ----
    if output_activation == "sigmoid":
        img_transform = v2.Compose([
            v2.Resize(size=(size, size), interpolation=InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:  # tanh
        img_transform = v2.Compose([
            v2.Resize(size=(size, size), interpolation=InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
        ])

    if preprocess_labels:
        lbl_transform = v2.Compose([
            v2.Resize(size=(size, size), interpolation=InterpolationMode.NEAREST),
            v2.PILToTensor(),
            v2.ToDtype(torch.int64, scale=False),
            v2.Lambda(lambda x: x.squeeze(0))
        ])

    # ---- Paths ----
    images_dir = os.path.join(root, "leftImg8bit_trainextra", "leftImg8bit", "train_extra")
    targets_dir = os.path.join(root, "gtCoarse", "train")

    # Collect image paths
    image_paths = []
    if os.path.exists(images_dir):
        for city in os.listdir(images_dir):
            city_img_dir = os.path.join(images_dir, city)
            if not os.path.isdir(city_img_dir):
                continue
            for file in os.listdir(city_img_dir):
                if file.endswith("_leftImg8bit.png"):
                    image_paths.append(os.path.join(city_img_dir, file))
        
        image_paths.sort()

    print(f"Found {len(image_paths)} images")
    print(f"Generating size: {size}x{size}")
    print(f"Scaling mode: {output_activation}")
    print()

    # Define output directories once based on the single size
    output_images_dir = os.path.join(
        output_root,
        output_activation,
        f"{size}x{size}",
        "leftImg8bit_trainextra",
        "leftImg8bit",
        "train_extra"
    )

    output_targets_dir = os.path.join(
        output_root,
        output_activation,
        f"{size}x{size}",
        "gtCoarse",
        "train"
    )

    # ---- Main loop (FAST VERSION) ----
    for img_path in tqdm(image_paths, desc="Processing images"):

        # Load image
        image = Image.open(img_path).convert("RGB")

        # ---- Image Processing & Saving ----
        image_tensor = img_transform(image)

        rel_path = os.path.relpath(img_path, images_dir)
        output_img_path = os.path.join(output_images_dir, rel_path)
        output_img_path = output_img_path.replace("_leftImg8bit.png", "_leftImg8bit.pt")

        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        torch.save(image_tensor, output_img_path)

        # ---- Label Processing & Saving ----
        if preprocess_labels:
            target_path = (
                img_path
                .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                .replace("_leftImg8bit.png", "_gtCoarse_labelIds.png")
            )

            if os.path.exists(target_path):
                target = Image.open(target_path)
                target_tensor = lbl_transform(target)

                rel_target_path = os.path.relpath(target_path, targets_dir)
                output_target_path = os.path.join(output_targets_dir, rel_target_path)
                output_target_path = output_target_path.replace(
                    "_gtCoarse_labelIds.png",
                    "_gtCoarse_labelIds.pt"
                )

                os.makedirs(os.path.dirname(output_target_path), exist_ok=True)
                torch.save(target_tensor, output_target_path)

    print("\nPreprocessing complete!")