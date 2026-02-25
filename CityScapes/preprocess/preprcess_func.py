import os
from PIL import Image
from torchvision.transforms import v2
import torch
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def preprocess_cityscapes(root='./CityScapes/data',
                          output_root='./preprocessed_data',
                          sizes=[512],
                          preprocess_labels=True,
                          output_activation="sigmoid"):
    """
    Args:
        sizes: list of ints or list of (h,w) tuples
               Example: [4,8,16,32] or [(64,64),(128,128)]
    """

    if output_activation not in ["sigmoid", "tanh"]:
        raise ValueError("output_activation must be 'sigmoid' or 'tanh'")

    # Ensure sizes is iterable
    if isinstance(sizes, (int, tuple)):
        sizes = [sizes]

    # Convert int sizes to square tuples
    processed_sizes = []
    for s in sizes:
        if isinstance(s, int):
            processed_sizes.append((s, s))
        else:
            processed_sizes.append(s)

    # ---- Build transforms per size ----
    image_transforms = {}
    label_transforms = {}

    for size in processed_sizes:

        if output_activation == "sigmoid":
            img_transform = v2.Compose([
                v2.Resize(size=size, interpolation=InterpolationMode.BILINEAR),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:  # tanh
            img_transform = v2.Compose([
                v2.Resize(size=size, interpolation=InterpolationMode.BILINEAR),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
            ])

        lbl_transform = v2.Compose([
            v2.Resize(size=size, interpolation=InterpolationMode.NEAREST),
            v2.PILToTensor(),
            v2.ToDtype(torch.int64, scale=False),
            v2.Lambda(lambda x: x.squeeze(0))
        ])

        image_transforms[size] = img_transform
        label_transforms[size] = lbl_transform

    # ---- Paths ----
    images_dir = os.path.join(root, "leftImg8bit_trainextra", "leftImg8bit", "train_extra")
    targets_dir = os.path.join(root, "gtCoarse", "train")

    # Collect image paths
    image_paths = []
    for city in os.listdir(images_dir):
        city_img_dir = os.path.join(images_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for file in os.listdir(city_img_dir):
            if file.endswith("_leftImg8bit.png"):
                image_paths.append(os.path.join(city_img_dir, file))

    image_paths.sort()

    print(f"Found {len(image_paths)} images")
    print(f"Generating sizes: {processed_sizes}")
    print(f"Scaling mode: {output_activation}")
    print()

    # ---- Main loop (FAST VERSION) ----
    for img_path in tqdm(image_paths, desc="Processing images"):

        # Load image ONCE
        image = Image.open(img_path).convert("RGB")

        if preprocess_labels:
            target_path = (
                img_path
                .replace("leftImg8bit_trainextra/leftImg8bit", "gtCoarse")
                .replace("_leftImg8bit.png", "_gtCoarse_labelIds.png")
            )

            target = Image.open(target_path) if os.path.exists(target_path) else None
        else:
            target = None

        # For each size
        for size in processed_sizes:

            output_images_dir = os.path.join(
                output_root,
                output_activation,          # <-- added
                f"{size[0]}x{size[1]}",
                "leftImg8bit_trainextra",
                "leftImg8bit",
                "train_extra"
            )

            output_targets_dir = os.path.join(
                output_root,
                output_activation,          # <-- added
                f"{size[0]}x{size[1]}",
                "gtCoarse",
                "train"
            )

            # ---- Image ----
            image_tensor = image_transforms[size](image)

            rel_path = os.path.relpath(img_path, images_dir)
            output_img_path = os.path.join(output_images_dir, rel_path)
            output_img_path = output_img_path.replace("_leftImg8bit.png", "_leftImg8bit.pt")

            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            torch.save(image_tensor, output_img_path)

            # ---- Label ----
            if preprocess_labels and target is not None:

                target_tensor = label_transforms[size](target)

                rel_target_path = os.path.relpath(target_path, targets_dir)
                output_target_path = os.path.join(output_targets_dir, rel_target_path)
                output_target_path = output_target_path.replace(
                    "_gtCoarse_labelIds.png",
                    "_gtCoarse_labelIds.pt"
                )

                os.makedirs(os.path.dirname(output_target_path), exist_ok=True)
                torch.save(target_tensor, output_target_path)

    print("\nPreprocessing complete!")