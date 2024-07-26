import cv2
import os
from utils import normalize
import numpy as np


def visualizer(pathes, anomaly_map, save_path, masked=False):
    for idx, path in enumerate(pathes):

        if "Ungood" not in path:
            continue

        filename = path.split("/")[-1]
        name, ext = filename.split(".")
        # Load the image in original size
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_mask = cv2.imread(path.replace("img", "anomaly_mask"))
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_RGB2BGR)

        # Get the original image size
        original_size = (image.shape[1], image.shape[0])
        # print(anomaly_map[idx].shape)
        # Normalize the anomaly map
        mask = normalize(anomaly_map[idx])

        # Resize the anomaly map to the original image size
        mask_resized = cv2.resize(mask, original_size)

        # Apply the anomaly map to the original image
        vis = apply_ad_scoremap(image, mask_resized)

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        save_vis = os.path.join(save_path, "imgs")

        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        if masked:
            cv2.imwrite(
                os.path.join(save_vis, name + "_map.jpg"),
                mask_image(vis),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(save_vis, name + "_mask.jpg"),
                mask_image(binary_mask),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(save_vis, name + "_img.jpg"),
                mask_image(image),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )

            cv2.imwrite(
                os.path.join(save_vis, name + "_gt.jpg"),
                mask_image(image_mask),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
        else:
            cv2.imwrite(
                os.path.join(save_vis, name + "_map.jpg"),
                vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(save_vis, name + "_mask.jpg"),
                binary_mask,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(save_vis, name + "_img.jpg"),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(save_vis, name + "_gt.jpg"),
                image_mask,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    # print(scoremap.shape)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def mask_image(image, mask_fraction=7.8):
    """
    Mask the first and last mask_fraction of rows of the image.

    Args:
    image (np.array): Input image to be masked.
    mask_fraction (float): Fraction of the image height to mask.

    Returns:
    np.array: Masked image.
    """
    height, width = image.shape[:2]
    mask_height = int(height / mask_fraction)

    # Create a mask
    mask = np.ones_like(image)
    mask[:mask_height, :] = 0
    mask[-mask_height:, :] = 0

    # Apply the mask directly
    masked_image = image.copy()
    masked_image[mask == 0] = 0

    return masked_image
