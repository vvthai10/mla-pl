import cv2
import os
from utils import normalize
import numpy as np


def visualizer(pathes, anomaly_map, save_path):
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

        # vis = cv2.cvtColor(
        #     cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB
        # )  # RGB
        # mask = normalize(anomaly_map[idx])
        # vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        save_vis = os.path.join(save_path, "imgs")

        if not os.path.exists(save_vis):
            os.makedirs(save_vis)

        cv2.imwrite(
            os.path.join(save_vis, name + "_map.jpg"),
            vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        cv2.imwrite(
            os.path.join(save_vis, name + "_mask.jpg"),
            binary_mask,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        cv2.imwrite(
            os.path.join(save_vis, name + "_img.jpg"),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

        cv2.imwrite(
            os.path.join(save_vis, name + "_gt.jpg"),
            image_mask,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    # print(scoremap.shape)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
