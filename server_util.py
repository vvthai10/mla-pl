# import os
# import cv2
# import numpy as np
# import base64


# def normalize(pred, max_value=None, min_value=None):
#     if max_value is None or min_value is None:
#         return (pred - pred.min()) / (pred.max() - pred.min())
#     else:
#         return (pred - min_value) / (max_value - min_value)

# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     np_image = np.asarray(image, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     # print(scoremap.shape)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

# def mask_image(image, mask_fraction=7.8):
#     """
#     Mask the first and last mask_fraction of rows of the image.

#     Args:
#     image (np.array): Input image to be masked.
#     mask_fraction (float): Fraction of the image height to mask.

#     Returns:
#     np.array: Masked image.
#     """
#     height, width = image.shape[:2]
#     mask_height = int(height / mask_fraction)

#     # Create a mask
#     mask = np.ones_like(image)
#     mask[:mask_height, :] = 0
#     mask[-mask_height:, :] = 0

#     # Apply the mask directly
#     masked_image = image.copy()
#     masked_image[mask == 0] = 0

#     return masked_image

# def visualizer(size_ori_image, image, anomaly_map):
#     results = {}
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     original_size = (size_ori_image[1], size_ori_image[0])
#     mask = normalize(anomaly_map[0])
#     mask_resized = cv2.resize(mask, original_size)
#     vis = apply_ad_scoremap(image, mask_resized)
#     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#     binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

#     _, map_buffer = cv2.imencode('.jpg', mask_image(vis), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     results["map"] = base64.b64encode(map_buffer).decode('utf-8')

#     _, mask_buffer = cv2.imencode('.jpg', mask_image(binary_mask), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     results["mask"] = base64.b64encode(mask_buffer).decode('utf-8')

#     _, img_buffer = cv2.imencode('.jpg', mask_image(image), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     results["img"] = base64.b64encode(img_buffer).decode('utf-8')

#     return results

import os
import cv2
import numpy as np
import base64


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    # Ensure both image and scoremap have the same dimensions
    if np_image.shape != scoremap.shape:
        scoremap = cv2.resize(scoremap, (np_image.shape[1], np_image.shape[0]))
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualizer(size_ori_image, image, anomaly_map):
    results = {}
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    original_size = (size_ori_image[1], size_ori_image[0])
    mask = normalize(anomaly_map[0])
    mask_resized = cv2.resize(
        mask, (image.shape[1], image.shape[0])
    )  # Resize to match the image
    vis = apply_ad_scoremap(image, mask_resized)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR

    # Resize the binary mask to match the image size
    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

    _, map_buffer = cv2.imencode(
        ".jpg",vis, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    results["map"] = base64.b64encode(map_buffer).decode("utf-8")

    _, mask_buffer = cv2.imencode(
        ".jpg", binary_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    results["mask"] = base64.b64encode(mask_buffer).decode("utf-8")

    _, img_buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    results["img"] = base64.b64encode(img_buffer).decode("utf-8")

    return results
