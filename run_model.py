from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_keypoints(image, keypoints):
    """Draw keypoints on the image."""
    for kp_set in keypoints:
        for kp in kp_set[16:22]:
            x, y = int(kp[0]), int(kp[1])
            if x == 0 and y == 0:
                continue
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def draw_keypoint_connections(image, keypoints):
    """Draw lines connecting all keypoints on the image."""
    for kp_set in keypoints:
        for j in range(22):
            x1, y1 = int(kp_set[i][0]), int(kp_set[i][1])
            x2, y2 = int(kp_set[j][0]), int(kp_set[j][1])
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

def crop_to_keypoints(image, keypoints):
    """Crop the image to cover all keypoints and set the background to white."""
    x_coords = []
    y_coords = []
    for kp_set in keypoints:
        for kp in kp_set[16:22]:
            x, y = int(kp[0]), int(kp[1])
            if x != 0 and y != 0:
                x_coords.append(x)
                y_coords.append(y)

    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cropped_image = image[max(y_min - 10, 0):min(y_max + 10, image.shape[0]), max(x_min - 10, 0):min(x_max + 10, image.shape[1])]
        white_background = np.ones_like(image) * 255
        white_background[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
        return white_background
    else:
        return image

filepath = 'uploads/test_dog.jpg'
MODEL_WEIGHTS_PATH = "weights/best.pt"
pose_model = YOLO(MODEL_WEIGHTS_PATH)
image = cv2.imread(filepath)
results = pose_model.predict(image, conf=0.3, iou=0.55)[0].cpu()
keypoints = results.keypoints.xy.numpy()  # Extract keypoint coordinates

# Create copies of the original image for different drawings
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
processed_image = draw_keypoints(image.copy(), keypoints)
cropped_connected_image = crop_to_keypoints(draw_keypoint_connections(image.copy(), keypoints), keypoints)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left: Original image
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Right: Image with keypoints connected and cropped
axes[1].imshow(cv2.cvtColor(cropped_connected_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("Keypoints Connected (Cropped)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
