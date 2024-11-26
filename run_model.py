from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def draw_keypoints(image, keypoints):
    """Draw keypoints on the image."""
    for kp_set in keypoints:
        for kp in kp_set[16:22]:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

filepath = 'uploads/test_dog.jpg'
MODEL_WEIGHTS_PATH = "weights/best.pt"
pose_model = YOLO(MODEL_WEIGHTS_PATH)
image = cv2.imread(filepath)
results = pose_model.predict(image, conf=0.3, iou=0.55)[0].cpu()
print(results)
keypoints = results.keypoints.xy.numpy()  # Extract keypoint coordinates
processed_image = draw_keypoints(image, keypoints)
plt.imshow(processed_image)
plt.show()