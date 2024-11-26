# -*- coding: utf-8 -*-
"""Animal_Pose_Estimation_using_YOLOv8.ipynb"""

import os
import yaml
from ultralytics import YOLO
from dataclasses import dataclass, field

@dataclass(frozen=True)
class TrainingConfig:
    DATASET_YAML:   str = "animal-keypoints.yaml"
    MODEL:          str = "yolov8m-pose.pt"
    EPOCHS:         int = 20
    KPT_SHAPE:    tuple = (24, 3)
    PROJECT:        str = "Animal_Keypoints"
    NAME:           str = f"{MODEL.split('.')[0]}_{EPOCHS}_epochs"
    CLASSES_DICT:  dict = field(default_factory=lambda: {0: "dog"})


@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE:    int   = 640
    BATCH_SIZE:    int   = 16
    CLOSE_MOSAIC:  int   = 10
    MOSAIC:        float = 0.4
    FLIP_LR:       float = 0.0  # Turn off horizontal flip.


def main():
    train_config = TrainingConfig()
    data_config = DatasetConfig()

    # Prepare dataset YAML configuration
    current_dir = os.getcwd()

    data_dict = dict(
        path=os.path.join(current_dir, "animal-pose-data"),
        train=os.path.join("train", "images"),
        val=os.path.join("valid", "images"),
        names=train_config.CLASSES_DICT,
        kpt_shape=list(train_config.KPT_SHAPE),
    )

    with open(train_config.DATASET_YAML, "w") as config_file:
        yaml.dump(data_dict, config_file)

    # Initialize YOLOv8 model
    pose_model = YOLO(train_config.MODEL)

    # Start training
    pose_model.train(
        data=train_config.DATASET_YAML,
        epochs=train_config.EPOCHS,
        imgsz=data_config.IMAGE_SIZE,
        batch=data_config.BATCH_SIZE,
        project=train_config.PROJECT,
        name=train_config.NAME,
        close_mosaic=data_config.CLOSE_MOSAIC,
        mosaic=data_config.MOSAIC,
        fliplr=data_config.FLIP_LR,
        device=0 # Specify GPU if available
    )

    # Save the best model weights
    weights_dir = os.path.join(train_config.PROJECT, train_config.NAME, "weights")
    best_weights_path = os.path.join(weights_dir, "best.pt")
    print(f"Training complete. Best weights saved to: {best_weights_path}")

    # Evaluation using the best weights
    model_pose = YOLO(best_weights_path)
    metrics = model_pose.val()

    print(f"Evaluation Metrics: {metrics}")
    return best_weights_path


if __name__ == '__main__':
    main()
