from datetime import datetime
from ultralytics import YOLO
from time import sleep
import os
from clearml import Task

def train():
    base_name = "test_labels_base_dir_feature"
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = f"{base_name}_{now_time}"
    project_name = "YOLOv8"
    if project_name:
        Task.init(
            project_name='YOLOv8',
            task_name=task_name,
            task_type=Task.TaskTypes.training,
        )
    # Load a model
    # model = YOLO("yolov8m-pose.yaml")   # build a new model from YAML, not pretrained
    model = YOLO("yolov8m-pose.pt")  # load a pretrained model 
    dataset_path = "/thor/yolo_datasets/test_not_tiny1/dataset.yaml"
    batch_size_per_gpu = 16
    num_workers = 8

    # dataset_path = "coco8-pose.yaml"

    # Train the model
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_devices = len(visible_devices.split(","))
    if num_devices == 0:
        print("warning: No CUDA_VISIBLE_DEVICES found. returning")
        return
    batch_size = batch_size_per_gpu * num_devices
    
    results = model.train(
        data=dataset_path,
        save_dir="/thor/yolo_pose/models",
        name=task_name,
        save_period=5,
        epochs=2,
        imgsz=640,
        device=visible_devices,
        batch=batch_size,
        workers=num_workers,
        # box=30, # quadruple the weight of boxes vs the default
        plots=True,
        shear=10.0,
        scale=0.7,
        translate=0.3,
        degrees=45.0,
        auto_augment='augmix',
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        close_mosaic=20,
        mosaic=0.5,
    )

if __name__ == "__main__":
    train()