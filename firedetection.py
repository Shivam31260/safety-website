import torch
import os
# Path to YOLOv5 model weights
# MODEL_PATH = r"best.pt files\firedetection.pt"
MODEL_PATH = os.path.join('best.pt files', 'firedetection.pt')


def loadfiredetection_model(conf_threshold=0.6, iou_threshold=0.45):
    """
    Loads the YOLOv5 model with the specified settings.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model.eval()
    model.conf = conf_threshold  # confidence threshold
    model.iou = iou_threshold    # IoU threshold
    return model
