import torch

# Path to YOLOv5 model weights
MODEL_PATH = r"C:\Users\DELL\Desktop\FInal Project\DetectionSystem\best.pt files\facemask.pt"

def facemask_model(conf_threshold=0.6, iou_threshold=0.45):
    """
    Loads the YOLOv5 model with the specified settings.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model.eval()
    model.conf = conf_threshold  # confidence threshold
    model.iou = iou_threshold    # IoU threshold
    return model
