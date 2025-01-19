import torch
import os

# Use an environment variable or a relative path for compatibility
MODEL_PATH = os.getenv('HELMET_HEAD_MODEL_PATH', './static/best_pt_files/helmet_head.pt')

def helmet_head_model(conf_threshold=0.6, iou_threshold=0.45):
    """
    Load the YOLOv5 helmet and head detection model with the specified settings.
    :param conf_threshold: Confidence threshold for detection.
    :param iou_threshold: Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
    :return: Loaded YOLOv5 model.
    """
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        # Load the YOLOv5 model from ultralytics hub with the custom weights
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model.eval()  # Set model to evaluation mode
        model.conf = conf_threshold  # Set confidence threshold
        model.iou = iou_threshold    # Set IoU threshold
        print(f"Helmet and head detection model loaded successfully with conf={conf_threshold}, iou={iou_threshold}")
        return model
    except Exception as e:
        print(f"Error loading helmet and head detection model: {e}")
        raise e
