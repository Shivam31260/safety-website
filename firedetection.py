import torch
import os

# Path to YOLOv5 model weights
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(STATIC_FOLDER, 'firedetection.pt')
ogmodel = os.path.join(STATIC_FOLDER, 'yolov5')

# Global variable to cache the loaded model
_cached_model = None

def fire_detection_model(conf_threshold=0.6, iou_threshold=0.45):
    """
    Returns the cached YOLOv5 model if already loaded, or loads it for the first time.
    """
    global _cached_model
    
    # If the model is not cached yet, load it from the .pt file
    if _cached_model is None:
        _cached_model = torch.hub.load(ogmodel, 'custom', path=MODEL_PATH, force_reload=False, source='local')
        _cached_model.eval()  # Set the model to evaluation mode
        print("Model loaded for the first time.")
    else:
        print("Using cached model.")
    
    # Set thresholds for confidence and IoU
    _cached_model.conf = conf_threshold
    _cached_model.iou = iou_threshold
    
    return _cached_model
