import torch
import pathlib

def load_model(path):
    pathlib.PosixPath = pathlib.WindowsPath
    
    model = torch.hub.load("./yolov5", "custom", path=path, source="local", force_reload=True)
    
    return model