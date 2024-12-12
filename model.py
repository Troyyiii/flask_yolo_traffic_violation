import torch
import pathlib

def load_model(path):
    pathlib.PosixPath = pathlib.WindowsPath
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("./yolov5", "custom", path=path, source="local", force_reload=True)
    model.to(device)
    
    print(f"Model is running on: {device}")
    
    return model