import torch

import numpy as np
from scipy.spatial import procrustes

def align_pred_target(predicted, target):
    """
    Align the predicted and target poses using Procrustes analysis to remove translation and rotation.
    Assumes predicted and target have shape (num_joints, 3) for each frame.
    """
    predicted_np = predicted.cpu().numpy()  
    target_np = target.cpu().numpy()        

    _, aligned_pred, _ = procrustes(target_np, predicted_np)

    aligned_pred = torch.tensor(aligned_pred, dtype=torch.float32).to(predicted.device)
    return aligned_pred

def loss_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    
    aligned_pred = align_pred_target(predicted, target)
    
    distance = torch.norm(aligned_pred - target, dim=-1)
    
    return torch.mean(distance)
