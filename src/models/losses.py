import torch


def iou_score(
    pred_masks: torch.Tensor, target_masks: torch.Tensor, smooth: float = 1
) -> torch.Tensor:
    intersection = torch.sum(pred_masks * target_masks)
    union = torch.sum(pred_masks) + torch.sum(target_masks) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
