import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, weight=None, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, pred, ground_truth):
        """
        Args:
            pred: (batch_size, num_classes, height, width) - mask of the model
            ground_truth: (batch_size, height, width) - ground truth labels
        """
        num_classes = pred.size(1)

        # Apply softmax to pred to get probabilities
        probs = F.softmax(pred, dim=1)

        # One-hot encode the ground truth labels
        true_1_hot = torch.eye(num_classes).to(pred.device)[ground_truth.type(torch.long)].permute(0, 3, 1, 2).float().to(pred.device)

        if self.ignore_index is not None:
            mask = (ground_truth != self.ignore_index)
            true_1_hot = true_1_hot * mask.unsqueeze(1)
            probs = probs * mask.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weight is not None:
            dice_score = dice_score * self.weight

        return 1. - dice_score.mean()


class ClsLoss(nn.Module):
    def __init__(self, ignore_index=-100, weight=None, dice_weight=1., cross_entropy_weight=1., smooth=1.):
        super(ClsLoss, self).__init__()
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_loss = DiceLoss(ignore_index=ignore_index, weight=weight, smooth=smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred, true):
        dice_loss = self.dice_loss(pred, true)
        cross_entropy_loss = self.cross_entropy_loss(pred, true)
        cls_loss = self.dice_weight * dice_loss + self.cross_entropy_weight * cross_entropy_loss
        return cls_loss


# if __name__ == '__main__':
#     model_output = torch.randn(2, 4, 512, 512).cuda()  # Example model output
#     ground_truth = torch.ones(2, 512, 512).type(torch.long).cuda()  # Example ground truth
#     criterion = ClsLoss(1, torch.FloatTensor([1, 1, 2, 1]).cuda()).cuda()
#     loss = criterion(model_output, ground_truth)
