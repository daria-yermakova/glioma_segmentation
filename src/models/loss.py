import torch
import torch.nn.functional as F

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Compute the binary cross-entropy loss between logits and targets.

        Args:
        - logits (torch.Tensor): Tensor of shape (1, 1, 64, 64) containing the logits.
        - targets (torch.Tensor): Tensor of shape (1, 1, 64, 64) containing the target probabilities.

        Returns:
        - loss (torch.Tensor): Scalar tensor representing the cross-entropy loss.
        """
        # Flatten the logits and targets tensors
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        # Compute the binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat)

        return loss

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.

        # Resize input to match the target size
        input = torch.nn.functional.interpolate(input, size=target.size()[2:], mode='nearest')

        iflat = input.view(-1)
        tflat = target.view(-1)

        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def PSNRLoss(batch_1, batch_2):
    """peak signal-to-noise ratio loss"""
    mse = torch.nn.MSELoss()
    mse_loss = mse(batch_1, batch_2)
    psnr = 10 * torch.log10(1 / mse_loss)
    return psnr