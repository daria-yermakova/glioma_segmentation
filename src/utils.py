import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Using cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print('Using mps')
    else:
        device = torch.device('cpu')
        print('Using cpu')
    return device

def IoU(target, predicted_mask):
    """
    Args:
        target: (torch.tensor (batchxCxHxW)) Binary Target Segmentation from training set
        predicted_mask: (torch.tensor (batchxCxHxW)) Predicted Segmentation Mask

    Returns:
        IoU: (Float) Average IoUs over Batch
    """

    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(target == 0, predicted_mask == 1)).sum()
    false_n = (torch.logical_and(target == 1, predicted_mask == 0)).sum()

    sample_IoU = (smooth+float(true_p))/(float(true_p) + float(false_p)+float(false_n)+smooth)
    sample_IoU2 = (float(true_p))/(float(true_p) + float(false_p)+float(false_n)+smooth)

    print(f'iou - tp={true_p} fp={false_p} fn={false_n} iou1={sample_IoU} iou2={sample_IoU2}')

    return sample_IoU

def IoU2(target, predicted_mask):
    """
    Args:
        target: (torch.tensor (batchxCxHxW)) Binary Target Segmentation from training set
        predicted_mask: (torch.tensor (batchxCxHxW)) Predicted Segmentation Mask

    Returns:
        IoU: (Float) Average IoUs over Batch
    """

    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(target == 0, predicted_mask == 1)).sum()
    false_n = (torch.logical_and(target == 1, predicted_mask == 0)).sum()

    sample_IoU = (smooth+float(true_p))/(float(true_p) + float(false_p)+float(false_n)+smooth)
    sample_IoU2 = (float(true_p))/(float(true_p) + float(false_p)+float(false_n)+smooth)

    print(f'iou - tp={true_p} fp={false_p} fn={false_n} iou1={sample_IoU} iou2={sample_IoU2}')

    return sample_IoU2

def compute_average_pos_weight(train_loader):
    """
    Compute the average pos_weight for binary cross-entropy loss based on the target labels in the training set.

    Args:
    - train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.

    Returns:
    - average_pos_weight (torch.Tensor): Scalar tensor representing the average pos_weight.
    """
    num_positive_total = 0
    num_negative_total = 0

    # Iterate over batches in the training set
    for batch in train_loader:
        inputs, targets, _ = batch

        # Count the number of positive and negative samples in the batch
        num_positive_batch = torch.sum(targets == 1).item()
        num_negative_batch = torch.sum(targets == 0).item()

        # Accumulate counts for the entire training set
        num_positive_total += num_positive_batch
        num_negative_total += num_negative_batch

    # Compute the overall ratio of negative to positive samples
    overall_ratio = num_negative_total / num_positive_total

    # Compute the average pos_weight as the inverse of the overall ratio
    average_pos_weight = torch.tensor([1 / overall_ratio])

    return average_pos_weight

# res = compute_average_pos_weight(train_dataloader)
# print(res) - 0.0015