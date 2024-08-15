'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def recall_score(output, target, average='macro'):
    """Computes the recall score"""
    preds = output.argmax(dim=1)
    true_positive = torch.zeros(output.size(1)).to(output.device)
    false_negative = torch.zeros(output.size(1)).to(output.device)
    
    for i in range(len(preds)):
        if preds[i] == target[i]:
            true_positive[target[i]] += 1
        else:
            false_negative[target[i]] += 1

    # 避免除零
    valid = true_positive + false_negative != 0
    recall = true_positive[valid] / (true_positive[valid] + false_negative[valid])
    if average == 'macro':
        recall = recall.mean()  # macro average
    elif average == 'micro':
        recall = recall.sum() / (true_positive + false_negative).sum()  # micro average

    # 处理可能的NaN值
    if recall.isnan().any():
        recall[recall.isnan()] = 0

    return recall * 100  # convert to percentage

def f1_score(output, target, average='macro'):
    """Computes the F1 score"""
    preds = output.argmax(dim=1)
    n_classes = output.size(1)
    true_positive = torch.zeros(n_classes, device=output.device)
    false_positive = torch.zeros(n_classes, device=output.device)
    false_negative = torch.zeros(n_classes, device=output.device)
    for i in range(n_classes):
        true_positive[i] = ((preds == i) & (target == i)).sum().item()
        false_positive[i] = ((preds == i) & (target != i)).sum().item()
        false_negative[i] = ((preds != i) & (target == i)).sum().item()
    
    # Prevent division by zero and calculate precision and recall
    precision = true_positive / (true_positive + false_positive + 1e-8)
    # print(precision)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    # print(recall)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)  # Remove NaNs

    if average == 'macro':
        precision = precision.mean()
        recall = recall.mean()
        f1 = f1.mean()
    elif average == 'micro':
        total_tp = true_positive.sum()
        total_fp = false_positive.sum()
        total_fn = false_negative.sum()
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision * 100, recall * 100, f1 * 100  # convert to percentage
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
