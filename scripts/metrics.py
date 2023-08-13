import torch

def find_metrics(inputs, targets, batch_size = 32):
    # Micro precision and recall takes and calculates everything in case
    # of the globally does not takes anything in account.
    arg_maxed = torch.argmax(inputs, axis = 1)
    input_zeros = torch.zeros(inputs.shape[0], 30).cuda()
    input_zeros[torch.arange(inputs.shape[0]), arg_maxed] = 1
    tp = torch.sum(torch.logical_and(input_zeros == 1, targets == 1))
    fp = torch.sum(torch.logical_and(input_zeros == 0, targets == 1))
    fn = torch.sum(torch.logical_and(input_zeros == 1, targets == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

    # Macro average precision sum over all class individual precision and average by number of classes.


def find_metrics_macro(inputs, targets, batch_size = 32):
    # Micro precision and recall takes and calculates everything in case
    # of the globally does not takes anything in account.
    arg_maxed = torch.argmax(inputs, axis = 1)
    input_zeros = torch.zeros(inputs.shape[0], 30).cuda()
    input_zeros[torch.arange(inputs.shape[0]), arg_maxed] = 1
    tps = torch.sum(torch.logical_and(input_zeros == 1, targets == 1), axis = 0)
    fps = torch.sum(torch.logical_and(input_zeros == 0, targets == 1), axis = 0)
    fns = torch.sum(torch.logical_and(input_zeros == 1, targets == 0), axis = 0)
    # return tps, fps, fns
    precision = torch.sum(torch.nan_to_num(tps / (tps + fps))) / inputs.shape[1]
    recall = torch.sum(torch.nan_to_num(tps / (tps + fns))) / inputs.shape[1]
    return precision, recall