import torch

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad())

def calculate_accuracy(predictions: torch.FloatTensor, labels:torch.LongTensor)->torch.FloatTensor:
    assert len(predictions.shape) == 2, f'predictions tensor must be 2-D, got{predictions.shape}'
    assert len(labels.shape) == 1, f'labels tensor must be 1-D, got {labels.shape}'
    batch_size, n_classes = predictions.shape
    assert batch_size == labels.shape[0], f'predictions {predictions.shape} and labels {labels.shape} shape mismatch'
    top_predictions = predictions.argmax(dim=1)
    correct = top_predictions.eq(labels).sum()
    accuracy = correct/batch_size
    return accuracy
