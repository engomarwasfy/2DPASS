import numpy as np

__all__ = ['cosine_schedule_with_warmup']


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size, num_gpu):
    batch_size *= num_gpu

    warmup_iters = 0 if num_gpu == 1 else 1000 // num_gpu
    if k < warmup_iters:
        return (k + 1) / warmup_iters
    iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
    return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) / (num_epochs * iter_per_epoch)))



