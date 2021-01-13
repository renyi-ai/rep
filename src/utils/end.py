import pickle
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

from src.bin import device

def get_transformation():
    trans = torch.from_numpy
    return trans

def load_activations(filepath):
    return pickle.load(open(filepath,'rb'))

def run_model_on_data(model, data, batch_size=32, verbose=True, with_grads=False, data_labels=None):
    results = []
    grads = []

    trans = get_transformation()
    iterator = range(0,len(data),batch_size)
    iterator = tqdm(iterator) if verbose else iterator
    for i in iterator:
        start = i
        end = min(i+batch_size, len(data))
        x = [trans(x) for x in data[start:end]]
        x = torch.stack(x, 0).to(device)
        if with_grads:
            x.requires_grad = True
        out = model(x)
        if with_grads:
            loss = torch.matmul(out, torch.nn.functional.one_hot(torch.tensor(data_labels[start:end]), 10).float().transpose(0,1).to(device))
            loss.backward()
            grads.extend(list(x.grad.detach().cpu().numpy()))
        batch_results = list(out.detach().cpu().numpy())
        results.extend(batch_results)
        model.zero_grad()
    results = np.array(results)

    if with_grads:
        grads = np.array(grads)
        return results, grads

    return results
