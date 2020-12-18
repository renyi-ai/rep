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

def run_model_on_data(model, data, batch_size=32, verbose=True):
    results = []
    trans = get_transformation()
    iterator = range(0,len(data),batch_size)
    iterator = tqdm(iterator) if verbose else iterator
    for i in iterator:
        start = i
        end = min(i+batch_size, len(data))
        x = torch.from_numpy(data[start:end])
        x = x.to(device)
        batch_results = list(model(x).detach().cpu().numpy())
        results.extend(batch_results)
    results = np.array(results)
    return results
