
#basic
import functools
import numpy as np
from time import time
import _pickle as pkl
import json
from typing import List, Dict, Tuple
from datetime import datetime
import pytz
from tqdm import tqdm
import os
import sys
import requests
import random
import zipfile
import hashlib

#hotam
from hotam import get_logger

#torch
from torch import is_tensor
import torch

logger = get_logger(__name__)


def ensure_flat(item, mask=None):

    if not isinstance(item[0], np.int):
        item = item.flatten()

        # Ugly fix for 2d arrays with differnet lengths
        if not isinstance(item[0], np.int):
            item = np.hstack(item)
    
    if mask is not None:
        mask = ensure_flat(ensure_numpy(mask)).astype(bool)
        if mask.shape == item.shape:
            item = item[mask]
    

    return item


def zero_pad(a):
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))])
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b

def string_pad(a, dtype="<U30"):
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))]).astype(dtype)
    b[:] = ""
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b


def ensure_numpy(item):

    if torch.is_tensor(item):
        item = item.cpu().detach().numpy()

    if type(item) is not np.ndarray:
        item = np.array(item)

    return item

def to_tensor(item, dtype=torch.float):
    try:
        return torch.tensor(item, dtype=dtype)
    except ValueError as e:
        return item
    except TypeError as e:
        return item



def check_gpu(self, gpu:int, verbose=1) -> Tuple[bool, torch.device]:
    """checks if there is a gpu device available

    Parameters
    ----------
    gpu : int
        gpu to check for 
    verbose : int, optional
        print information or not, by default 1

    Returns
    -------
    Tuple[bool, torch.device]
        returns a bool for if the gpu is available or not and the device
    """
    gpu_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu}' if gpu_available and isinstance(gpu,int) else 'cpu')

    if verbose != 0:
        logger.info("----------------------")
        logger.info(f"nr GPUs found: {torch.cuda.device_count()}")

        if "cuda" in str(device):
            logger.info(f"Will use GPU number: {gpu}")
        else:
            logger.info("will use CPU")
        logger.info("----------------------")

    return gpu_available, device


class RangeDict(dict):
    def __getitem__(self,query_item):
        if type(query_item) == int:
            for k in self:
                if query_item >= k[0] and query_item <= k[1]:
                    return self[k]
            raise KeyError(query_item)
        else:
            return super().__getitem__(query_item)

    def get(self, query_item, default=None):
        try:
            return self.__getitem__(query_item)
        except KeyError:
            return default



def pickle_data(data, file_path):
    with open(file_path, "wb") as f:
        pkl.dump(data,f,  protocol=4)


def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data


def timer(func):

	@functools.wraps(func)
	def calc_time(*args, **kwargs):

		start = time()
		output = func(*args, **kwargs)
		end = time()
		logger.info("Time taken to run {}: {}".format(func, end-start))
		return output

	return calc_time


def one_tqdm(desc:str):
    def decorator(f):
        def wrapper(*args, **kwargs):
            pbar = tqdm(total=1, desc=desc)
            # try:
            f(*args, **kwargs)
            # except error as e:
            #     return e
            pbar.update(1)
            pbar.close()
        
        return wrapper
    
    return decorator


def get_timestamp():
    sweden = pytz.timezone('Europe/Stockholm')
    timestamp = datetime.now().astimezone(sweden).timestamp()
    return int(timestamp)


def copy_and_vet_dict(input_dict:dict):

    output_dict = {}
    for k,v in input_dict.items():

        if isinstance(k, int):
            k = str(k)

        if isinstance(v, (str, int, float, bool)):
            pass
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif not v:
            pass
        elif isinstance(v, np.float32):
            v = float(v)
        elif isinstance(v, dict):
            v = copy_and_vet_dict(v)
        elif isinstance(v, (list, tuple)):
            pass
        elif getattr(v, "name", None):
            att = getattr(v, "name", None)

            if callable(att):
                v = v.name()
            else:
                v = v.name
        else:
            raise ValueError(f'"{v}" of type {type(v)} is not a valid type')

        output_dict[k] = v
    
    return output_dict


def download(url:str, save_path:str, desc:str):

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length')) #response.iter_content(chunk_size=max(int(total/1000), 1024*1024)), 
    progress_bar = tqdm(total=total, unit='iB', unit_scale=True, desc=desc)

    if total is None:
        f.write(response.content)
        return

    with open(save_path, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total != 0 and progress_bar.n != total:
        raise RuntimeError("ERROR, something went wrong")


def unzip(zip_path:str, save_path:str):

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(save_path)


def dynamic_update(src, v, pad_value=0): 

    a = np.array(list(src.shape[1:]))
    b = np.array(list(v.shape))
    new_shape = np.maximum(a, b)
    new_src = np.full((src.shape[0]+1, *new_shape), pad_value, dtype=src.dtype)

    if len(v.shape) > 2:
        new_src[:src.shape[0],:src.shape[1], :src.shape[2]] = src
        new_src[src.shape[0],:v.shape[0], :v.shape[1]] = v
    #if len(v.shape) == 2:
    #    new_src[:src.shape[0],:src.shape[1], :] = src
    #    new_src[src.shape[0],:v.shape[0], :] = v
    else:
        new_src[:src.shape[0],:src.shape[1]] = src
        new_src[src.shape[0],:v.shape[0]] = v

    return new_src


def set_random_seed(nr, using_gpu:bool=False):
    random.seed(nr)
    np.random.seed(nr)
    torch.manual_seed(nr)

    if using_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tensor_dtype(numpy_dtype):

    if numpy_dtype == np.uint8:
        return torch.uint8

    if numpy_dtype == np.float or numpy_dtype == np.float32:
        return torch.float

    if numpy_dtype == np.int:
        return torch.long

    if numpy_dtype == np.bool:
        return torch.bool



def create_uid(string):
    uid = str(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 10**8)
    return uid
