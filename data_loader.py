from datasets import load_dataset
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
import torch


def load_tinystories_dataset():
    """Load the TinyStories dataset."""
    return load_dataset("roneneldan/TinyStories")


def process_example(example, encoder):
    """Map each story to its tokens and length of tokens."""
    ids = encoder.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}


def tokenize_and_save_dataset(dataset_name="train.bin"):
    """Tokenize dataset and save to binary files."""
    ds = load_tinystories_dataset()
    enc = tiktoken.get_encoding("gpt2")
    
    if not os.path.exists(dataset_name):
        tokenized = ds.map(
            lambda example: process_example(example, enc),
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )
        
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16
            
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024
            
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()


def get_batch(split, batch_size, block_size, device_type, device):
    """Get a batch of data for training or validation."""
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y