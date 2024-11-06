import sys 
import os 
cwd = os.getcwd()
print('Current working directory is:', cwd)
sys.path.append(os.path.join(cwd, 'src'))
sys.path.append(os.path.join(cwd, '../src'))


import torch
from tqdm import tqdm
import argparse
import numpy as np 
import zipfile 
from files import FastaFile
from emb import *
import shutil 

# Trying to fix some memory issues. See https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['kmer', 'glm'], help='The type of embedding to generate.', default='plm')
    parser.add_argument('--input-path', '-i', type=str, default=None)
    parser.add_argument('--direction', default='+', choices=['+', '-'], type=str)
    args = parser.parse_args()

    file_name, _ = os.path.splitext(os.path.basename(args.input_path)) 
    file_name += f'_{args.type}.zip' # Add the embedding type and zip extension. 
    dir_name = os.path.dirname(args.input_path)
    output_path = os.path.join(dir_name, file_name)

    embedder = GlmEmbedder(direction=args.direction) if (args.type == 'glm') else KmerEmbedder(direction=args.direction)
    file = FastaFile(args.input_path)
    embedder(file, path=output_path)


# def get_memory_usage(input_path:str, emb_dim:int=1280, half_precision:bool=False, mean_pool:bool=False): # , dtype=torch.float32):
#     '''Estimate the amount of memory which will be taken up by the embeddings of the sequences in the file.
#     Returns the size in bytes.'''
    
#     n = 0
#     for record in SeqIO.parse(args.input_path, 'fasta'):
#         n += len(str(record.seq)) if (not mean_pool) else 1
#     # print(f'get_memory_usage: Number of embedded items is {n}.')
#     itemsize = 2 if half_precision else 4 # Get the size of the placeholder in bytes. 
#     return itemsize * n * emb_dim
    