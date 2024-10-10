import torch
import pickle
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np 
import zipfile 
import os 
import shutil 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed(seq:Seq, model, mean_pool:bool=False, direction:str='+') -> torch.Tensor:
    # Nucleotides NEED to be lower case. Also, <+> or <-> indicates strand.
    seq = f'<{args.direction}>{str(record.seq).lower()}'
    encodings = tokenizer([seq], return_tensors='pt')
    with torch.no_grad():
        embedding = model(encodings.input_ids.to(device), output_hidden_states=True).last_hidden_state
    embedding = embedding.cpu()# .numpy()

    if mean_pool:
        embedding = torch.ravel(torch.mean(embeddings, axis=0))
        assert len(embedding) == 1280, 'embed: The mean-pooled embeddings are the wrong shape.'
    
    return embeddings


def get_memory_usage(input_path:str, emb_dim:int=1280, half_precision:bool=False, mean_pool:bool=False): # , dtype=torch.float32):
    '''Estimate the amount of memory which will be taken up by the embeddings of the sequences in the file.
    Returns the size in bytes.'''
    
    n = 0
    for record in SeqIO.parse(args.input_path, 'fasta'):
        n += len(str(record.seq)) if (not mean_pool) else 1
    # print(f'get_memory_usage: Number of embedded items is {n}.')
    itemsize = 2 if half_precision else 4 # Get the size of the placeholder in bytes. 
    return itemsize * n * emb_dim
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='tattabio/gLM2_650M')
    # https://github.com/TattaBio/gLM2
    parser.add_argument('--input-path', '-i', type=str, default=None)
    parser.add_argument('--output-dir', '-d', type=str, default=None)
    parser.add_argument('--direction', default='+', choices=['+', '-'], type=str)
    # parser.add_argument('--dtype', choices=['float16', 'float32'], type=str)
    parser.add_argument('--half-precision', action='store_true')
    parser.add_argument('--mean-pool', action='store_true')

    args = parser.parse_args()

    # torch_dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    torch_dtype = torch.bfloat16 if args.half_precision else torch.float32
    file_name, _ = os.path.splitext(os.path.basename(args.input_path))
    dir_name = os.path.dirname(args.input_path)
    # Use the same output directory as the specified input directory default.
    output_dir = args.output_dir if (args.output_dir is not None) else dir_name

    # model_name = 'tattabio/gLM2_650M'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    mem = get_memory_usage(args.input_path, half_precision=args.half_precision, mean_pool=args.mean_pool)
    print('Predicted memory usage of embeddings:', np.round(mem / 1e9, 2), 'GB')

    # If we are mean-pooling, memory is less of a concern, and we can keep everything in memory at once. 
    if args.mean_pool: 
        embs = dict()
        for record in tqdm(SeqIO.parse(args.input_path, 'fasta'), desc='Embedding sequences...'):
            embs[record.id] = embed(record.seq, model, mean_pool=args.mean_pool, direction=args.direction)

        output_path = os.path.join(output_dir, file_name + f'_{args.model_name.split('/')[-1]}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(embs, f)
    
    # If we are not mean-pooling, we will store eveything as a zip archive. 
    else:
        output_path = os.path.join(output_dir, file_name + f'_{args.model_name.split('/')[-1]}.zip')
        print(f'Creating zipped archive at {output_path}')
        with zipfile.ZipFile(output_path, 'w') as zf:

            for record in tqdm(SeqIO.parse(args.input_path, 'fasta'), desc='Embedding sequences...'):
                emb = embed(record.seq, model, mean_pool=args.mean_pool, direction=args.direction) # Output of this is a tensor. 
                
                tmp_file_path = os.path.join(output_dir, f'{record.id}.pt')
                # np.savetxt(tmp_file_path, emb)
                torch.save(emb, tmp_file_path)

                zf.write(tmp_file_path) # Add the temporary file to the zip archive. 
                os.remove(tmp_file_path) # Remove the temporary file. 

    print(f'Embeddings written to {output_path}')