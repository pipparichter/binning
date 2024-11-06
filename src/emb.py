import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np 
import zipfile 
import os 
import shutil 
from files import FastaFile
import itertools


class Embedder():

    def __init__(self):
        pass 

    def __call__(self, file:FastaFile, path:str=None):

        skipped_due_to_memory = 0
        output_dir = os.path.dirname(path)

        with zipfile.ZipFile(path, 'w') as zf:
            for entry in tqdm(file, desc='Embedder.__call__: Embedding sequences...'):
                emb = self.embed(entry.seq) # Output of this is a tensor. 
                if emb is None: # If there's an out-of-memory error, skip the sequence and continue. 
                    skipped_due_to_memory += 1
                    continue
                
                tmp_file_path = os.path.join(output_dir, f'{entry.id_}.pt')
                torch.save(emb, tmp_file_path)

                zf.write(tmp_file_path) # Add the temporary file to the zip archive. 
                os.remove(tmp_file_path) # Remove the temporary file. 
        
        if skipped_due_to_memory > 0:
            print(f'Embedder.__call__: Skipped {skipped_due_to_memory} sequences due to an OutOfMemoryError.')
        print(f'Embedder.__call__: Embeddings written to {path}')



class GlmEmbedder(Embedder):
    # https://github.com/TattaBio/gLM2

    def __init__(self, model_name:str='tattabio/gLM2_650M', direction='+', half_precision:bool=True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 if args.half_precision else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype, trust_remote_code=True).to(self.device)
        self.direction = direction


    def embed(self, seq:str) -> torch.Tensor:

        def tokenize(seq:str) -> str:
            # Nucleotides NEED to be lower case. Also, <+> or <-> indicates strand.
            seq = f'<{self.direction}>{str(seq).lower()}'
            encodings = self.tokenizer([seq], return_tensors='pt').to(self.device)
            return encodings.input_ids

        try:
            with torch.no_grad():
                embedding = self.model(tokenize(seq), output_hidden_states=True).last_hidden_state
            embedding = embedding.cpu()
            return embedding

        except torch.OutOfMemoryError:
            return None

 
class KmerEmbedder(Embedder):
    alphabet = ['A', 'C', 'T', 'G', 'N']

    def __init__(self, k:int=5, direction='+', half_precision:bool=False):
        super().__init__()
        self.dtype = torch.bfloat16 if half_precision else torch.float32
        self.direction = direction
        self.k = k
        # Ensure the k-mers are ordered consistently. 
        self.kmers = sorted(list(itertools.product(KmerEmbedder.alphabet, repeat=k)))
        self.kmers = [''.join(kmer) for kmer in self.kmers] # Code above spits out tuples. 

    @staticmethod
    def reverse(seq:str) -> str:
        seq = Seq(seq) 
        seq = seq.reverse_complement()
        return str(seq)

    def embed(self, seq:str) -> torch.Tensor:
        # If specified, get the reverse complement. 
        seq = KmerEmbedder.reverse(seq) if (self.direction == '-') else seq

        kmer_counts = {kmer:0 for kmer in self.kmers}
        # Iterate over the sequence in a sliding window. 
        i = 0
        while i + self.k < len(seq):
            kmer = seq[i:i + self.k]
            kmer_counts[kmer] += 1
            i += 1
        embedding = torch.Tensor([kmer_counts[kmer] for kmer in self.kmers])
        return embedding.to(self.dtype)
