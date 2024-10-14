'''A script for taking a zip file generated by the embed.py script and mean-pooling the embeddings.'''
import argparse
import zipfile 
import io
import torch 
from tqdm import tqdm
import pickle
import os 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    dir_name = os.path.dirname(args.input_path)
    # Use the same output directory as the specified input directory default.
    output_dir = args.output_dir if (args.output_dir is not None) else dir_name

    embeddings = []
    with zipfile.ZipFile(args.input_path, 'r') as f:
        for name in tqdm(f.namelist(), desc='Mean-pooling embeddings...'):
            info = dict()
            # The name of the file is the contig ID. 
            info['id'] = os.path.basename(name.replace('.pt', '')) # Remove the file extension and the path stuff.  
            emb = f.read(name) # Read the contents of the file in the archive. 
            emb = torch.load(io.BytesIO(emb), weights_only=False)
            info['length'] = len(emb) # Number of elements on the genome axis should correspond to the contig length. 
            info['embedding'] = torch.ravel(torch.mean(emb, axis=1))
            # assert len(emb) == 1280, f'embed: The mean-pooled embeddings are the wrong shape. Shape is {emb.shape}.'
            embeddings.append(info)

    # Replace the file extension. 
    output_path = args.input_path.replace('.zip', '') + '.pkl'
    output_path = os.path.join(output_dir, output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f'Mean-pooled embeddings written to {output_path}')
        