import torch
import pickle
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed(seq, model):
    encodings = tokenizer([sequence], return_tensors='pt')
    with torch.no_grad():
        embedding = model(encodings.input_ids.to(device), output_hidden_states=True).last_hidden_state
    return embedding.cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='tattabio/gLM2_650M')
    parser.add_argument('--input-path', '-i', type=str, default=None)
    parser.add_argument('--output-path', '-o', type=str, default=None)
    parser.add_argument('--direction', default='+', choices=['+', '-'], type=str)

    args = parser.parse_args()

    # model_name = 'tattabio/gLM2_650M'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).cuda()

    seq_dict = dict()
    for record in SeqIO.parse(args.input_path, 'fasta'):
        seq_dict[record.id] = f'<{args.direction}>{str(record.seq).lower()}'

    embs = {}
    for id_, seq in tqdm(seq_dict.items(), desc='Embedding sequences...'):
        embs[id_] = embed(seq, model)

    with open(args.output_path, 'wb') as file:
        pickle.dump(embs, file)

    print(f'Embeddings written to {args.output_path}')