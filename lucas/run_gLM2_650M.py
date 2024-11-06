import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import random
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
from huggingface_hub import login
import os

# login(token='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')
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
model_name = 'tattabio/gLM2_650M'
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
pretrained_model = AutoModel.from_pretrained(model_name,trust_remote_code=True).cuda()

# add a new token to the tokenizer
next_token = '<next>'

# create the finetuning dataset
# for overlap in ['overlap0','overlap100','overlap200','overlap300','overlap400','overlap500']:
overlap = 'overlap0'
dataset = load_dataset('bioLLM/evo_ecoli_len1000',split=overlap)
seqs = dataset['sequence'][:-1]
seqs = [f'<+>{seq.lower()}' for seq in seqs]

positive_label = 1
positive_pairs = []
for i in range(1,len(seqs)):
    positive_pairs.append( (f'{seqs[i-1]}{next_token}{seqs[i]}',positive_label) )

idxs = []
for _ in range(len(seqs)-1):
    while True:
        tup = tuple(random.randint(0, len(seqs)-1) for _ in range(2))
        if abs(tup[0] - tup[1]) > 1 and tup[0] != tup[1]:
            idxs.append(tup)
            break

negative_label = 0
negative_pairs = [ (f'{seqs[idx[0]]}{next_token}{seqs[idx[1]]}',negative_label) for idx in idxs]

assert len(positive_pairs) == len(negative_pairs)
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

new_tokens = [next_token]
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

new_tokens = [next_token, positive_label, negative_label]
new_vocab_size = pretrained_model.tok_embeddings.num_embeddings + len(new_tokens)
new_embeddings = torch.nn.Embedding(new_vocab_size, pretrained_model.tok_embeddings.embedding_dim)
new_embeddings.weight.data[:pretrained_model.tok_embeddings.num_embeddings] = pretrained_model.tok_embeddings.weight.data
pretrained_model.tok_embeddings = new_embeddings

tokenized_data = [tokenizer(pair[0]) for pair in all_pairs]
test_dataset = Dataset.from_dict({
    'input_ids': [x['input_ids'] for x in tokenized_data],
    'token_type_ids': [x['token_type_ids'] for x in tokenized_data],
    'attention_mask': [x['attention_mask'] for x in tokenized_data],
    'label': [pair[1] for pair in all_pairs]
})
test_dataset = test_dataset.class_encode_column('label')
test_dataset = test_dataset.train_test_split(test_size=0.2,seed=0,stratify_by_column='label')

print(overlap, pd.DataFrame(test_dataset['test']['label']).value_counts())

accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(1280, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0]
        logits = self.classifier(pooled_output)       
        loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

model = ClassificationModel(pretrained_model=pretrained_model, num_labels=2).cuda()

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.01,
    run_name=overlap+model_name.split('/')[-1]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=test_dataset['train'],  
    eval_dataset=test_dataset['test'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()