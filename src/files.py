
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd 
import numpy as np 
from typing import List, NoReturn
from collections import namedtuple
import copy
import re


FastaEntry = namedtuple('FastaEntry', ['id_', 'seq', 'description'])

class FastaFile():
    genome_id_pattern = re.compile(r'GC[AF]_\d{9}\.\d{1}')

    def __init__(self, path:str, nrows:int=None):
        '''Initialize a FastaFile object.'''

        f = open(path, 'r')
        self.seqs, self.ids, self.descriptions = [], [], []

        for record in SeqIO.parse(path, 'fasta'):
            self.ids.append(record.id)
            self.descriptions.append(record.description.replace(record.id, '').strip())
            self.seqs.append(str(record.seq))

            # Only load a specified number of rows, if specified. 
            if nrows and (len(self.ids) == nrows):
                break

        
        try: # Try to extract genome IDs from the contig IDs. 
            self.genome_ids = [re.match(FastaFile.genome_id_pattern, id_).group(0) for id_ in self.ids]
        except:
            self.genome_ids = None

        self.i = 0

    def __len__(self):
        return len(self.seqs)

    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self):
            raise StopIteration
        else:
            entry = FastaEntry(self.ids[self.i], self.seqs[self.i], self.descriptions[self.i])
            self.i += 1
            return entry

    def __len__(self):
        return len(self.seqs)

    def subset(self, genome_ids:List[str]):
        '''Grab a subset of the file corresponding to the specified genome IDs.'''
        idxs = np.isin(self.genome_ids, genome_ids)

        self.ids = np.array(self.ids)[idxs].tolist()
        self.descriptions = np.array(self.descriptions)[idxs].tolist()
        self.seqs = np.array(self.seqs)[idxs].tolist()
        self.genome_ids = np.array(self.genome_ids)[idxs].tolist()


    def write(self, path:str):
        records = []

        for entry in self:
            record = SeqRecord(Seq(entry.seq), id=entry.id_, description=entry.description, name='')
            records.append(record)
        # Write the records to the specified output file. 
        with open(path, 'w') as f:
            SeqIO.write(records, f, 'fasta')


        
