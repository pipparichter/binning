
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd 
import numpy as np 
from typing import List, NoReturn

class File():

    def __init__(self, path:str):

        if path is not None:
            self.path = path 
            self.dir_name, self.file_name = os.path.split(path) 
        # self.data = None # This will be populated with a DataFrame in child classes. 
        self.genome_id = None # This will be populated with the genome ID extracted from the filename for everything but the MetadataFile class.



class FastaFile(File):

    def __init__(self, path:str=None, seqs:List[str]=None, ids:List[str]=None, descriptions:List[str]=None):
        '''Initialize a FastaFile object.'''
        super().__init__(path) 

        if (path is not None):
            f = open(path, 'r')
            self.seqs, self.ids, self.descriptions = [], [], []
            for record in SeqIO.parse(path, 'fasta'):
                self.ids.append(record.id)
                self.descriptions.append(record.description.replace(record.id, '').strip())
                self.seqs.append(str(record.seq))
            f.close()
        else:
            self.seqs, self.ids, self.descriptions = seqs, ids, descriptions

    def __len__(self):
        return len(self.seqs)

    @classmethod
    def from_df(cls, df:pd.DataFrame, include_cols:List[str]=None):
        ids = df.index.values.tolist()
        seqs = df.seq.values.tolist()

        include_cols = df.columns if (include_cols is None) else include_cols
        cols = [col for col in df.columns if (col != 'seq') and (col in include_cols)]
        descriptions = []
        for row in df[include_cols].itertuples():
            # Sometimes there are equal signs in the descriptions, which mess everything up... 
            description = {col:getattr(row, col) for col in include_cols if (getattr(row, col, None) is not None)}
            description = {col:value.replace('=', '').strip() for col, value in description.items() if (type(value) == str)}
            description = ';'.join([f'{col}={value}' for col, value in description.items()])
            descriptions.append(description)
        return cls(ids=ids, seqs=seqs, descriptions=descriptions)
            
    def to_df(self, parse_description:bool=True) -> pd.DataFrame:
        '''Load a FASTA file in as a pandas DataFrame. If the FASTA file is for a particular genome, then 
        add the genome ID as an additional column.'''

        def parse(description:str) -> dict:
            '''Descriptions should be of the form >col=val;col=val. '''
            # Headers are of the form >col=value;...;col=value
            return dict([entry.split('=') for entry in description.split(';')])
        
        df = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            row = {'description':description} if (not parse_description) else parse(description)
            row['id'] = id_
            row['seq'] = seq 
            df.append(row)

        return pd.DataFrame(df).set_index('id')

    def write(self, path:str) -> NoReturn:
        f = open(path, 'w')
        records = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            record = SeqRecord(Seq(seq), id=id_, description=description)
            records.append(record)
        SeqIO.write(records, f, 'fasta')
        f.close()
