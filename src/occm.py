'''An implementation of the probabalistic "occupancy model" from "Occupancy Modeling, Maximum Contig Size Probabilities
and Designing Metagenomics Experiments"'''

# import networkx as nx 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord 
import Bio.SeqIO as SeqIO
from typing import List
from tqdm import tqdm
import numpy as np 
import sys 
import time 

class Read():

    def __init__(self, label:int, seq:str=None):

        self.seq = seq 
        self.bin_type = 'w' if (label > 0) else 't'
        self.next = [] 
        self.prev = []
        self.label = label

    def __len__(self):
        return len(self.seq)

    def is_terminal(self):
        return len(self.next) == 0

    def is_next(self, read):
        '''Determines whether or not the input read is immediately downstream of the prinicpal read on
        the original sequence.'''
        # If the bin types are the same, just check to see if the labels are consecutive numbers. 
        assert read.label != self.label, 'Read.is_next: Can\'t compare a read to itself.'
        return ((read.label - self.label) <= 1) and ((read.label - self.label) > 0)



class Contig():
    def __init__(self, read:Read):

        self.seq = read.seq
        self.curr_read = read
        self.overlap = len(read) // 2

    def __str__(self):
        return self.seq 
    
    def __len__(self):
        return len(self.seq)

    def extend(self, read):

        assert read in self.curr_read.next, 'Node.extend: Node must be a neighbor to join.'
        if read.bin_type == self.curr_read:
            self.seq = self.seq + read.seq 
        else:
            self.seq =  self.seq[:-self.overlap] + read.seq
        self.curr_read = read 

    def copy(self):
        new_contig = Contig(self.curr_read)
        new_contig.seq = self.seq 
        return new_contig


class Graph():
    def __init__(self, overlap:int=None):
        
        self.reads = dict()
        self.edges = [] 

    def add_edge(self, label:int, next_label:int):
        # Make a bi-directional connection between reads (nodes). 
        self.reads[label].next.append(self.reads[next_label])
        self.reads[next_label].prev.append(self.reads[label])
        self.edges.append((label, next_label))
    
    def add_read(self, read):
        self.reads[read.label] = read

    def longest_contig(self, start_read:Read) -> Contig:
        
        read = start_read
        contig = Contig(start_read)
        while not (read.is_terminal()):
            contig.extend(read.next[0])
            read = read.next[0]
        return contig

    def get_start_reads(self):
        '''Get all the reads in the graph which do not have any upstream neighboring reads.'''
        start_reads = []
        for label, read in self.reads.items():
            if len(read.prev) == 0:
                start_reads.append(read)
        return start_reads

    def get_terminal_reads(self):
        '''Get all reads in the graph which do not have any downstream neighboring reads.'''
        terminal_reads = []
        for label, read in self.reads.items():
            if len(read.next) == 0:
                terminal_reads.append(read)
        return terminal_reads

    def get_contigs(self) -> List[Contig]:

        contigs = []
        start_reads = self.get_start_reads()
        # print(f'Graph.get_contigs: Testing {len(start_reads)} paths through the graph.')
        # if np.all([start_read.is_terminal() for start_read in start_reads]): print('Graph.get_contigs: All start reads are terminal.')
        # print(f'Graph.get_contigs: {np.sum([start_read.is_terminal() for start_read in start_reads])} of the start reads are terminal.')

        # print(f'Graph.get_contigs: {len(self.get_terminal_reads())} terminal reads in the graph.')
        # NOTE: This is a bug, there should always be at least one. 
        assert len(start_reads) > 0, 'Graph.get_contigs: There are no start reads in the graph.'
        # for start_read in tqdm(start_reads, desc='Graph.get_contigs'):
        for start_read in start_reads:
            contigs.append(self.longest_contig(start_read))
        return contigs


class Sequence():

    def __init__(self, seq:str, read_size:int=300):
        self.seq = seq 
        self.read_size = read_size
        self.overlap = read_size // 2

        # How to handle the end cases? Perhaps just truncate the sequence?
        offset = read_size // 2
        self.n_bins_w = len(seq) // read_size
        self.n_bins_t = self.n_bins_w - 1
        self.bins_w = [seq[(i * read_size):((i + 1) * read_size)] for i in range(self.n_bins_w)] 
        self.bins_t = [seq[(i * read_size) + offset:((i + 1) * read_size) + offset] for i in range(self.n_bins_t)] 
        # print(f'Sequence.__init__: Generated {self.n_bins_w} Wendl-discretized bins and {self.n_bins_t} overlapping bins.')

        self.graph = None
    
    def __len__(self):
        return len(self.seq)

    def n_bins(self):
        return self.n_bins_w + self.n_bins_t

    def get_read(self, label:int):
        '''Get the bin section of the discretized sequence corresponding to the index.'''
        # assert label != 0, 'Sequence.get_bin: idx should not be zero.'
        seq = self.bins_t[int(label) - 1] if (int(label) < label) else self.bins_w[int(label) - 1]
        return Read(label, seq=seq)

    def get_coverage(self, contigs:List[Contig]):
        # Get the cumulative length of all contigs. 
        # Because contigs are generated using unique labels, there shouldn't be any duplicates. 
        return sum([len(contig) for contig in contigs]) / self.__len__()

    
    def build_graph(self, read_depth:int=10000):
        '''I think the cleanest way to do this is with a graph.'''
        # Sample possible bin indices. Shift the indices to represent read labels (rather than list indices)
        labels_w = np.arange(self.n_bins_w) + 1
        labels_t = np.arange(self.n_bins_t) + 1.5

        labels = np.random.choice(np.concat([labels_w, labels_t]), replace=True, size=read_depth)
        labels = np.unique(labels).tolist()
        labels = sorted(labels)

        # self.beta = len(labels) / self.n_bins()
        # print('Simulated beta:', self.beta)

        ti = time.perf_counter()
        graph = Graph()

        # Add all reads to the graph. 
        for label in labels:
            graph.add_read(self.get_read(label))
        # print(f'Sequence.build_graph: Added {len(graph.reads)} reads to the graph.')

        for i in range(len(labels) - 1):
            label = labels[i]
            # Grab the next two labels which are each candidates for consecutive labels. 
            # for next_label in labels[i + 1:i + 3]:
            next_label = labels[i + 1]
            if next_label - label <= 1:
                graph.add_edge(labels[i], labels[i + 1])

        # print(f'Sequence.build_graph: Added {len(graph.edges)} edges to the graph.')
        tf = time.perf_counter()
        # print(f'Sequence.build_graph: Graph with {len(graph.reads)} reads and {len(graph.edges)} edges constructed in {np.round(tf - ti)} seconds.')
        self.graph = graph

    def get_contigs(self):
        return self.graph.get_contigs()


class Genome():
    '''A Genome object is a collection of sequences which belong to a single organism.'''

    def __init__(self, path:str, read_size:int=200):

        self.seqs = []
        # Parse the FASTA file, storing each record as a separate sequence. 
        for record in SeqIO.parse(path, 'fasta'):
            self.seqs.append(Sequence(str(record.seq), read_size=read_size))
        self.n_seqs = len(self.seqs)
        print(f'Genome.__init__: Loaded {self.n_seqs} sequences from the FASTA file.')

    def __len__(self):
        '''Returns the total number of bases in the genome.'''
        return sum([len(seq) for seq in self.seqs])

    def build_graphs(self, read_depth:int=10000):
        # Want to partition the distribution of reads across sequences. 
        ratios = [len(seq) / len(self) for seq in self.seqs]
        for ratio, seq in zip(ratios, self.seqs):
            seq.build_graph(read_depth=int(ratio * read_depth))

    def get_contigs(self):
        contigs = []
        for seq in self.seqs:
            contigs += seq.get_contigs()
        return contigs

    def n_bins(self):
        return sum([seq.n_bins() for seq in self.seqs])


# def coverage(b:int=None, l:int=None, r:int=None):
#     '''Prediction of genome coverage based on the math I did and the assumptions of the occupancy model.
#     Genome coverage is the percentage of the genome which is mapped to a generated contig.'''
#     alpha = 2 * (b / l) - 1 # The total number of bins. 
#     beta = 1 - (1 - (1/alpha)) ** r # To get beta, need to account for both Wendl and overlapping bins. 

#     # n_bins_w = b // l
#     # n_bins_t = b // l - 1
#     n = b // l 

#     p1 = (2 * n - 2) / (n * (n - 1)) # Probability of overlap.
#     p2 = ((n - 2) * (n - 1)) / (n * (n - 1)) # Probability of no overlap. 
#     p3 = (n - 2) / (n * (n - 1))

#     # p1 = p1 - p3

#     print('coverage: p1 + p2 =', p1 + p2) # This quantity should equal 1. 
#     # coverage = n * beta * (l * (1 - p_overlap) + (l / 2) * p_overlap) + (n - 1) * beta * (l * (1 - p_overlap) + (l / 2) * p_overlap)
#     # coverage = (n * beta  + (n - 1) * beta) * (l * (1 - p_overlap) + (l / 2) * p_overlap)
#     coverage = beta * l * p2 * n + beta * (l/2) * p1 * n + beta * l * p2 * (n - 1) + beta * (l/2) * p1 * (n - 1)
#     # coverage -= beta * (l/2) * p3 * (n - 2)
#     return coverage / b # Return coverage as percent of the total genome. 



def get_beta(b:int=None, l:int=None, r:int=None):
    alpha = 2 * (b / l) - 1 # The total number of bins. 
    beta = 1 - (1 - (1/alpha)) ** r
    return beta

def pcgtk(k:int=None, b:int=None, l:int=None, r:int=None, verbose:bool=False):
    '''Implements the probability distribution P(C >= k) determined using the occupancy model. 
    P(C >= k) Gives the probability that the length of the longest obtained contig is greater than k. 

    :param k: The lower bound for longest contig size, in units bp. 
    :param b: The size of the genome, in units bp. 
    :param l: The size of reads, in units bp. 
    :param r: The read depth, in units of number of reads. 
    '''
    beta = get_beta(b=b, l=l, r=r)

    # thetaW is log{1/beta}((B/L - 1)(1 - beta) + 1). Use change-of-base to represent. 
    # thetaW is similar, just subtract an extra one. I wrote down the derivation in my notebook, but should review. 
    thetaW = np.log(((b / l) - 1) * (1 - beta) + 1) / np.log(1 / beta)
    thetaT = np.log(((b / l) - 2) * (1 - beta) + 1) / np.log(1 / beta)

    if verbose:
        print('alpha =', alpha)
        print('beta =', beta)
        print('thetaW =', thetaW)
        print('thetaT =', thetaT)

    p = 1 - np.exp((-beta ** k) * (beta**(-thetaW) + beta**(-thetaT)))
    return p.item()










