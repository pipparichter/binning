'''Implementation of different models for genome coverage.'''
from scipy.special import comb 
import numpy as np 
from decimal import Decimal 
from tqdm import tqdm 
import numpy as np
import pandas as pd

# I think I will go about this by fixing total R to be 

def coverage_wendl_2013(alpha:float=None, R:int=None, L:int=None, gamma:int=None):
    '''A probability-based model of coverage based on an extension of something called Steven's theorem. 
    Has some advantages (see paper) over the expectation-based approaches, such as those which are derived from the
    Lander-Waterman model for single genomes.
    
    https://link.springer.com/article/10.1007/s00285-012-0586-x
    
    :param alpha: A Bernoulli probability, the chance that a randomly selected read represents the target species. 
        This parameter, understood as the “abundance”, is project-dependent.
    :param R: 

    '''
    # k = 0 # For genome coverage, take the number of gaps to be 0. 
    # phi is probability that a read covers a particular base position within the target species’ genome
    phi = L / gamma
    # NOTE: Why doesn't eta account for L? Chance of overlap would increase. 
    eta = min(int(R), int(1 / phi)) # The maxinum number of reads which can be placed without overlap. 

    # print('phi =', phi)

    summation = Decimal(0)
    pbar = tqdm(range(eta + 1), desc='coverage_wendl_2013')
    for beta in pbar:
        s = Decimal(comb(R, beta, exact=True))
        s *= Decimal((-alpha) ** beta)
        s *= Decimal((1 - beta * phi) ** (beta - 1))
        s *= Decimal((1 - beta * alpha * phi) ** (R - beta))
        # print(Decimal(comb(R, beta, exact=True)), s)
        summation += s
        pbar.set_description(f'coverage_wendl_2013: Current value of summation is {np.round(float(summation), 4)}.')
        pbar.update(1)

    return summation

    # summation = 0
    # for beta in range(k, eta + 1):
    #     s = comb(R - k, beta - k, exact=True) 
    #     s *= (-1) ** (beta - k)
    #     s *= alpha ** beta
    #     s *= (1 - beta * phi) ** (beta - 1)
    #     s *= (1 - beta * alpha * phi) ** (R - beta)
    #     summation += s




def coverage_lander_waterman_1998(G:int=None, L:int=None, R:int=None, alpha:float=None, T:float=int):
    '''This returns the expected coverage, not the probability of complete coverage.'''
    N = R * alpha # Assume that the number of reads mapping to a target genome is Poisson-distributed, and use the expected value. 
    c = L * N / G # Redundance of coverage.
    # print(f'coverage_lander_waterman_1998: The average nucleotide will be sequenced {np.round(c, 2)} times on average.')
    theta = T / L 
    a = N / G # The probability per base of starting a new clone.
    sigma = 1 - theta 

    # E_num_islands = max(1, N * np.exp(-c * sigma)) # This becomes vanishingly small if coverage is too high. 
    # E_island_length = L * (((np.exp(c * sigma) - 1) / c) + (1 - sigma))
    # print(f'coverage_lander_waterman_1998: The predicted number of islands is {np.round(E_num_islands, 2)}.')
    # print(f'coverage_lander_waterman_1998: The predicted length of islands in nucleotides is {np.round(E_island_length, 2)}.')

    # return min(G, E_num_islands * E_island_length)
    return G * (1 - np.exp(-c))


def composition_lander_waterman_1998(coverage:float, genome_sizes:np.ndarray, read_size:int=200):
    '''Compute the read depth and abundances required for each genome to have equal coverage.
    Uses the Lander-Waterman model to estimate coverage. 

    :param coverage: The coverage required for each genome. 
    :param genome_size: An array-like object of genome sizes.
    :param read_size: The read size. 200 by default. 
    '''
    read_nums = []
    for size in genome_sizes:
        c = - np.log(1 - coverage)
        N = c * size / read_size # Get the number of reads required for the specified coverage. 
        read_nums.append(N)
    
    read_depth = int(sum(read_nums))
    abundances = np.array(read_nums) / read_depth
    return read_depth, abundances 