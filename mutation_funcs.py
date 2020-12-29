# Mutation helper functions:

import random
import numpy as np
def swap_weights(array):
    """
    Given the matrix of weights of the neural network, it mutates it.
    Chooses randomly how how many columns and which ones will be swapped.
    Then it swaps the positions of the weights columns.

    array : np.ndarray

    Returns : np.ndarray
    """
    number_of_swaps = (np.random.randint(0, array.shape[0] * 2, 1) // 2)[0]
    if number_of_swaps == 0:
        return array
    indexes = np.random.choice(array.shape[0], number_of_swaps, replace=False)
    swap_pairs = [(indexes[i], indexes[i+1]) for i in range(0, number_of_swaps, 2) \
                 if i+1 != number_of_swaps]
    
    for pair in swap_pairs:
        tmp = array[pair[0]]
        array[pair[0]] = array[pair[1]]
        array[pair[1]] = tmp
    return array

def replace_weights(array):
    """
    Given the matrix of weights of the neural network, it mutates it.
    Randomly chooses the columns to be completely replaced by randomly generated
    new columns.

    array : np.ndarray

    Returns : np.ndarray
    """
    number_of_replaces = np.random.randint(2, array.shape[0], 1)[0]
    indexes = np.random.choice(array.shape[0], number_of_replaces, replace=False)
    
    for index in indexes:
        array[index] = np.random.uniform(-1,1)
    return array