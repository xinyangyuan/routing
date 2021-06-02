import torch
import numpy as np

def beam_search(start_node, weight_matrix, num_beam=3):
    """non-repeat beam search for best probability sequence
    Args:
        start_node: (int) fixed starting node
        weight_matrix: (torch.tensor, numpy.ndarray) of weighted directed graph
        num_beam: (int) number of searching branch kept
    Returns:
        output: (list) of sequence of best search
    Example:
        beam_search(start_node=3, weight_matrix=torch.rand((10,10)), num_beam=5)
    """
    station = start_node
    matrix = np.array(weight_matrix)
    n = num_beam
    step = matrix.shape[1]-1

    assert matrix.shape[0] == matrix.shape[1], "weight_matrix should be a square matrix"
    assert n <= matrix.shape[0], "num_beam should be less than weight_matrix size"

    # initialize node search storage 
    node = np.array([[station]]*n) # (n,1)
    nodes = node # (n,1)
    nodes0 = nodes # (n,1)
    node = np.unique(node,axis=0)
    # print (node)
    # initial state
    state = np.ones((1,1)) # (1,1)
    # print (state)

    for _ in range(step):
        # calculate cumulative probability of candidate
        candidate = matrix[node].squeeze() # (n,len)
        # print (candidate)
        candidate = state * candidate
        # print (candidate)

        # run from second loop
        # set history nodes to zero
        if _ == 0:
            # set station column to 0
            candidate[:, station] = 0
        else:  
            for i in range(n):
                zero_idx = nodes[i]
                candidate[i,zero_idx] = 0
        # print (candidate)

        # obtain nbest index
        row,col = np.unravel_index(np.argsort(candidate.ravel()),candidate.shape)
        row,col = row[-n:],col[-n:]
        nbest_idx = np.vstack((row,col)).T
        # print (nbest_idx)

        # update state with nbest values
        state = np.array([candidate.item(tuple(i)) for i in nbest_idx]).reshape(n,1)
        # print (state)

        # update nodes branching 
        for i in range(n):
            nodes[i] = nodes0[nbest_idx[i,0]]
        # print (nodes)
        # update nodes
        node = nbest_idx[:,-1:]
        # node = np.unique(node,axis=0)
        nodes = np.concatenate((nodes,node),1)
        nodes0 = nodes.copy()
        # print (node)
        # print (nodes)

    output = nodes[np.argmax(state)]
    print ("max cumulative probability: ", np.max(state))
    print ("output: ", nodes[np.argmax(state)])
    # print (np.sort(output))
    return output
