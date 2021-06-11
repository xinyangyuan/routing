import heapq
import typing

import torch
import numpy as np

class PriorityQueue:
    """Priority-Queue that maintain top-k items"""
    def __init__(self, maxsize) -> None:
        self.queue =[]
        self.n = 0
        self.maxsize = maxsize
    
    def get(self):
        return heapq.heappop(self.queue) if self.n else None

    def put(self, item) -> None:
        """put item to queue"""

        if self.n < self.maxsize:
            heapq.heappush(self.queue, item)
            self.n += 1

        if len(self.queue) > 0 and item[0] < self.queue[-1][0]:
            self.queue[-1] = item # replace worst/largest score item


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

def beam_search_v2(start_node:int, weight_matrix: torch.Tensor, num_beam:int=3) -> typing.List[int]:
    """Non-repeat beam search for best probability sequence
    Args:
        start_node: (int) fixed starting node
        weight_matrix: (torch.tensor, numpy.ndarray) stop-pair transition matrix or weighted directed graph (num_stops, num_stops)
        num_beam: (int) number of searching branch kept
    Returns:
        output: (list) of sequence of best search
    Example:
        beam_search(start_node=3, weight_matrix=torch.rand((10,10)), num_beam=5)
    """
    
    n = num_beam
    station = start_node
    matrix = weight_matrix # (n, n)
    num_stops = matrix.size(1) # number of stops

    assert matrix.shape[0] == matrix.shape[1], "weight_matrix should be a square matrix"
    assert n <= matrix.shape[0], "num_beam should be less than weight_matrix size"

    # Initialization
    sequences = PriorityQueue(maxsize=n) # stop candidate stops list in priority queue
    for _ in range(n): sequences.put((0, [station] * num_stops)) # (log_prop, [station, station, ..., station])

    # Hierarchical beam search
    for step in range(num_stops - 1):
        
        # a temporary sequence for current step
        sequences_step = PriorityQueue(maxsize=n)
        
        # iterate over branches
        for _ in range(num_beam):
            # get current searching stop
            neg_log_prop, sequence = sequences.get()
            stop = sequence[step] 

            # get topk candidate-stops
            candidates = matrix[stop].clone()
            candidates[sequence[0:step+1]] = float("-inf") # masked out visited stops
            log_props, candidates = candidates.topk(min(num_beam, num_stops - step)) # torch tensors (num_stops - step -1)
            
            # put to temporary priority queue
            for log_prop, candidate in zip(log_props.tolist(), candidates.tolist()):
                assert log_prop <= 0, "log_prop should be less or equal to zero"
                sequence[step + 1] = candidate
                sequences_step.put((neg_log_prop - log_prop, [candidate if i == (step+1) else stop for i,stop in enumerate(sequence)])) # PriorityQueue is in ascending order, flip sign of log_prop

        # update priority queue
        sequences = sequences_step

    neg_log_prop, output = sequences.get()

    print ("max cumulative log-probability: ", -neg_log_prop)
    print ("output: ", output)
    
    return output

if __name__ == "__main__":
    weight_matrix = torch.rand(5,5)

    print(torch.log(weight_matrix))

    beam_search(start_node=3, weight_matrix=weight_matrix)
    beam_search_v2(start_node=3, weight_matrix=torch.log(weight_matrix))