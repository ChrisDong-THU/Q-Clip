from typing import Literal
import numpy as np
import faiss
import networkit as nk


def deduplicate_cc(vectors: np.ndarray, index_type: Literal["BinaryFlat", "FlatIP"], K: int = 8, T: float = 0.8):
    '''
    Deduplicate connected components in a graph using union-find algorithm.
    
    Args:
        vectors: np.ndarray, shape (N, D), input vectors to deduplicate
        index_type: Literal["BinaryFlat", "FlatIP"], type of faiss index to use
        K: int, number of nearest neighbors to search for each vector
        T: float, threshold for distance to consider two vectors as connected, hamming distance has been normalized to [0, 1]
    
    Returns:
        unique_frame_ids: List[int], list of unique frame ids in each connected component
    '''
    N, D = vectors.shape

    if index_type == "BinaryFlat":
        index = faiss.IndexBinaryFlat(D*8) # D*8 bits for each binary vector
    elif index_type == "FlatIP":
        index = faiss.IndexFlatIP(D) # D for each float vector
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    index.add(vectors)

    # GPU not supported for binaryFlat index
    if index_type == "FlatIP":
        index = faiss.index_cpu_to_all_gpus(index)
    
    # search for k nearest neighbors (including self)
    distances, indices = index.search(vectors, K+1)
    distances, indices = distances[:, 1:], indices[:, 1:] # remove itself
    
    # src < dst to avoid bidirectional edges
    row_ids = np.arange(N)[:, None]

    # hamming distance to distance
    if index_type == "BinaryFlat":
        distances = 1 - distances / (D*8)
    
    mask = (distances >= T) & (row_ids < indices)
    
    src, dst = np.where(mask)
    edge_indices = indices[src, dst]
    
    G = nk.Graph(n=N, weighted=False, directed=False)
    for u, v in zip(src, edge_indices):
        G.addEdge(int(u), int(v))

    cc = nk.components.ConnectedComponents(G).run()
    components = cc.getComponents() # list of list, each list is a connected component
    
    unique_frame_ids = [component[0] for component in components] # only pick the first frame in each component
    
    return unique_frame_ids