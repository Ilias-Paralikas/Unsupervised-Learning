import torch
import torch.nn as nn
import torch.nn.functional as F

def group_norm_32(C):
    return nn.GroupNorm(min(32, C), C)

def group_norm_16(C):
    return nn.GroupNorm(min(16, C), C)


def compute_component_similarity_loss(component_vectors):
    """
    component_vectors: (batch_size, N, vector_dim)
    Returns: scalar loss representing the average pairwise similarity
    """
    # Normalize to unit vectors for cosine similarity
    norm_vectors = F.normalize(component_vectors, p=2, dim=-1)
    
    # Compute cosine similarity matrix: (batch_size, N, N)
    # This does a batch matrix multiplication (B x N x D) * (B x D x N)
    sim_matrix = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    
    # We want to ignore the diagonal (self-similarity is always 1)
    n = sim_matrix.size(1)
    diagonal_mask = torch.eye(n, device=sim_matrix.device).bool()
    
    # Extract only the off-diagonal elements
    off_diag_sim = sim_matrix.masked_select(~diagonal_mask).view(-1, n, n - 1)
    
    # You can return the mean similarity. 
    # To encourage diversity, minimize this value.
    return off_diag_sim.mean()

# Add any custom activations or layers here
REGISTRY = {
    # Norms
    "BatchNorm2d": nn.BatchNorm2d,
    "LayerNorm": nn.LayerNorm,
    "group_norm_32": group_norm_32,
    "group_norm_16": group_norm_16,
    
    # Activations
    "ReLU": nn.ReLU(inplace=True),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.2),
    "Sigmoid": nn.Sigmoid(),
    "Softmax": nn.Softmax(dim=1),
    "Identity": nn.Identity(),

    # Optimization
    "Adam": torch.optim.Adam,
    "L1Loss": nn.L1Loss(),
    "MSELoss":nn.MSELoss(),
    "CosineSimilarity": compute_component_similarity_loss,
    "CrossEntropyLoss": nn.CrossEntropyLoss()

}

def get_component(name):
    if name in REGISTRY:
        return REGISTRY[name]
    # Fallback for standard types (ints, strings, etc)
    return name