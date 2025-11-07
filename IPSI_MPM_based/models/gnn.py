import torch
from torch import Tensor
from torch import nn
# from torch_scatter import scatter


class GNN(nn.Module):
    """
    Reimplementaion of the Message-Passing class in torch-geometric to allow more flexibility.
    """
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x: Tensor, es: Tensor, f_e: Tensor=None, agg: str='mean') -> Tensor:
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            x: [node, ..., dim], node embeddings 
        """
        msg, idx, size = self.message(x, es, f_e)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str = 'mean') -> Tensor:
        return scatter_replacement(msg, idx, dim_size=size, reduce=agg)

    # def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str='mean') -> Tensor:
    #     """
    #     Args:
    #         msg: [E, ..., dim * 2]
    #         idx: [E]
    #         size: number of nodes
    #         agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'
    #
    #     Return:
    #         aggregated node embeddings
    #     """
    #     assert agg in {'add', 'mean', 'max'}
    #     return scatter(msg, idx, dim_size=size, dim=0, reduce=agg)

    def node2edge(self, x_i: Tensor, x_o: Tensor, f_e: Tensor) -> Tensor:
        """
        Args:
            x_i: [E, ..., dim], embeddings of incoming nodes
            x_o: [E, ..., dim], embeddings of outcoming nodes
            f_e: [E, ..., dim * 2], edge embeddings

        Return:
            edge embeddings
        """
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x: Tensor, es: Tensor, f_e: Tensor=None, option: str='o2i'):
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            option: default: 'o2i'
                'o2i': collecting incoming edge embeddings
                'i2o': collecting outcoming edge embeddings

        Return:
            mgs: [E, ..., dim * 2], edge embeddings
            col: [E], indices of 
            size: number of nodes
        """
        if option == 'i2o':
            row, col = es
        if option == 'o2i':
            col, row = es
        else:
            raise ValueError('i2o or o2i')
        x_i, x_o = x[row], x[col]
        msg = self.node2edge(x_i, x_o, f_e)
        return msg, col, len(x)

    def update(self, x):
        return x


def scatter_replacement(msg: Tensor, idx: Tensor, dim_size: int, reduce: str = 'mean') -> Tensor:
    """
    Replacement for torch_scatter.scatter using native PyTorch.

    Args:
        msg: [E, ..., dim]
        idx: [E]
        dim_size: int
        reduce: 'add', 'mean', or 'max'
    """
    out_shape = (dim_size,) + msg.shape[1:]
    out = torch.zeros(out_shape, dtype=msg.dtype, device=msg.device)

    if reduce == 'add':
        out.index_add_(0, idx, msg)
    elif reduce == 'mean':
        out.index_add_(0, idx, msg)
        count = torch.zeros(dim_size, dtype=msg.dtype, device=msg.device)
        count.index_add_(0, idx, torch.ones_like(idx, dtype=msg.dtype))
        # Reshape count to match dimensions for broadcasting
        count = count.clamp(min=1).view([-1] + [1] * (msg.dim() - 1))
        out = out / count
    elif reduce == 'max':
        out.fill_(-float('inf'))
        out.index_copy_(0, idx, msg)  # crude, not exact max if duplicate idx
        # For real max, see below*
    else:
        raise ValueError(f"Unsupported reduce type: {reduce}")

    return out