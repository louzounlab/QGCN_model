import torch
import torch.nn as nn
import math

srss = lambda x: 1 - (2 / (x ** 2 + 1))


def to_batch_features(x: torch.Tensor, batch: torch.Tensor):
    """This function gets pytorch_geometric's batch node features (data.x and data.batch), 
    and return the features in 3D shape - [graphs, nodes, features]. 

    Args:
        x (torch.Tensor): The features for all the nodes in the batch
        batch (torch.Tensor): The graph number for each node

    Returns:
        torch.Tensor: The nodes features in 3D shape.
    """
    num_of_graphs = len(batch.unique())

    lst = [[] for _ in range(num_of_graphs)]

    for graph, features in zip(batch, x):
        lst[int(graph)].append(features)

    # Build the padding featues matrix:
    max_nodes = max([len(l) for l in lst])
    num_of_features = x.shape[1]
    
    padding_features = torch.zeros([num_of_graphs, max_nodes, num_of_features])

    for i, l in enumerate(lst):
        features = torch.stack(l)
        shape = features.shape
        padding_features[i, :shape[0], :shape[1]] = features

    return padding_features


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        This class implements gcn layer in graph neural network.
        Args:
            in_dim (int): Number of features for node in the input.
            out_dim (int): Number of features for node in the output. 
        """
        super(GCN, self).__init__()
        self._linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters(out_dim)

    def reset_parameters(self, out_dim):
        stdv = 1. / math.sqrt(out_dim)
        self._linear.weight.data.uniform_(-stdv, stdv)

    def forward(self, A, x0):
        Ax = torch.matmul(A, x0)
        x = self._linear(Ax)

        return x


class QGCNLastLayer(nn.Module):
    def __init__(self, left_in_dim: int, right_in_dim: int, out_dim: int):
        """This layer is used in graph classification missions,
        for pooling input size with unknown shape to a fix size. 

        The forward function gets the adj matrix (A) of the graph, the orignal features vector (x0), 
        and the features vector after manipulations (x1). 
        Args:
            left_in_dim (int): Number of features per node in the new vector.
            right_in_dim (int): Number of features per node in the original vector.
            out_dim (int): Size of the output vector.
        """
        super(QGCNLastLayer, self).__init__()
        self._left_linear = nn.Linear(left_in_dim, 1)
        self._right_linear = nn.Linear(right_in_dim, out_dim)
        self.reset_parameters(out_dim)

    def reset_parameters(self, out_dim):
        stdv = 1. / math.sqrt(1)
        self._left_linear.weight.data.uniform_(-stdv, stdv)
        stdv_r = 1. / math.sqrt(out_dim)
        self._right_linear.weight.data.uniform_(-stdv_r, stdv_r)

    def forward(self, A, x0, x1):
        """Apply QGCN layer on the inputs.

        Args:
            A (torch.Tensor): The adj matirx of the graph
            x0 (torch.Tensor): The original node features
            x1 (torch.Tensor): The node features after layers

        Returns:
            torch.Tensor: Fix size vector - the output of the model.
        """
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.matmul(x1.permute(0, 2, 1), A)
        W2_x1_A = self._left_linear(x1_A.permute(0, 2, 1))
        W2_x1_A_x0 = torch.matmul(W2_x1_A.permute(0, 2, 1), x0)
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        return W2_x1_A_x0_W3.squeeze(dim=1)


class QGCN(nn.Module):
    def __init__(self, num_of_features: int,
                 number_of_classes: int=2,
                 dropout: float=0,
                 inside_dim: int=64,
                 activation=srss):
        """Graph classification based on nodes features model,
        using GCN layers and QGCN final layer.

        Args:
            num_of_features (int): The number of features for each node.
            number_of_classes (int, optional): Number of possible classes.
            dropout (float, optional): Dropout rate for the layers in the model. Defaults to 0.
            inside_dim (int, optional): The number of neurons in the inner layers. Defaults to 64.
            activation (function, optional): The model's activation fucntion. Defaults to srss.
        """
        
        super(QGCN, self).__init__()
        self.num_classes = number_of_classes
        self._is_binary = (number_of_classes == 2)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.linear1 = GCN(num_of_features, inside_dim)
        self.linear2 = GCN(inside_dim, inside_dim)
        self.linear3 = GCN(inside_dim, inside_dim)

        self._qgcn_layer = QGCNLastLayer(left_in_dim=inside_dim, right_in_dim=num_of_features,
                                              out_dim=1 if self._is_binary else self.num_classes)


    def forward(self, A, x0):
        x1 = x0
        x1 = self.linear1(A, x1)
        x1 = self.dropout(x1)
        x1 = self.activation(x1)
        
        x1 = self.linear2(A, x1)
        x1 = self.dropout(x1)
        x1 = self.activation(x1)
        
        x1 = self.linear3(A, x1)
        x1 = self.activation(x1)
        
        # We can use different matrix instead of x0,
        # we just need matrix with information about the graph.
        x2 = self._qgcn_layer(A, x0, x1)  
        return x2
