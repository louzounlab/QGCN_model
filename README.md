# QGCN
> QGCN method for graph classification - you can read the full article [here](https://arxiv.org/abs/2104.06750).  
> TL;DR: Use the QGCN layer instead of pooling the GCN output for a fix size.  

## Execution examples:

pytorch_geometric:
```python
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from qgcn import QGCN, to_batch_features

dataset = TUDataset(root='TUDataset', name='dataset-name')
dataset = dataset.shuffle()

model = QGCN(num_of_features=dataset.num_features,
             number_of_classes=dataset.num_classes)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

for data in loader:
    # Convert the data from the shape of pytorch_geometric to the shape of the model. 
    A = to_dense_adj(data.edge_index, data.batch)
    x0 = to_batch_features(data.x, data.batch)
    
    preds = model(A, x0)
```

## Installation
required packages:
* torch~=1.13.0


## Links
There is a full version of this model, which executes the learning end to end - you can see it [here](https://github.com/louzounlab/QGCN).  
For more details, visit the [lab website](https://yolo.math.biu.ac.il/)
