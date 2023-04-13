from typing import Union

from torch_geometric.data import HeteroData, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('remove_edge_efatuers')
class RemoveEdgeFeatures(BaseTransform):
    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            del store.edge_attr
        return data