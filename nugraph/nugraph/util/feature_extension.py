"""Transform to add set of extended features to planar nodes"""
import torch
from torch_geometric.transforms import BaseTransform

def calculate_features(wt_coords, edge_index, idx):
    """Calculate the four extended features for each node."""

    # Pairwise euclidean distances in wire-time space
    dist_table = torch.cdist(wt_coords, wt_coords)
    dist_table.fill_diagonal_(float('inf'))

    # Find a (n_nodes, 2) matrix containing the distances and
    # indices of the two closest nodes to each node
    dists_2closest_nodes, idxs_2closest_nodes = torch.topk(dist_table, 2, dim=1,
                                                           largest=False, sorted=True)
    
    # Finding the double delta of wire and time differences with the two closest neighbors
    dwire = (2*wt_coords[:, 0] - wt_coords[idxs_2closest_nodes[:,1], 0]
                          - wt_coords[idxs_2closest_nodes[:,0], 0]).view(-1,1) # wire
    dtime = (2*wt_coords[:, 1] - wt_coords[idxs_2closest_nodes[:,1], 1]
                          - wt_coords[idxs_2closest_nodes[:,0], 1]).view(-1,1) # time

    # Adding node degree
    nodes_degree = torch.unique(edge_index[0], sorted=True,
                                return_counts=True)[1].view(-1,1)
    nodes_degree = torch.log(nodes_degree[idx].clamp(min=1))

    # Adding shortest edge length ('dists_2closest_nodes' is sorted in ascending order)
    min_dist = dists_2closest_nodes[:,0].view(-1,1)

    return torch.cat((dwire, dtime, nodes_degree, min_dist), dim=-1)


class FeatureExtension(BaseTransform):
    """
    Transform to add additional values to planar feature tensor

    Args:
        planes: Tuple of detector plane names
    """
    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def forward(self, data: "pyg.data.HeteroData") -> "pyg.data.HeteroData":
        """
        FeatureExtension forward function

        This function appends additional features to the planar node
        embedding.

        Args:
            data: Graph data object
        """

        if "hit" in data.node_types:
            data['hit'].x = torch.cat((data['hit'].x, torch.zeros(data['hit'].x.shape[0],4)),
                                      dim=-1)

        for i, p in enumerate(self.planes):

            if "hit" in data.node_types:
                idx = (data["hit"].plane == i).nonzero().squeeze(dim=1)
                pos = data["hit"]["pos"][idx]
                edge = data['hit', 'delaunay-planar', 'hit']
                x = data['hit'].x[idx]
            else:
                idx = torch.arange(data[p].x.shape[0])
                pos = data.collect("pos")[p]
                edge = data[p, 'plane', p]
                x = torch.cat((data[p].x, torch.zeros(data[p].x.shape[0],4)), dim=-1)

            # Adding delta wire an delta time (dwire/dtime doesn't work; some infs)
            # Extracting wire and time information
            wt_coords = torch.stack((pos[:, 0], pos[:, 1]), dim=1) # [wire, time]

            # Calculate the four extended features for each node
            extended_vars = calculate_features(wt_coords, edge.edge_index, idx)
            
            # Write extended features back to the data object; x is a copy
            # (fancy indexing), so explicit assignment is required in both branches
            x[:, -4:] = extended_vars
            if "hit" in data.node_types:
                data['hit'].x[idx] = x
            else:
                data[p].x = x

        return data


   