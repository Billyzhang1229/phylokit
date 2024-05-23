import numpy as np

from . import core
from . import jit
from . import util


@jit.numba_njit()
def _permute_node_seq(nodes, ordering, reversed_map):
    ret = np.zeros_like(nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        old_node = nodes[v]
        if old_node != -1:
            ret[u] = reversed_map[old_node]
        else:
            ret[u] = -1
    return ret


def permute_tree(ds, ordering):
    """
    Returns a new dataset in which the tree nodes have been permuted according
    to the specified ordering such that node u in the new dataset will be
    equivalent to ``ordering[u]``.
    :param xarray.DataSet ds: The tree dataset to permute.
    :param list ordering: The permutation to apply to the nodes.
    :return: A new dataset with the permuted nodes.
    :rtype: xarray.DataSet
    """
    num_nodes = ds.node_left_child.shape[0]
    if len(ordering) != num_nodes:
        raise ValueError(
            "The length of the ordering must be equal to the number of nodes"
        )

    for node in ordering:
        util.check_node_bounds(ds, node)

    reversed_map = np.zeros(num_nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        reversed_map[v] = u

    return core.create_tree_dataset(
        parent=_permute_node_seq(ds.node_parent.data, ordering, reversed_map),
        left_child=_permute_node_seq(ds.node_left_child.data, ordering, reversed_map),
        right_sib=_permute_node_seq(ds.node_right_sib.data, ordering, reversed_map),
        samples=np.array([reversed_map[s] for s in ds.sample_node.data]),
    )


def spr(ds, source, destination):
    """
    Perform a Subtree Prune and Regraft (SPR) operation on the tree dataset.

    :param xarray.DataSet ds: The tree dataset.
    :param int source: The node to prune.
    :param int destination: The node to regraft.
    :return: The tree dataset with the specified SPR operation applied.
    """

    parent = ds.node_parent
    left_child = ds.node_left_child
    right_sib = ds.node_right_sib
    time = ds.node_time

    # Validate inputs
    if (
        source == destination
        or source < 0
        or destination < 0
        or source >= ds.nodes.shape[0]
        or destination >= ds.nodes.shape[0]
    ):
        raise ValueError("Invalid source or destination.")

    src_parent = parent[source]
    dest_parent = parent[destination]
    src_parent_parent = parent[src_parent]

    if destination == src_parent:
        raise ValueError("Destination node cannot be the parent of the source node.")

    # Detach the subtree at source
    if src_parent_parent != -1:
        if left_child[src_parent_parent] == src_parent:
            if left_child[src_parent] == source:
                left_child[src_parent_parent] = right_sib[source]
                right_sib[right_sib[source]] = right_sib[src_parent]
                parent[right_sib[source]] = src_parent_parent
            else:
                left_child[src_parent_parent] = left_child[src_parent]
                right_sib[left_child[src_parent]] = right_sib[src_parent]
                parent[left_child[src_parent]] = src_parent_parent
        else:
            if left_child[src_parent] == source:
                right_sib[left_child[src_parent_parent]] = right_sib[source]
                parent[right_sib[source]] = src_parent_parent
            else:
                right_sib[left_child[src_parent_parent]] = left_child[src_parent]
                right_sib[left_child[src_parent]] = right_sib[source]
                parent[left_child[src_parent]] = src_parent_parent
    else:
        raise ValueError("Source parent node cannot be the root in this operation.")

    parent[src_parent] = -1
    left_child[src_parent] = -1
    right_sib[src_parent] = -1
    right_sib[source] = -1

    # Update time and branch length
    time[src_parent] = time[destination] + (time[dest_parent] - time[destination]) / 2

    # Attach the subtree at destination
    if left_child[dest_parent] == destination:
        right_sib[src_parent] = right_sib[destination]
        left_child[src_parent] = destination
        left_child[dest_parent] = src_parent
        parent[src_parent] = dest_parent
        parent[destination] = src_parent
        right_sib[destination] = source
    else:
        right_sib[left_child[dest_parent]] = src_parent
        parent[src_parent] = dest_parent
        parent[destination] = src_parent
        left_child[dest_parent] = destination
        right_sib[destination] = source

    # Update branch length
    ds.node_branch_length[:] = util.get_node_branch_length(ds)

    return ds
