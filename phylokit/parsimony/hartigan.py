import numpy as np

from .. import jit


@jit.numba_njit
def _hartigan_preorder_vectorised(parent, optimal_set, left_child, right_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    allele_count = np.zeros((num_sites, num_alleles), dtype=np.int32)
    child = left_child[parent]
    while child != -1:
        _hartigan_preorder_vectorised(child, optimal_set, left_child, right_sib)
        allele_count += optimal_set[child]
        child = right_sib[child]

    if left_child[parent] != -1:
        for j in range(num_sites):
            site_allele_count = allele_count[j]
            # max_allele_count = np.max(site_allele_count)
            max_allele_count = 0
            for k in range(num_alleles):
                if site_allele_count[k] > max_allele_count:
                    max_allele_count = site_allele_count[k]
            for k in range(num_alleles):
                if site_allele_count[k] == max_allele_count:
                    optimal_set[parent, j, k] = 1


@jit.numba_njit
def _hartigan_postorder_vectorised(node, state, optimal_set, left_child, right_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    mutations = np.zeros(num_sites, dtype=np.int32)
    # Strictly speaking we only need to do this if we mutate it. Might be worth
    # keeping track of - but then that would complicate the inner loop, which
    # could hurt vectorisation/pipelining/etc.
    state = state.copy()
    for j in range(num_sites):
        site_optimal_set = optimal_set[node, j]
        if site_optimal_set[state[j]] == 0:
            # state[j] = np.argmax(site_optimal_set)
            maxval = -1
            argmax = -1
            for k in range(num_alleles):
                if site_optimal_set[k] > maxval:
                    maxval = site_optimal_set[k]
                    argmax = k
            state[j] = argmax
            mutations[j] = 1

    v = left_child[node]
    while v != -1:
        v_muts = _hartigan_postorder_vectorised(
            v, state, optimal_set, left_child, right_sib
        )
        mutations += v_muts
        v = right_sib[v]
    return mutations


@jit.numba_njit
def _hartigan_initialise_vectorised(optimal_set, genotypes, samples):
    for k, site_genotypes in enumerate(genotypes):
        for j, u in enumerate(samples):
            optimal_set[u, k, site_genotypes[j]] = 1


def numba_hartigan_parsimony_vectorised(ds, genotypes, alleles):

    left_child = ds.node_left_child.data
    right_sib = ds.node_right_sib.data

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_sites = genotypes.shape[0]
    num_nodes = left_child.shape[0] - 1
    samples = ds.sample_node.data

    optimal_set = np.zeros((num_nodes, num_sites, num_alleles), dtype=np.int8)
    _hartigan_initialise_vectorised(optimal_set, genotypes, samples)
    _hartigan_preorder_vectorised(-1, optimal_set, left_child, right_sib)
    ancestral_state = np.argmax(optimal_set[-1], axis=1)
    return _hartigan_postorder_vectorised(
        -1, ancestral_state, optimal_set, left_child, right_sib
    )
