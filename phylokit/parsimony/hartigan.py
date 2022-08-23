import numba
import numpy as np
import xarray as xr


@numba.njit()
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


@numba.njit()
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


@numba.njit()
def _hartigan_initialise_vectorised(optimal_set, genotypes, samples):
    for k, site_genotypes in enumerate(genotypes):
        for j, u in enumerate(samples):
            optimal_set[u, k, site_genotypes[j]] = 1


def numba_hartigan_parsimony_vectorised(
    left_child, right_sib, samples, genotypes, alleles
):

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_sites = genotypes.shape[0]
    num_nodes = left_child.shape[0] - 1

    optimal_set = np.zeros((num_nodes, num_sites, num_alleles), dtype=np.int8)
    _hartigan_initialise_vectorised(optimal_set, genotypes, samples)
    _hartigan_preorder_vectorised(-1, optimal_set, left_child, right_sib)
    ancestral_state = np.argmax(optimal_set[-1], axis=1)
    return _hartigan_postorder_vectorised(
        -1, ancestral_state, optimal_set, left_child, right_sib
    )


def get_hartigan_parsimony_score(ds, genotypes, alleles, chunk_size=1000):
    """
    Calculate the parsimony score for each site in the dataset.

    :param ds: The dataset to calculate the parsimony score for.
    :param genotypes: The genotypes to calculate the parsimony score for
    e.g. ts.genotype_matrix().
    :param alleles: The alleles to calculate the parsimony score for
    e.g. ["A", "C", "G", "T"].
    :param chunk_size: The size of the chunks to use when calculating the
    parsimony score.
    :return: The parsimony score for each site in the dataset.
    :rtype: xarray.DataArray
    """
    ds["sites_genotypes"] = (["sites", "samples"], genotypes)
    chunked = ds.chunk({"sites": chunk_size})

    return xr.apply_ufunc(
        numba_hartigan_parsimony_vectorised,
        chunked.node_left_child,
        chunked.node_right_sib,
        chunked.sample_node,
        chunked.sites_genotypes,
        alleles,
        input_core_dims=[["nodes"], ["nodes"], ["samples"], ["samples"], ["alleles"]],
        dask="parallelized",
        output_dtypes=[np.uint32],
    ).compute()


def append_parsimony_score(ds, genotypes, alleles, chunk_size=1000):
    """
    Append the parsimony score to the dataset.

    :param ds: The dataset to append the parsimony score to.
    :param genotypes: The genotypes to calculate the parsimony score for
    e.g. ts.genotype_matrix().
    :param alleles: The alleles to calculate the parsimony score for
    e.g. ["A", "C", "G", "T"].
    :param chunk_size: Number of sites to calculate the parsimony score for at a time.
    The size of each chunk should be between 100MB and 1GB for optimal performance.
    :return: The dataset with the parsimony score appended.
    :rtype: xarray.Dataset
    """
    ds["sites_parsimony_score"] = get_hartigan_parsimony_score(
        ds, genotypes, alleles, chunk_size=chunk_size
    )
    return ds
