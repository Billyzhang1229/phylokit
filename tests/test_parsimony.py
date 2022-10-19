import msprime
import numba
import numpy as np
import pytest
import xarray.testing as xt

import phylokit as pk
from phylokit.inference import upgma
from phylokit.parsimony.hartigan import ts_to_dataset


def simulate_ts(num_samples, num_sites, seed=1234):
    tsa = msprime.sim_ancestry(
        num_samples, sequence_length=num_sites, ploidy=1, random_seed=seed
    )
    return msprime.sim_mutations(tsa, rate=0.01, random_seed=seed)


class Test_Hartigan_Parsimony_Vectorised:
    def generate_test_tree_ds():
        trees = []
        for i in range(1, 6):
            ts_in = simulate_ts(10, 100, seed=i * 88)
            ds = ts_to_dataset(ts_in)
            ds_tree = upgma(ds)
            ds_merged = ds_tree.merge(ds)
            trees.append(ds_merged.squeeze("ploidy"))
        return trees

    @numba.jit
    def _hartigan_preorder_vectorised(self, parent, optimal_set, left_child, right_sib):
        num_sites, num_alleles = optimal_set.shape[1:]

        allele_count = np.zeros((num_sites, num_alleles), dtype=np.int32)
        child = left_child[parent]
        while child != -1:
            self._hartigan_preorder_vectorised(
                child, optimal_set, left_child, right_sib
            )
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

    @numba.jit
    def _hartigan_postorder_vectorised(
        self, node, state, optimal_set, left_child, right_sib
    ):
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
            v_muts = self._hartigan_postorder_vectorised(
                v, state, optimal_set, left_child, right_sib
            )
            mutations += v_muts
            v = right_sib[v]
        return mutations

    @numba.jit
    def _hartigan_initialise_vectorised(self, optimal_set, genotypes, samples):
        for k, site_genotypes in enumerate(genotypes):
            for j, u in enumerate(samples):
                optimal_set[u, k, site_genotypes[j]] = 1

    def numba_hartigan_parsimony_vectorised(self, ds, genotypes):

        left_child = ds.node_left_child.data
        right_sib = ds.node_right_sib.data

        # Simple version assuming non missing data and one root
        num_alleles = np.max(genotypes) + 1
        num_sites = genotypes.shape[0]
        num_nodes = left_child.shape[0] - 1
        samples = ds.sample_node.data

        optimal_set = np.zeros((num_nodes, num_sites, num_alleles), dtype=np.int8)
        self._hartigan_initialise_vectorised(optimal_set, genotypes, samples)
        self._hartigan_preorder_vectorised(-1, optimal_set, left_child, right_sib)
        ancestral_state = np.argmax(optimal_set[-1], axis=1)
        return self._hartigan_postorder_vectorised(
            -1, ancestral_state, optimal_set, left_child, right_sib
        )

    @pytest.mark.parametrize("ds", generate_test_tree_ds())
    def test_hartigan_parsimony_vectorised(self, ds):
        assert np.array_equal(
            pk.numba_hartigan_parsimony_vectorised(
                ds.node_left_child.data,
                ds.node_right_sib.data,
                ds.sample_node.data,
                ds.call_genotype.data,
            ),
            self.numba_hartigan_parsimony_vectorised(ds, ds.call_genotype.data),
        )

    @pytest.mark.parametrize("ds", generate_test_tree_ds())
    def test_hartigan_parsimony_vectorised_parallel(self, ds):
        assert np.array_equal(
            pk.get_hartigan_parsimony_score(ds),
            self.numba_hartigan_parsimony_vectorised(ds, ds.call_genotype.data),
        )

    @pytest.mark.parametrize("ds", generate_test_tree_ds())
    def test_append_parsimony_score(self, ds):
        ds["sites_parsimony_score"] = self.numba_hartigan_parsimony_vectorised(
            ds, ds.call_genotype.data
        )
        xt.assert_equal(pk.append_parsimony_score(ds), ds)
