import msprime
import numpy as np
import xarray.testing as xt

import phylokit as pk


class Test_Hartigan_Parsimony_Vectorised:
    def simulate_ts(self, num_samples, num_sites, seed=1234):
        tsa = msprime.sim_ancestry(
            num_samples, sequence_length=num_sites, ploidy=1, random_seed=seed
        )
        return msprime.sim_mutations(tsa, rate=0.01, random_seed=seed)

    def setup(self):
        ts_in = self.simulate_ts(10, 100)
        ds = pk.ts_to_dataset(ts_in)
        ds_tree = pk.upgma(ds)
        ds_merged = ds_tree.merge(ds)
        return ds_merged.squeeze("ploidy")

    def result_dataset(self):
        pk_mts = self.setup()
        pk_mts["sites_parsimony_score"] = ("variants", np.array([1, 1, 1, 1, 1, 1]))
        return pk_mts

    def test_hartigan_parsimony_vectorised(self):
        pk_mts = self.setup()
        assert np.array_equal(
            pk.numba_hartigan_parsimony_vectorised(
                pk_mts.node_left_child.data,
                pk_mts.node_right_sib.data,
                pk_mts.sample_node.data,
                pk_mts.call_genotype.data,
            ),
            np.array([1, 1, 1, 1, 1, 1]),
        )

    def test_hartigan_parsimony_vectorised_parallel(self):
        pk_mts = self.setup()
        assert np.array_equal(
            pk.get_hartigan_parsimony_score(pk_mts),
            np.array([1, 1, 1, 1, 1, 1]),
        )

    def test_append_parsimony_score(self):
        pk_mts = self.setup()
        xt.assert_equal(
            pk.append_parsimony_score(pk_mts),
            self.result_dataset(),
        )
