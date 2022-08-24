import msprime
import numpy as np
import xarray.testing as xt

import phylokit as pk


class Test_Hartigan_Parsimony_Vectorised:
    def setup(self):
        ts = msprime.sim_ancestry(10, sequence_length=100, random_seed=1234)
        mts = msprime.sim_mutations(ts, rate=0.01, random_seed=5678)
        pk_mts = pk.from_tskit(mts.first())
        return pk_mts, mts

    def result_dataset(self):
        pk_mts, mts = self.setup()
        pk_mts["sites_genotypes"] = (["sites", "samples"], mts.genotype_matrix())
        pk_mts["sites_parsimony_score"] = (
            "sites",
            np.array(
                [2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        )
        return pk_mts

    def test_hartigan_parsimony_vectorised(self):
        pk_mts, mts = self.setup()
        assert (
            pk.numba_hartigan_parsimony_vectorised(
                pk_mts.node_left_child.data,
                pk_mts.node_right_sib.data,
                pk_mts.sample_node.data,
                mts.genotype_matrix(),
                ["A", "C", "G", "T"],
            ).all()
            == np.array(
                [2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ).all()
        )

    def test_hartigan_parsimony_vectorised_parallel(self):
        pk_mts, mts = self.setup()
        assert (
            pk.get_hartigan_parsimony_score(
                pk_mts,
                mts.genotype_matrix(),
                ["A", "C", "G", "T"],
            ).all()
            == np.array(
                [2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ).all()
        )

    def test_append_parsimony_score(self):
        pk_mts, mts = self.setup()
        xt.assert_equal(
            pk.append_parsimony_score(
                pk_mts, mts.genotype_matrix(), ["A", "C", "G", "T"]
            ),
            self.result_dataset(),
        )
