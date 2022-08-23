import msprime
import numpy as np

import phylokit as pk


class Test_Hartigan_Parsimony_Vectorised:
    def setup(self):
        ts = msprime.sim_ancestry(10, sequence_length=100, random_seed=1234)
        mts = msprime.sim_mutations(ts, rate=0.01, random_seed=5678)
        pk_mts = pk.from_tskit(mts.first())
        return pk_mts, mts

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
