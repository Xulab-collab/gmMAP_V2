import numpy as np
import pandas as pd

from gmmap.core.signatures import build_signatures
from gmmap.core.scoring import score_signatures


def test_toy_scoring_runs():
    expr = pd.DataFrame(
        np.array([[5, 0, 1, 0], [0, 4, 0, 1], [3, 1, 2, 0]], dtype=float),
        index=["c1", "c2", "c3"],
        columns=["G1", "G2", "G3", "G4"],
    )
    z = pd.DataFrame({"TraitA": [3.0, -2.5, 1.5, -1.2]}, index=["G1", "G2", "G3", "G4"])
    sigs = build_signatures(z, top_n=2, min_valid_genes=1)
    up, down, net, _ = score_signatures(expr, sigs, method="smrs", stage="S3")
    assert up.shape == (3, 1)
    assert down.shape == (3, 1)
    assert net.shape == (3, 1)
