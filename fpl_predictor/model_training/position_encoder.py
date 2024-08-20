import numpy as np
from sklearn.preprocessing import OneHotEncoder


def position_encoder() -> OneHotEncoder:
    positions = np.array([["GKP", "DEF", "MID", "FWD"]]).transpose()
    return OneHotEncoder(sparse_output=False).fit(positions)
