"""Reproduce saved submission probabilities and whole-batch decisions without refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from day5_delivery import replay_submission


def main(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    saved=pd.read_csv(root/'submission/submission.csv',float_precision='round_trip').set_index('application_id')
    reproduced=replay_submission(root).set_index('application_id').loc[saved.index]
    if not np.allclose(saved.probability,reproduced.probability,rtol=0,atol=1e-12) or not np.array_equal(saved.decision,reproduced.decision):
        raise ValueError('Reproduced predictions or decisions differ from the saved submission.')
    print('REPLAY_MATCH: all challenge probabilities and whole-batch decisions reproduced.')


if __name__=='__main__':
    main()
