"""Rebuild the final course pipeline from released data and recorded configuration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from day5_final import (FinalConfig, split_final_roles, nested_comparison, worth_it, selected_oof,
    threshold_sweep, select_threshold, fit_final, export_model, predict_final,
    transport_threshold, apply_batch_policy)


def main(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    run=json.loads((root/'artifacts/day5_run.json').read_text(encoding='utf-8'))
    if run['source']!='LIVE':
        raise ValueError('Rebuild requires your live run, not a saved educational example.')
    config=FinalConfig(**run['config'])
    contract=json.loads((root/'data/data_contract.json').read_text(encoding='utf-8'))
    train=pd.read_csv(root/'data/tamweel_train.csv'); challenge=pd.read_csv(root/'data/tamweel_challenge.csv')
    pool,cal,_=split_final_roles(train)
    oof,scores,_=nested_comparison(pool,contract,config)
    _,gate=worth_it(scores,config)
    if gate['chosen']!=run['chosen']:
        raise ValueError('Rebuilt model choice changed; compare environment and source versions.')
    threshold=select_threshold(threshold_sweep(selected_oof(oof,gate['chosen'])))['threshold']
    candidate,mapping,_,_=fit_final(pool,cal,contract,gate['chosen'],config)
    directory=root/'rebuild_check/final_model'; export_model(candidate,mapping,contract,directory)
    reproduced,_=apply_batch_policy(predict_final(challenge,directory),transport_threshold(threshold,mapping))
    saved=pd.read_csv(root/'submission/submission.csv',float_precision='round_trip').set_index('application_id')
    reproduced=reproduced.set_index('application_id').loc[saved.index]
    if not np.allclose(saved.probability,reproduced.probability,rtol=0,atol=1e-9) or not np.array_equal(saved.decision,reproduced.decision):
        raise ValueError('Rebuild differs. Check pinned environment before changing any decision rule.')
    print('REBUILD_MATCH: live training, selection, calibration and all challenge decisions reproduced.')


if __name__=='__main__':
    main()
