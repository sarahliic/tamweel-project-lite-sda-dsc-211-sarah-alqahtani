"""Checksummed learner bundles and explicit evidence completeness; never automatic grading."""
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
import zipfile

import numpy as np
import pandas as pd
from day5_final import write_json, predict_final, apply_batch_policy
from day3_decision import CostPolicy

RESPONSES = ('ensemble_reason','validation_limits','calibration_limits','capacity_fairness',
             'explanation_scope','intended_use','monitoring_plan','executive_summary_ar','executive_summary_en')


def reflection_check(responses,source='LIVE'):
    missing=[key for key in RESPONSES if not isinstance(responses.get(key),str) or not responses[key].strip()]
    return {'status':'EXAMPLE_ONLY_NOT_SUBMITTABLE' if source!='LIVE' else 'LEARNER_WORK_REQUIRED' if missing else 'READY_FOR_REVIEW',
            'missing':missing,'automatic_grade':None,'content_quality':'requires human review'}


def safe_relative(name):
    p=PurePosixPath(name)
    if (not isinstance(name,str) or not name or '\\' in name or ':' in name or p.is_absolute()
            or '..' in p.parts or '.' in p.parts or str(p)!=name or any(part.startswith('.') for part in p.parts)):
        raise ValueError('Only normalized, relative project paths are allowed.')
    return p


def import_daily_evidence(archive,destination):
    """Extract a learner-selected daily export to its own evidence folder, never over code."""
    destination=Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist(); names=[e.filename for e in entries]
        if len(entries)>80 or len(set(names))!=len(names) or sum(e.file_size for e in entries)>32*1024*1024:
            raise ValueError('Daily archive is too large or contains duplicate paths.')
        for e in entries:
            path=safe_relative(e.filename)
            if e.is_dir() or path.suffix.lower() not in {'.csv','.json','.png','.md','.npz','.txt'} or ((e.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Daily evidence may contain only tables, notes, figures and numeric artifacts.')
            target=destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Evidence path escapes destination.')
            if target.exists() and sha256(target.read_bytes()).digest()!=sha256(z.read(e)).digest():
                raise ValueError('Existing evidence differs; choose a separate folder instead of overwriting it.')
        for e in entries:
            target=destination.joinpath(*PurePosixPath(e.filename).parts); target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(e))
    return names


def evidence_status(root):
    root=Path(root); rows=[]
    for day in range(1,5):
        folder=root/'evidence'/f'day{day}'
        candidates=list(folder.rglob(f'day{day}_reflection.json')) if folder.exists() else []
        status='MISSING'
        if len(candidates)==1:
            item=json.loads(candidates[0].read_text(encoding='utf-8'))
            status=item.get('checkpoint',item).get('status','UNRECOGNIZED')
        rows.append({'item':f'day{day}_learner_evidence','status':status,
            'meaning':'presence and self-reported source only; authenticity and reasoning need human review'})
    for filename in ('notebooks/01_baseline_boosting.ipynb','notebooks/02_validation_tuning.ipynb',
                     'notebooks/03_cost_sensitive_decision.ipynb','notebooks/04_explain_calibrate.ipynb',
                     'notebooks/05_final_model.ipynb','presentation/final_presentation.pdf'):
        path=root/filename
        rows.append({'item':filename,'status':'PRESENT_REQUIRES_REVIEW' if path.is_file() and path.stat().st_size>0 else 'MISSING',
            'meaning':'file presence does not verify execution, content, grade or submission receipt'})
    return rows


def validate_submission(submission,challenge,threshold,policy=CostPolicy()):
    if list(submission)!=['application_id','probability','decision'] or len(submission)!=len(challenge):
        raise ValueError('Submission requires all challenge rows and the exact three columns.')
    if (submission.application_id.isna().any() or not submission.application_id.is_unique
            or set(submission.application_id)!=set(challenge.application_id)
            or not submission.decision.isin([0,1]).all()):
        raise ValueError('Submission IDs must match challenge exactly; decisions must be binary.')
    expected,audit=apply_batch_policy(submission[['application_id','probability']],threshold,policy)
    if not np.array_equal(expected.decision,submission.decision):
        raise ValueError('Decisions do not match the frozen full-batch policy.')
    return audit


def replay_submission(root):
    root=Path(root)
    policy=json.loads((root/'artifacts/final_policy.json').read_text(encoding='utf-8'))
    challenge=pd.read_csv(root/'data/tamweel_challenge.csv')
    predictions=predict_final(challenge,root/'artifacts/final_model')
    return apply_batch_policy(predictions,policy['calibrated_threshold'],CostPolicy(**policy['cost_policy']))[0]


def create_bundle(root,relative_files,metadata):
    root=Path(root).resolve(); files={}
    forbidden={'submission/submission_manifest.json','submission/project_bundle.zip'}
    for name in sorted(set(relative_files)):
        safe_relative(name)
        path=(root/name).resolve()
        if (name in forbidden or not path.is_relative_to(root) or not path.is_file()
                or (root/name).is_symlink()):
            raise ValueError('Bundle must use an explicit list of real project files, excluding its manifest and ZIP.')
        data=path.read_bytes(); files[name]={'bytes':len(data),'sha256':sha256(data).hexdigest()}
    manifest={'format_version':1,'scope':'Day 5 reproducible project plus explicitly imported learner evidence',
        'metadata':metadata,'files':files,
        'commit_note':'Course support revision is provenance. Record your final repository SHA/tag separately after committing this manifest; no self-referential SHA.',
        'verification_scope':'Checksums and mechanical completeness are not grades, authenticity proof or a submission receipt.'}
    manifest_path=root/'submission/submission_manifest.json'; write_json(manifest_path,manifest)
    zip_path=root/'submission/project_bundle.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(root/name,name)
        z.write(manifest_path,'submission/submission_manifest.json')
    verify_bundle(zip_path)
    return manifest,zip_path


def verify_bundle(archive):
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist(); names=[e.filename for e in entries]
        if len(names)>500 or len(set(names))!=len(names) or sum(e.file_size for e in entries)>100*1024*1024:
            raise ValueError('Unexpected bundle size or duplicate paths.')
        for e in entries:
            safe_relative(e.filename)
            if e.is_dir() or ((e.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Bundle entries must be regular files.')
        manifest=json.loads(z.read('submission/submission_manifest.json'))
        if set(names)!=set(manifest['files'])|{'submission/submission_manifest.json'}:
            raise ValueError('Archive membership differs from manifest.')
        for name,record in manifest['files'].items():
            data=z.read(name)
            if len(data)!=record['bytes'] or sha256(data).hexdigest()!=record['sha256']:
                raise ValueError(f'Bundle checksum mismatch: {name}')
    return {'status':'BUNDLE_BYTES_VERIFIED','files':len(manifest['files']),
        'sha256':sha256(Path(archive).read_bytes()).hexdigest()}
