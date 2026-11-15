"""Explicit, resumable stage orchestration with a frozen training source lock."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from .. import core
from .data import prepare_pairs,fit_statistics,validate_pairs


def software_lock(config):
    root=core.ROOT
    files=sorted((root/'src/firefullres').glob('*.py'))+sorted((root/'src/firefullres/distill').glob('*.py'))
    files += [root/'tools'/file for file in ('run_teacher_student.py','teacher_student_stage.py',
        'evaluate_substantial_round.py','download_real_data.py')]
    result={'sources':{core.relative(p):core.sha256(p) for p in files},
        'teacher_config_sha256':core.sha256(config['teacher_config']),
        'student_config_sha256':core.sha256(config['student_config']),
        'parent_manifest_sha256':core.sha256(config['parent_manifest']),
        'student_statistics_sha256':core.sha256(config['student_statistics']),
        'baseline_checkpoint_sha256':core.sha256(config['baseline_checkpoint']),
        'data_versions':{p:importlib.metadata.version(p) for p in ('numpy','h5py','earthaccess')}}
    if result['baseline_checkpoint_sha256']!=config['baseline_checkpoint_sha256']:raise ValueError('Frozen supervised baseline changed')
    query="import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ['torch','numpy']}))"
    result['training_versions']=json.loads(subprocess.run([str(root/'.venv/Scripts/python.exe'),'-c',query],capture_output=True,text=True,check=True).stdout)
    path=f'results/{config["name"]}/software_lock.json'
    if core.path(path).exists() and core.read_json(path)!=result:
        raise ValueError('Teacher/student training software or data changed; preserve the experiment and create a new version')
    if not core.path(path).exists():core.write_json(path,result)
    return result


def run_stages(config,stage,config_path=None):
    name=config['name'];manifest=f'data/manifests/{name}_pairs.jsonl';stats=f'data/manifests/{name}_teacher_stats.json'
    if stage in ('all','prepare'):
        manifest=prepare_pairs(config);stats=fit_statistics(manifest,config)
        if stage=='prepare':return 0
    validate_pairs(manifest,config)
    if not core.read_json(f'results/{name}/paired_data_audit.json')['ready_for_teacher_training']:
        raise ValueError('Paired teacher data admission did not pass')
    software_lock(config)
    python=core.ROOT/'.venv/Scripts/python.exe';script=core.ROOT/'tools/teacher_student_stage.py'
    env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    def run(which):
        print(f'Starting teacher/student stage: {which}',flush=True)
        subprocess.run([str(python),str(script),which,'--config',config_path or f'configs/{name}.json'],check=True,env=env)
    if stage in ('all','teacher'):
        if not core.path(f'results/{name}/teacher_gpu_profile.json').exists():run('profile')
        run('teacher')
        if stage=='teacher':return 0
    if stage in ('all','cache'):
        epochs=core.read_json(config['teacher_config'])['epochs'];log=core.read_jsonl(f'logs/{name}_teacher/epochs.jsonl')
        if not log or log[-1]['epoch']+1!=epochs:raise ValueError('Finish the configured teacher run before caching targets')
        run('cache')
        if stage=='cache':return 0
    if stage in ('all','student'):
        run('student')
        if stage=='student':return 0
    if stage in ('all','evaluate'):
        epochs=core.read_json(config['student_config'])['epochs'];log=core.read_jsonl(f'logs/{name}_student/epochs.jsonl')
        if not log or log[-1]['epoch']+1!=epochs:raise ValueError('Finish the student training run before evaluation')
        run('evaluate')
    print(f'Completed teacher/student experiment. Read results/{name}/EVALUATION.md.',flush=True)
    return 0
