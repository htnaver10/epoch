"""Recover the epoch log if a process stops after checkpoint commit."""
import json
from .. import core


def reconcile_history(run_name,saved):
    path=core.path(f'logs/{run_name}/epochs.jsonl',True)
    records=core.read_jsonl(path) if path.exists() else []
    if [row['epoch'] for row in records]!=list(range(len(records))):raise ValueError('Noncontiguous epoch history')
    completed=saved['epoch']+1
    if len(records)==completed:return
    if len(records)==completed-1 and (saved.get('epoch_record') or {}).get('epoch')==saved['epoch']:
        with path.open('a') as stream:stream.write(json.dumps(saved['epoch_record'])+'\n')
        return
    raise ValueError('Epoch history and checkpoint disagree; preserve and inspect this run')
