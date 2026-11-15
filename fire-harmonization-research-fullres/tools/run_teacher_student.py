"""Run VIIRS teacher training, frozen soft targets and MODIS-only distillation."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.distill.acquisition import inventory,download
from download_real_data import local_credentials


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=('all','inventory','download','prepare','teacher','cache','student','evaluate'))
    parser.add_argument('--config',default='configs/teacher_student_v1.json')
    parser.add_argument('--no-prompt',action='store_true')
    args=parser.parse_args(argv)
    config=core.read_json(args.config);name=config['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Use an alphanumeric experiment name')
    lock=f'results/{name}/config.json'
    if core.path(lock).exists() and core.read_json(lock)!=config:raise ValueError('Teacher/student configuration changed; use a new version')
    if not core.path(lock).exists():core.write_json(lock,config)
    print(f'{name}: {config["scope"]}',flush=True)
    if args.stage in ('all','inventory','download'):
        rows=inventory(config)
        if args.stage=='inventory':return 0
        journal_path=f'results/{name}/l1b_downloads.json'
        journal=core.read_json(journal_path) if core.path(journal_path).exists() else {}
        complete=all(r['granule_id'] in journal and core.path(journal[r['granule_id']]['path']).is_file() for r in rows)
        if complete:download(config)
        else:
            with local_credentials(not args.no_prompt,'run-teacher-student.cmd'):download(config)
        if args.stage=='download':return 0
    from firefullres.distill.pipeline import run_stages
    return run_stages(config,args.stage,args.config)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (KeyboardInterrupt,EOFError):print('Stopped; completed downloads and checkpoints retained.',file=sys.stderr);raise SystemExit(130)
    except Exception as error:
        print('Teacher/student run stopped: '+(str(error) if isinstance(error,(ValueError,RuntimeError)) else type(error).__name__),file=sys.stderr)
        raise SystemExit(2)
