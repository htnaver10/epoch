"""Run the bounded expansion, preparation, thirty-epoch fit and held-out evaluation."""
import argparse
import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.campaign import Campaign
from download_real_data import local_credentials


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',choices=('all','inventory','download','prepare','train','evaluate'),default='all')
    parser.add_argument('--config',default='configs/substantial_round_v2.json')
    parser.add_argument('--no-prompt',action='store_true')
    args=parser.parse_args(argv);campaign=Campaign(args.config);cfg=campaign.config;name=cfg['name']
    print(f'{name}: {cfg["scope"]}',flush=True)
    print(f'Planned new-transfer budget: {cfg["download_budget_gib"]} GiB (metadata estimates); retain {cfg["reserve_free_gib"]} GiB free.',flush=True)
    if args.stage!='inventory':
        software_path=f'results/{name}/software_lock.json'
        sources=sorted((ROOT/'src/firefullres').glob('*.py'))+[ROOT/'tools'/p for p in
            ('run_substantial_round.py','evaluate_substantial_round.py','download_real_data.py')]
        software={'training_config_sha256':core.sha256(cfg['training_config']),
            'source_sha256':{core.relative(p):core.sha256(p) for p in sources},
            'data_versions':{p:importlib.metadata.version(p) for p in ('numpy','earthaccess','h5py','pyproj')}}
        version_query="import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ['torch','numpy']}))"
        software['training_versions']=json.loads(subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-c',version_query],
            check=True,capture_output=True,text=True).stdout)
        if core.path(software_path).exists() and core.read_json(software_path)!=software:
            raise ValueError('Campaign software/training configuration changed; preserve the run and create a new version')
        if not core.path(software_path).exists():core.write_json(software_path,software)
    if args.stage in ('all','inventory','download'):
        inventory=campaign.inventory()
        if args.stage=='inventory':
            print(json.dumps({key:{'candidate_days':len(value.get('candidates',[])),
                'existing_bundles':value.get('existing_bundles')} for key,value in inventory.items()},indent=2));return 0
    if args.stage in ('all','download'):
        if core.path(f'data/manifests/{name}_bundles.jsonl').exists():campaign.acquire()
        else:
            with local_credentials(not args.no_prompt,'run-substantial-round.cmd'):campaign.acquire()
        if args.stage=='download':return 0
    manifest=f'data/manifests/{name}.jsonl';stats=f'data/manifests/{name}_stats.json'
    if args.stage in ('all','prepare'):
        manifest,stats=campaign.prepare()
        if args.stage=='prepare':return 0
    python=ROOT/'.venv/Scripts/python.exe'
    if not python.is_file():raise ValueError('CUDA training environment missing')
    if args.stage in ('all','train'):
        if not core.read_json(f'results/{name}/dataset_audit.json')['ready_for_expanded_experiment']:
            raise ValueError('Data admission checks have not passed')
        profile=f'results/{name}/rtx_profile.json'
        if not core.path(profile).exists():
            subprocess.run([str(python),str(ROOT/'run.py'),'profile','--manifest',manifest,'--stats',stats,
                '--config',cfg['training_config'],'--output',profile,'--steps','5','--warmup-steps','2','--worst-case'],check=True)
        elif core.read_json(profile)['manifest_sha256']!=core.sha256(manifest):raise ValueError('Profile belongs to another dataset')
        run_folder=core.path(f'models/{name}');last=run_folder/'last.pt'
        if run_folder.exists() and not last.exists():
            log=core.path(f'logs/{name}/epochs.jsonl')
            if any(run_folder.glob('*.pt')) or log.exists() and log.stat().st_size:
                raise ValueError('Run history exists without last.pt; inspect it before selecting a new run')
            archive=core.path(f'.cache/{name}/uncheckpointed_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}')
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not run_folder.resolve().is_relative_to(core.ROOT) or not archive.resolve().is_relative_to(core.ROOT):
                raise ValueError('Run archive must remain inside the research folder')
            run_folder.rename(archive)
        command=[str(python),str(ROOT/'run.py'),'train','--manifest',manifest,'--stats',stats,
            '--config',cfg['training_config'],'--run',name,'--epochs',str(cfg['epochs'])]
        if last.exists():command.append('--resume')
        print('Starting/resuming GPU training. Epoch summaries follow.',flush=True)
        subprocess.run(command,check=True)
        if args.stage=='train':return 0
    if args.stage in ('all','evaluate'):
        subprocess.run([str(python),str(ROOT/'tools/evaluate_substantial_round.py'),'--config',args.config],check=True)
    print(f'Completed. Inspect results/{name}/evaluation.json before judging benefit.',flush=True);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (KeyboardInterrupt,EOFError):print('\nStopped; download and completed-source preparation journals are retained.',file=sys.stderr);raise SystemExit(130)
    except Exception as error:
        # Provider failures can include signed URLs. Keep credentials out of the
        # wrapper's diagnostics; local library output remains visible in terminal.
        if isinstance(error,(ValueError,RuntimeError)):
            print(f'Round stopped: {error}',file=sys.stderr)
        else:print(f'Round stopped ({type(error).__name__}); inspect local stage output.',file=sys.stderr)
        raise SystemExit(2)
