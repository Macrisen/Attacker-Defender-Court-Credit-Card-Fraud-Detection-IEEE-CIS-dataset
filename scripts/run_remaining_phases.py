"""Resumable, gated research execution. Completed runs are immutable."""
import os,sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if sys.platform=='darwin' and not os.environ.get('DYLD_LIBRARY_PATH','').startswith('/opt/homebrew/opt/libomp/lib'):
    env=dict(os.environ);env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'+(':'+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
sys.path.insert(0,str(ROOT))
import rl_agents
import yaml

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['all','4a','4b','4c','5','6'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--config',default=str(ROOT/'configs/attacker_ppo.yaml'));p.add_argument('--max-steps',type=int)
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:config['seeds']=[args.seed]
    if args.max_steps is not None:
        if args.max_steps<1024:raise ValueError('--max-steps must be >=1024')
        config['attacker']['max_steps']=args.max_steps//1024*1024;config['court']['max_steps']=args.max_steps
        config['bandit']['training_queries']=max(1024,config['attacker']['max_steps']//3//1024*1024)
        config['run_label']='explicit_reduced_budget_pilot'
    phases=['4a','4b','4c','5','6'] if args.phase=='all' else [args.phase]
    if args.dry_run:
        print(json.dumps(dict(phases=phases,config=config,resume=args.resume,execution=False),indent=2));return
    import fcntl
    lock_path=ROOT/'reports/execution_control/runner.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('Another research runner is active. Wait for it to finish before resuming.')
    lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
    from scripts.remaining_pipeline import execute_phase
    for phase in phases:
        result=execute_phase(phase,config,args.resume)
        print(json.dumps(dict(phase=phase,**result)),flush=True)
        if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
