import os,sys,json,subprocess,hashlib,shutil,tempfile,csv,html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
root=Path.cwd();batch=root/'results'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f');batch.mkdir()
Path('/tmp/cartpole_comparison_path').write_text(str(batch))
shutil.copy2(__file__,batch/'comparison_source.py')
files=sorted((root/'configs').glob('pid_[0-9][0-9]_*.json'))
assert len(files)==17
for p in files:
 a=json.loads(p.read_text());b=json.loads(p.with_name(p.name.replace('pid_','lqr_',1)).read_text())
 for k in ['parameters','initial_state','simulation','output']:assert a[k]==b[k]
 assert a['control']['sample_period_s']==b['control']['sample_period_s']
 assert a['control']['settings']['force_limit_N']==b['control']['settings']['force_limit_N']==20
 assert a['control']['settings']['tuning_poles']==[-1,-1.2,-1.5,-2,-2.5]
files=[q for p in files for q in (p,p.with_name(p.name.replace('pid_','lqr_',1)))]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[root/'feedback.py',root/'cartpole_model.py',root/'cartpole_simulation.py',root/'run_simulation.py',root/'cartpole_visualization.py']}
(batch/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
cache=tempfile.mkdtemp(prefix='cartpole-mpl-');env={**os.environ,'MPLCONFIGDIR':cache,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
print('BATCH',batch,flush=True)
def run(p):
 proc=subprocess.run([sys.executable,'run_simulation.py','--config',str(p)],cwd=root,env=env,text=True,capture_output=True)
 (batch/(p.stem+'.log')).write_text(proc.stdout+proc.stderr)
 if proc.returncode:raise RuntimeError(p.name+': '+proc.stderr)
 directory=Path(proc.stdout.split('结果目录：')[-1].strip())
 assert (directory/'config.json').read_bytes()==p.read_bytes()
 assert (directory/'feedback_source.py').read_bytes()==(root/'feedback.py').read_bytes()
 assert json.loads((directory/'metadata.json').read_text())['run']['status']=='completed'
 for name in ['response.png','animation.mp4']:assert (directory/name).stat().st_size>1000
 x=np.genfromtxt(directory/'trajectory.csv',delimiter=',',names=True);u=np.genfromtxt(directory/'control.csv',delimiter=',',names=True)
 assert len(x)==2001 and len(u)==1000
 cols=['position_m','velocity_m_s','angle_deg','angular_velocity_deg_s'];values=np.column_stack([x[k] for k in cols]);assert np.all(np.isfinite(values))
 good=np.all(np.abs(values)<[.01,.01,.1,.1],axis=1);bad=np.flatnonzero(~good)
 settle=float(x['time_s'][bad[-1]+1]) if len(bad) and bad[-1]+1<len(x) else (0. if not len(bad) else None)
 success=bool(np.all(good[x['time_s']>=15]))
 force=u['force_N'];dt=np.diff(np.r_[u['time_s'],20.]);assert np.max(abs(force))<=20+1e-10
 return dict(config=p.name,controller=p.name[:3].upper(),case=p.stem[4:],directory=directory.name,status='recovered' if success else 'late' if settle is not None else 'not_recovered',settling_time_s=settle,max_position_m=float(max(abs(values[:,0]))),max_angle_deg=float(max(abs(values[:,2]))),force_rms_N=float(np.sqrt(np.average(force**2,weights=dt))),force_saturation_fraction=float(np.average(abs(force)>=20-1e-9,weights=dt)),max_energy_error_J=float(max(abs(x['energy_balance_error_J']))),final_state=values[-1].tolist())
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for future in as_completed({pool.submit(run,p):p for p in files}):
  r=future.result();rows.append(r);rows.sort(key=lambda r:(r['case'],r['controller']))
  (batch/'summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
  print(len(rows),r['config'],r['status'],r['settling_time_s'],flush=True)
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
with (batch/'summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('DONE',batch,flush=True)
