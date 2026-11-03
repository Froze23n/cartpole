from pathlib import Path
import json,html,subprocess,os,tempfile
import numpy as np
os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='compare-plot-')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=Path('/tmp/cartpole_comparison_path').read_text().strip();folder=Path(folder)
rows=json.loads((folder/'summary.json').read_text());assert len(rows)==34
cases=sorted(set(r['case'] for r in rows));lookup={(r['controller'],r['case']):r for r in rows}
probe=Path('/opt/miniconda3/envs/cartpole/bin/ffprobe');videos=[]
for r in rows:
 result=subprocess.run([str(probe),'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate,width,height','-of','json',str(folder.parent/r['directory']/'animation.mp4')],capture_output=True,text=True,check=True)
 v=json.loads(result.stdout)['streams'][0];assert v['codec_name']=='h264' and v['nb_frames']=='401' and v['r_frame_rate']=='20/1'
 videos.append(dict(config=r['config'],**v))
(folder/'video_checks.json').write_text(json.dumps(videos,indent=2))
fig,ax=plt.subplots(1,2,figsize=(14,7))
y=np.arange(17)
for method,shift,color in [('PID',-.18,'#2878b5'),('LQR',.18,'#e68a2e')]:
 rr=[lookup[method,c] for c in cases];times=[r['settling_time_s'] if r['settling_time_s'] is not None else 20 for r in rr]
 ax[0].barh(y+shift,times,height=.32,label=method,color=color)
 for i,r in enumerate(rr):
  if r['status']!='recovered':ax[0].text(times[i]+.1,i+shift,'late' if r['status']=='late' else 'not settled',va='center',fontsize=7)
 good=[r['max_position_m'] if r['status']=='recovered' else np.nan for r in rr]
 ax[1].barh(y+shift,good,height=.32,label=method,color=color)
for a in ax:
 a.set_yticks(y,[c.split('_',1)[1] for c in cases]);a.invert_yaxis();a.grid(axis='x',alpha=.2);a.legend()
ax[0].axvline(15,color='gray',linestyle='--',linewidth=1);ax[0].set_xlim(0,27);ax[0].set_xlabel('Settling time [s]; dashed line: last-5-s criterion')
ax[1].set_xlabel('Peak |cart position| [m]; recovered trials only')
fig.suptitle('Fixed-parameter PID vs LQR | nonlinear plant, 20 s, 20 N limit')
fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
for case in ['06_angle_30deg','09_angle_45deg','10_angle_50deg']:
 fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
 for method in ['PID','LQR']:
  r=lookup[method,case];x=np.genfromtxt(folder.parent/r['directory']/'trajectory.csv',delimiter=',',names=True)
  for a,col in zip(axes,['angle_deg','position_m','force_N']):a.plot(x['time_s'],x[col],label=method)
 for a,label in zip(axes,['Angle [deg]','Position [m]','Force [N]']):a.set_ylabel(label);a.grid(alpha=.2);a.legend()
 axes[-1].set_xlabel('Time [s]');fig.suptitle(case);fig.tight_layout();fig.savefig(folder/(case+'.png'),dpi=150);plt.close(fig)
status={'recovered':'恢复','late':'恢复较晚','not_recovered':'20 s内未恢复'}
parts=['<!doctype html><meta charset="utf-8"><title>PID/LQR固定参数实验汇总</title><style>body{font:16px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse}td,th{padding:7px 12px;border:1px solid #ccc}img{max-width:100%}</style><h1>17对 PID/LQR 固定参数实验</h1><p>原连续非线性对象；M=m=1 kg，L=0.5 m，20 s，采样0.02 s，力上限20 N。配置和反馈源码校验一致，未调参。</p><p>恢复：最后5 s全部记录满足 |p|&lt;0.01 m、|v|&lt;0.01 m/s、|θ|&lt;0.1°、|ω|&lt;0.1°/s。恢复时间为进入并保持至结束的时刻；较晚进入阈值单列。角度沿用原始连续角度，未折返。无限轨道，恢复不等于满足实物行程约束。</p><img src="comparison.png"><table><tr><th>初态</th><th>控制器</th><th>结果</th><th>恢复时间/s</th><th>最大|位置|/m</th><th>力RMS/N</th><th>力饱和占比</th><th>归档</th></tr>']
for c in cases:
 for method in ['PID','LQR']:
  r=lookup[method,c];d='../'+r['directory'];t='—' if r['settling_time_s'] is None else f"{r['settling_time_s']:.2f}"
  parts.append(f'<tr><td>{c}</td><td>{method}</td><td>{status[r["status"]]}</td><td>{t}</td><td>{r["max_position_m"]:.3f}</td><td>{r["force_rms_N"]:.3f}</td><td>{r["force_saturation_fraction"]:.1%}</td><td><a href="{d}/response.png">曲线</a> · <a href="{d}/animation.mp4">视频</a> · <a href="{d}/config.json">配置</a> · <a href="{d}/trajectory.csv">数据</a></td></tr>')
parts.append('</table><p><a href="summary.csv">完整指标CSV</a> · <a href="summary.json">JSON及目录映射</a> · <a href="video_checks.json">视频检查</a></p>')
(folder/'index.html').write_text('\n'.join(parts))
for c in cases:
 print(c,[(m,lookup[m,c]['status'],lookup[m,c]['settling_time_s'],round(lookup[m,c]['max_position_m'],3),round(lookup[m,c]['force_rms_N'],3)) for m in ['PID','LQR']])
print('SUMMARY',folder)
