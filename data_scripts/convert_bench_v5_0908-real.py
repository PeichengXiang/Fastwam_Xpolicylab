#!/usr/bin/env python3
"""Format-only HDF5 -> LeRobot conversion. Explicit raw action and state, no temporal shift."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,subprocess,hashlib,concurrent.futures,argparse
from pathlib import Path
import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq
RAW=Path('/personal/xspark_shared/hand_data/hdf5/spark0_real/bench_v5')
OUT=Path('/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/data/bench_v5_0908-real')
CODE=Path('/personal/xiangpc/training_0908-real/fastwam_0908-real/code_0908-real')
sys.path.insert(0,str(CODE))
from XPolicyLab.utils.process_data import decode_image_bit,get_robot_action_dim_info,pack_robot_state
CAMS={'cam_high':'cam_head','cam_left_wrist':'cam_left_wrist','cam_right_wrist':'cam_right_wrist'}
ROBOT=get_robot_action_dim_info('tianji_marvin_wuji')
DIM=sum(ROBOT['arm_dim'])+sum(ROBOT['ee_dim'])
assert DIM==54

def dump(p,o):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,indent=2)+'\n');tmp.replace(p)
def vector(h,kind):
 vals={k:h[kind][k][:] for k in h[kind]}
 return pack_robot_state({kind:vals},'joint',ROBOT,source_type='dataset',state_type=kind).astype(np.float32)
def stats(a):
 return {'min':a.min(axis=0).tolist(),'max':a.max(axis=0).tolist(),'mean':a.mean(axis=0,dtype=np.float64).tolist(),'std':a.std(axis=0,dtype=np.float64).tolist(),'count':[len(a)]}
def materialize(job):
 i,task,taskidx,path,start=job
 path=Path(path)
 with h5py.File(path,'r') as h:
  a=vector(h,'action');s=vector(h,'state');n=len(a)
  assert a.shape==s.shape==(n,DIM) and np.isfinite(a).all() and np.isfinite(s).all(),str(path)
  fps=int(h['additional_info/frequency'][()]);assert fps==30,(str(path),fps)
  instr=h['instruction'][()].decode().rstrip('\0')
  shapes={k:list(map(int,h[f'vision/{v}/shape'][:])) for k,v in CAMS.items()}
  for c in CAMS.values():assert len(h[f'vision/{c}/colors'])==n
  data={'observation.state':pa.FixedSizeListArray.from_arrays(pa.array(s.ravel()),DIM),'action':pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()),DIM),'timestamp':pa.array(np.arange(n,dtype=np.float32)/fps),'frame_index':pa.array(np.arange(n,dtype=np.int64)),'episode_index':pa.array(np.full(n,i,dtype=np.int64)),'index':pa.array(np.arange(start,start+n,dtype=np.int64)),'task_index':pa.array(np.full(n,taskidx,dtype=np.int64))}
  table=pa.table(data)
  features={k:({'feature':{'dtype':'float32','_type':'Value'},'length':DIM,'_type':'Sequence'} if k in ['action','observation.state'] else {'dtype':'float32' if k=='timestamp' else 'int64','_type':'Value'}) for k in data}
  table=table.replace_schema_metadata({b'huggingface':json.dumps({'info':{'features':features}}).encode()})
  dest=OUT/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,str(dest)+'.tmp');Path(str(dest)+'.tmp').replace(dest)
  # Reload every numeric row and assert exact float32 conversion from the designated HDF5 sources.
  read=pq.read_table(dest,columns=['action','observation.state'])
  assert np.array_equal(np.asarray(read['action'].to_pylist(),dtype=np.float32),a)
  assert np.array_equal(np.asarray(read['observation.state'].to_pylist(),dtype=np.float32),s)
  return {'episode_index':i,'task':task,'task_index':taskidx,'source':str(path),'length':n,'frames':n,'tasks':[instr],'instruction':instr,'fps':fps,'shapes':shapes,'action_float32_max_abs_error':0.0,'state_float32_max_abs_error':0.0,'action_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'state_sha256':hashlib.sha256(s.tobytes()).hexdigest(),'stats':{'action':stats(a),'observation.state':stats(s)}}
def encode(job):
 i,task,taskidx,path,start=job
 import cv2,fcntl
 lockpath=OUT/f'locks_0908-real/episode_{i:06d}.lock'
 lockpath.parent.mkdir(parents=True,exist_ok=True)
 lockfile=lockpath.open('a')
 fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX)
 cv2.setNumThreads(1)
 with h5py.File(path,'r') as h:
  n=len(h['action/left_arm_joint_states']);report={}
  for key,camera in CAMS.items():
   dst=OUT/f'videos/chunk-{i//1000:03d}/observation.images.{key}/episode_{i:06d}.mp4'
   marker=dst.with_suffix('.verified.json')
   if dst.exists() and marker.exists(): report[key]='existing';continue
   dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.stem+'.partial.mp4')
   height,width,channels=map(int,h[f'vision/{camera}/shape'][:]);assert channels==3
   args=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(tmp)]
   proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
   try:
    colors=h[f'vision/{camera}/colors']
    for off in range(0,n,16):
     frames=np.asarray(decode_image_bit(colors[off:off+16]))
     assert frames.shape==(min(16,n-off),height,width,3) and frames.dtype==np.uint8
     proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close();err=proc.stderr.read().decode();rc=proc.wait();assert rc==0,err
   except BaseException:
    proc.kill();proc.wait();raise
   check=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,r_frame_rate,start_time','-of','json',str(tmp)]))['streams'][0]
   assert int(check['nb_frames'])==n,(i,key,check,n)
   assert float(check.get('start_time','0'))==0.0,check
   tmp.replace(dst);dump(marker,{'frames':n,'raw_source':path,'camera':camera,'rgb_decoder':'XPolicyLab.utils.process_data.decode_image_bit','codec':'libx264 crf18 ultrafast','source_shape':[height,width,3],**check})
   report[key]='encoded'
 return {'episode_index':i,'task':task,'frames':n,'videos':report}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=48);parser.add_argument('--video-only',action='store_true');parser.add_argument('--reverse',action='store_true');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((RAW/'conversion_manifest.json').read_text());tasks=sorted(manifest['task_counts']);jobs=[];offset=0
 for tid,task in enumerate(tasks):
  fs=sorted((RAW/task).glob('episode_*.hdf5'));assert len(fs)==manifest['task_counts'][task]
  for p in fs:
   with h5py.File(p,'r') as h:n=len(h['action/left_arm_joint_states'])
   jobs.append((len(jobs),task,tid,str(p),offset));offset+=n
 print('INVENTORY',len(jobs),'frames',offset,flush=True)
 if not args.video_only:
  results=[]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
   for r in ex.map(materialize,jobs):
    results.append(r)
    if len(results)%100==0:print('PARQUET',len(results),flush=True)
  template=Path('/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/data/spark0_mnt20260812_fastwam_collect_objects_ep000_049_v21_joint54/meta/info.json')
  info=json.loads(template.read_text());info.update(total_episodes=len(jobs),total_frames=offset,total_tasks=len(tasks),total_videos=len(jobs)*3,total_chunks=(len(jobs)+999)//1000,fps=30,splits={'train':f'0:{len(jobs)}'})
  for k,v in CAMS.items():
   h,w,c=results[0]['shapes'][k];feat=info['features'][f'observation.images.{k}'];feat['shape']=[c,h,w];feat['info'].update({'video.height':h,'video.width':w,'video.fps':30})
  dump(OUT/'meta/info.json',info)
  (OUT/'meta/episodes.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['episode_index','tasks','length']})+'\n' for r in results))
  (OUT/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['episode_index','stats']})+'\n' for r in results))
  (OUT/'meta/tasks.jsonl').write_text(''.join(json.dumps({'task_index':i,'task':manifest['task_instructions'][task]})+'\n' for i,task in enumerate(tasks)))
  dump(OUT/'meta/source_manifest_0908-real.json',{'source_root':str(RAW),'output_root':str(OUT),'action_source':'raw HDF5 action/* at t; no shift','state_source':'raw HDF5 state/* at t; no shift','joint_order':['left_arm7','left_hand20','right_arm7','right_hand20'],'episodes':[{k:v for k,v in r.items() if k!='stats'} for r in results],'episode_count':len(jobs),'frame_count':offset,'status':'parquet_complete_videos_pending'})
  print('ALL_PARQUET_READY',flush=True)
 done=0;started=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures={ex.submit(encode,j):j[0] for j in (list(reversed(jobs)) if args.reverse else jobs) if j[0]%args.shards==args.shard}
  for f in concurrent.futures.as_completed(futures):
   r=f.result();done+=1
   with (OUT/'video_progress_0908-real.jsonl').open('a') as log:log.write(json.dumps(r)+'\n')
   print('VIDEO_EPISODE',done,len(jobs),'ep',r['episode_index'],'seconds',round(time.time()-started,1),flush=True)
 dump(OUT/f'conversion_shard_{args.shard}_of_{args.shards}_0908-real.json',{'episodes':len(jobs),'frames':offset,'video_count':len(jobs)*3,'source_root':str(RAW),'status':'complete','action_source':'HDF5 action/* without temporal shift','state_source':'HDF5 state/* without temporal shift'})
 print('CONVERSION_COMPLETE',flush=True)
if __name__=='__main__':main()
