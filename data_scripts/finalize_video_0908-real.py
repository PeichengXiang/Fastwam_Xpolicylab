import json,time
from pathlib import Path
root=Path('/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/data/bench_v5_0908-real')
while not all((root/f'conversion_shard_{i}_of_4_0908-real.json').exists() for i in range(4)):
 time.sleep(30)
m=json.loads((root/'meta/source_manifest_0908-real.json').read_text())
count=0
for ep in m['episodes']:
 i=ep['episode_index']
 for cam in ['cam_high','cam_left_wrist','cam_right_wrist']:
  p=root/f'videos/chunk-{i//1000:03d}/observation.images.{cam}/episode_{i:06d}.mp4'
  r=json.loads(p.with_suffix('.verified.json').read_text());assert p.stat().st_size>0 and r['frames']==ep['length'];count+=1
m.update(status='complete',video_count=count,all_parquet_action_state_verified=True,all_video_frame_counts_verified=True)
(root/'meta/source_manifest_0908-real.json').write_text(json.dumps(m,indent=2)+'\n')
result={'status':'complete','episodes':len(m['episodes']),'frames':m['frame_count'],'videos':count,'action_source':'raw HDF5 action/* at t without shift','state_source':'raw HDF5 state/* at t without shift','finished_time':time.time()}
p=root/'conversion_complete_0908-real.json';p.with_suffix('.tmp').write_text(json.dumps(result,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
print(json.dumps(result),flush=True)
