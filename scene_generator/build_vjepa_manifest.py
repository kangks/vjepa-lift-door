#!/usr/bin/env python3
"""Align 24fps synthetic frame labels to candidate 4fps V-JEPA PNGs."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--ground-truth',required=True);p.add_argument('--frames-dir',required=True)
p.add_argument('--sample-fps',type=float,default=4)
a=p.parse_args();gt=json.loads(Path(a.ground_truth).read_text());sf=gt['fps']
if a.sample_fps<=0 or a.sample_fps>sf:p.error('invalid sample FPS')
pics=sorted(Path(a.frames_dir).glob('clip_*.png'))
if not pics:p.error('no clip_*.png')
records=[]
for k,pic in enumerate(pics):
    idx=min(len(gt['frames'])-1,int(round(k*sf/a.sample_fps)))
    r=gt['frames'][idx]
    records.append({'sample_index':k,'image':pic.name,'sample_time_sec':round(k/a.sample_fps,6),
      'source_frame_index_0based':idx,'source_time_sec':r['time_sec'],
      'anomaly':r['anomaly'],'mechanical_anomaly':r['mechanical_anomaly'],
      'observable':r['observable'],'mechanical_fault_labels':r['mechanical_fault_labels'],
      'electrical_fault_labels':r['electrical_fault_labels'],
      'environmental_labels':r['environmental_labels'],
      'door_sync_error_m':r['door_sync_error_m'],
      'interior_light_dip_fraction':r['interior_light_dip_fraction']})
out=Path(a.frames_dir)
(out/'vjepa_sample_manifest.json').write_text(json.dumps({'scenario':gt['scenario'],
 'video_health':gt['video_health'],'sample_fps':a.sample_fps,'source_fps':sf,
 'sampling_assumption':'constant FPS PTS starts at 0; exact indexing at 24 -> 4 FPS',
 'samples':records},indent=2))
with (out/'vjepa_sample_manifest.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader()
    for r in records:
        r=r.copy()
        for n in ['mechanical_fault_labels','electrical_fault_labels','environmental_labels']:
            r[n]=';'.join(r[n])
        w.writerow(r)
print('Sampled %d frames for %s'%(len(records),gt['scenario']))
