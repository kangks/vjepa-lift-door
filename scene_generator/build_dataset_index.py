#!/usr/bin/env python3
"""Build a cross-scenario table, including videos not yet rendered."""
import argparse, csv, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root)
records=[]
for path in sorted(root.glob('*/*/ground_truth.json')):
    gt=json.loads(path.read_text());video=path.parent/gt['video_file']
    records.append({'scenario':gt['scenario'],'video_health':gt['video_health'],
       'video':str(video.relative_to(root)),'video_exists':video.is_file(),
       'ground_truth':str(path.relative_to(root)),
       'fault_families':';'.join(gt['fault_families']),
       'disturbances':';'.join(gt['disturbance_types']),
       'fps':gt['fps'],'duration_sec':gt['duration_sec'],
       'resolution':f"{gt['resolution'][0]}x{gt['resolution'][1]}",
       'anomalous_frames':sum(bool(x['anomaly']) for x in gt['frames']),
       'unobservable_frames':sum(not x['observable'] for x in gt['frames'])})
(root/'dataset_index.json').write_text(json.dumps(records,indent=2))
if records:
    with (root/'dataset_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print(f'Indexed {len(records)} scenarios')
