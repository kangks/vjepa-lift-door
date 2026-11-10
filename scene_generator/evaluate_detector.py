#!/usr/bin/env python3
"""Simple frame-aligned anomaly evaluation. No V-JEPA inference is done here.
Input scores CSV columns: sample_index,anomaly_score
"""
import argparse, csv, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True)
p.add_argument('--scores',required=True,help='CSV columns: sample_index,anomaly_score')
p.add_argument('--threshold',type=float,default=0.7)
a=p.parse_args()
m=json.loads(Path(a.manifest).read_text())
samples={int(r['sample_index']):r for r in m['samples']}
with open(a.scores,newline='') as f:
    scores=list(csv.DictReader(f))
if not scores: p.error('No detector scores')
seen=set();tp=fp=tn=fn=nuisance_fp=0
for r in scores:
    k=int(r['sample_index'])
    if k in seen: p.error(f'Duplicate sample_index {k}')
    seen.add(k)
    if k not in samples:p.error(f'Score for unknown sample_index {k}')
    true=bool(samples[k]['mechanical_anomaly'])
    pred=float(r['anomaly_score'])>=a.threshold
    if true and pred:tp+=1
    elif true and not pred:fn+=1
    elif not true and pred:
        fp+=1
        if samples[k]['environmental_labels']:nuisance_fp+=1
    else:tn+=1

def ratio(a,b):return round(a/b,4) if b else None
print(json.dumps({'evaluated_samples':len(scores),'threshold':a.threshold,
    'TP':tp,'FP':fp,'TN':tn,'FN':fn,
    'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
    'specificity':ratio(tn,tn+fp),'nuisance_false_positives':nuisance_fp,
    'caveat':'Frame-point labels vs model outputs; window-level V-JEPA scores require time/window alignment and an explicit aggregation policy.'}, indent=2))
