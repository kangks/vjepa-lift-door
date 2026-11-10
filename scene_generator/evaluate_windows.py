#!/usr/bin/env python3
"""Evaluate V-JEPA window scores, excluding windows with full camera occlusion by default."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--scores',required=True)
p.add_argument('--threshold',type=float,default=.7);p.add_argument('--label-policy',choices=['any','majority'],default='any')
p.add_argument('--include-unobservable',action='store_true');a=p.parse_args()
samples=json.loads(Path(a.manifest).read_text())['samples'];scores=list(csv.DictReader(open(a.scores)))
tp=fp=tn=fn=skipped=0;report=[]
for row in scores:
    first=int(row['window_start_sample']);last=int(row['window_end_sample_exclusive'])
    if not 0<=first<last<=len(samples):p.error('invalid sample indices')
    w=samples[first:last];observed=all(x['observable'] for x in w)
    if not observed and not a.include_unobservable:skipped+=1;continue
    faulty=sum(bool(x['anomaly']) for x in w)
    truth=(faulty>0 if a.label_policy=='any' else faulty>len(w)/2)
    score=float(row['anomaly_score']);prediction=score>=a.threshold
    tp+=int(truth and prediction);fp+=int(not truth and prediction)
    tn+=int(not truth and not prediction);fn+=int(truth and not prediction)
    report.append({'start':first,'end_exclusive':last,'truth':truth,'predicted':prediction,
                   'score':score,'observable':observed})
def ratio(n,d):return round(n/d,4) if d else None
print(json.dumps({'TP':tp,'FP':fp,'TN':tn,'FN':fn,'skipped':skipped,
 'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
 'specificity':ratio(tn,tn+fp),'windows':report},indent=2))
