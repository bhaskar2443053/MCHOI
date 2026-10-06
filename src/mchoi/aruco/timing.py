"""Estimate constant RGB timestamp offsets from matching decoded marker corners.

Fit on alternating 2-second blocks, report held-out blocks separately. The original
PC timestamps and calibration remain unchanged. Positive lag means a matching
RealSense image has a later saved PC timestamp than the ZED image.
"""
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def estimate(take, calibration, output, reference_camera='zed_left', reference_lag_ns=0):
    cal = json.loads(calibration.read_text())['cameras']
    serials = {v['device']['serial']: k for k,v in cal.items() if 'serial' in v['device']}
    observations = {}
    selection=output/'marker_selection.json'
    selected=set(json.loads(selection.read_text())['marker_ids']) if selection.exists() else set(range(50))
    for path in (take/'aruco_annotated').glob('*detections.csv'):
        for row in csv.DictReader(path.open()):
            if row['detected'] != '1' or int(row['marker_id']) not in selected: continue
            cam = 'zed_'+row['view'] if 'view' in row else serials[row['camera'].split('_')[1]]
            c = cal[cam]
            q = np.array([[float(row[f'corner_{i}_{a}_px']) for a in ('x','y')] for i in range(4)])
            u = cv2.undistortPoints(q.reshape(4,1,2),np.array(c['camera_matrix']),np.array(c['distortion_coefficients'])).reshape(4,2)
            observations.setdefault((cam,int(row['marker_id'])),[]).append((int(row['timestamp_ns']),u))
    if not observations:raise ValueError('No marker observations for timing analysis')
    origin = min(v[0][0] for v in observations.values())
    lags = np.arange(-250,251,1,dtype=float)/1000
    result = dict(source_calibration=str(calibration),method='Epipolar corner alignment; 1 ms lag grid from -250 to +250 ms; fit even 2-second blocks, validate odd blocks',
                  reference_camera=reference_camera, reference_lag_ns=int(reference_lag_ns),
                  timestamp_convention='corrected_reference_timestamp_ns = saved_pc_timestamp_ns - rs_minus_zed_lag_ns',hardware_synchronized=False,cameras={})
    for cam in ['d435','d455']:
        if cam == reference_camera:
            continue
        ca,cb = cal[reference_camera],cal[cam]
        T = np.array(cb['T_camera_from_reference'])@np.array(ca['T_reference_from_camera'])
        x,y,z = T[:3,3]
        E = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@T[:3,:3]
        curves, per_id = [], {}
        for mid in sorted(selected):
            if (cam,mid) not in observations or (reference_camera,mid) not in observations: continue
            za,ra = observations[reference_camera,mid],observations[cam,mid]
            tz=np.array([(r[0]-origin)/1e9 for r in za]);uz=np.array([r[1] for r in za])
            tc=np.array([(r[0]-origin)/1e9 for r in ra]);uc=np.array([r[1] for r in ra])
            lo=np.searchsorted(tc,tz-.25)-1;hi=np.searchsorted(tc,tz+.25)
            valid=(lo>=0)&(hi<len(tc));indices=np.flatnonzero(valid)
            valid[indices]=[np.max(np.diff(tc[l:h+1]))<.08 for l,h in zip(lo[indices],hi[indices])]
            tz,uz=tz[valid],uz[valid]
            if len(tz)<20: continue
            line=np.concatenate([uz,np.ones((len(uz),4,1))],axis=2)@E.T
            scale=cb['camera_matrix'][1][1]/np.linalg.norm(line[:,:,:2],axis=2)
            train=(tz//2).astype(int)%2==0
            if min(sum(train),sum(~train))<5:continue
            costs=[]
            for lag in lags:
                uv=np.stack([np.interp(tz+lag,tc,uc[:,j,a]) for j in range(4) for a in range(2)],axis=1).reshape(-1,4,2)
                e=np.abs(np.sum(line*np.concatenate([uv,np.ones((len(uv),4,1))],axis=2),axis=2))*scale
                costs.append([float(np.mean(np.minimum(e[mask],20))) for mask in (train,~train)])
            costs=np.array(costs)
            curves.append(costs)
            per_id[mid]=(costs,int(sum(train)),int(sum(~train)))
        if not curves:
            result['cameras'][cam]=dict(rs_minus_zed_lag_ns=0,validated=False,status='insufficient_shared_observations',
                limitation='Original PC timestamps retained; no reliable content-based time correction could be estimated.',per_marker={})
            continue
        # Weight by available support, capped at 500 observations per marker/block set.
        # A handful of fleeting forearm detections must not outweigh a long track.
        weights=np.array([[min(ntrain,500),min(nhold,500)] for _,ntrain,nhold in per_id.values()])
        aggregate=np.sum(np.array(curves)*weights[:,None,:],axis=0)/weights.sum(axis=0)
        best=int(np.argmin(aggregate[:,0])); heldout_best=int(np.argmin(aggregate[:,1]))
        entry=dict(rs_minus_zed_lag_ns=int(round(lags[best]*1e9)) + int(reference_lag_ns),
                   training_best_ms=float(lags[best]*1000) + reference_lag_ns / 1e6,
                   independent_holdout_best_ms=float(lags[heldout_best]*1000) + reference_lag_ns / 1e6,
                   reference_camera=reference_camera, reference_lag_ns=int(reference_lag_ns),
                   relative_training_best_ms=float(lags[best]*1000), per_marker={})
        for mid,(costs,ntrain,nhold) in per_id.items():
            entry['per_marker'][str(mid)]=dict(training_frames=ntrain,holdout_frames=nhold,
                training_original_px=float(costs[250,0]),training_corrected_px=float(costs[best,0]),
                holdout_original_px=float(costs[250,1]),holdout_corrected_px=float(costs[best,1]))
        entry['validated']=bool(abs(lags[best]-lags[heldout_best])<=.025 and aggregate[best,1]<aggregate[250,1] and abs(lags[best])<.25)
        entry['status']='content_alignment_validated' if entry['validated'] else 'content_alignment_not_validated'
        if reference_camera not in ('zed_left', 'zed_right'):
            entry['limitation'] = 'Indirect alignment through an independently validated RealSense reference; its timing uncertainty also applies. Per-marker original residuals use zero relative lag.'
        if not entry['validated']:
            entry['proposed_lag_ns']=entry['rs_minus_zed_lag_ns']
            entry['rs_minus_zed_lag_ns']=0
            entry['limitation']='Proposed correction failed holdout validation; original PC timestamps retained. Proposed residuals below are diagnostic only.'
        result['cameras'][cam]=entry
    (output/'timing_alignment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    estimate(*(Path(s).resolve() for s in sys.argv[1:4]))
