"""Choose measured pose branches over time without smoothing or inventing poses.

Dynamic programming favors continuity across camera handoffs. Multi-view fits are
fixed anchors. A sequence with no multi-view anchor has unresolved absolute planar
orientation, even if its selected branch is smooth; this is explicit in the CSV.
"""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from mchoi.aruco.poses import CAMERAS,POSE_FIELDS,write_csv,Geometry,dataset
from mchoi.common import rotation_angle


def track(output):
    metadata=json.loads((output/'pose_metadata.json').read_text())
    saved,serials,times,adjusted,obs,ids,counts,groups,corrections=dataset(Path(metadata['source_recording']),Path(metadata['source_calibration']),output)
    geometry=Geometry(saved)
    raw=output/'independent_estimates.csv'
    if not raw.exists():shutil.copy2(output/'poses.csv',raw)
    rows=list(csv.DictReader(raw.open()));candidates=defaultdict(list);byid=defaultdict(list)
    for c in csv.DictReader((output/'monocular_candidates.csv').open()):candidates[int(c['group_index']),int(c['marker_id'])].append(c)
    for i,row in enumerate(rows):byid[int(row['marker_id'])].append(i)
    def state(row,multi=False):
        R=np.array([[float(row[f'R{i}{j}']) for j in range(3)] for i in range(3)])
        t=np.array([float(row[k]) for k in ('tx_m','ty_m','tz_m')])
        return dict(row=row,R=R,t=t,multi=multi,cost=float(row['reprojection_rms_px'])**2)
    for mid,indices in byid.items():
        segments=[];segment=[];last=None
        for i in indices:
            row=rows[i];ts=int(row['timestamp_ns']);key=(int(row['group_index']),mid)
            states=[state(row,True)] if row['pose_status']=='multiview' else [state(c) for c in candidates[key]]
            if not states:continue
            if last is not None and ts-last>250_000_000:
                segments.append(segment);segment=[]
            segment.append((i,states));last=ts
        if segment:segments.append(segment)
        for segment in segments:
            anchored=any(s['multi'] for _,states in segment for s in states)
            costs=np.array([s['cost'] for s in segment[0][1]]);backs=[]
            for (pi,prev),(i,current) in zip(segment,segment[1:]):
                dt=max((int(rows[i]['timestamp_ns'])-int(rows[pi]['timestamp_ns']))/1e9,.001)
                angle_scale=max(5,100*dt);position_scale=max(.005,.5*dt)
                transition=np.array([[(rotation_angle(a['R'],b['R'])/angle_scale)**2+(np.linalg.norm(a['t']-b['t'])/position_scale)**2 for b in current] for a in prev])
                total=costs[:,None]+transition
                back=np.argmin(total,axis=0);backs.append(back)
                costs=total[back,np.arange(len(current))]+np.array([s['cost'] for s in current])
            path=[int(np.argmin(costs))]
            for back in reversed(backs):path.append(int(back[path[-1]]))
            path.reverse()
            for (i,states),choice in zip(segment,path):
                selected=states[choice];row=rows[i];c=selected['row']
                row['orientation_support']='multiview_geometry' if selected['multi'] else ('temporal_sequence_with_multiview_anchor' if anchored else 'monocular_sequence_no_multiview_anchor')
                row['orientation_is_multiview']=int(selected['multi'])
                row['sequence_has_multiview_anchor']=int(anchored)
                if not selected['multi']:
                    row['pose_status']='single_view_tracked' if anchored else 'single_view_tracked_unanchored'
                    row['pose_valid']='1';row['source_cameras']=c['camera']
                    row['excluded_cameras']='|'.join(cam for cam in row['detected_cameras'].split('|') if cam!=c['camera'])
                    for key in POSE_FIELDS+[f'R{a}{b}' for a in range(3) for b in range(3)]:row[key]=c[key]
                    row['reprojection_rms_px']=c['reprojection_rms_px']
                    x=np.array([float(c[k]) for k in ('rx_rad','ry_rad','rz_rad','tx_m','ty_m','tz_m')])
                    q=obs[c['camera'],int(c['frame_index']),mid]
                    row['max_corner_error_px']=float(np.linalg.norm(geometry.project(x,c['camera'])-q,axis=1).max())
                    for cam in CAMERAS:row[cam+'_reprojection_rms_px']=c['reprojection_rms_px'] if cam==c['camera'] else ''
        previous=None
        for i in indices:
            row=rows[i]
            for key in ['rotation_step_deg','normal_step_deg','position_step_mm','source_handoff']:row[key]=''
            if row['pose_valid']!='1':continue
            current=state(row);ts=int(row['timestamp_ns'])
            if previous and 0<ts-previous[0]<=100_000_000:
                old=previous[1]
                normal=float(np.degrees(np.arccos(np.clip(old['R'][:,2]@current['R'][:,2],-1,1))))
                if normal>45:
                    row['pose_valid']='0';row['pose_status']='abrupt_orientation_rejected';row['rejected_normal_step_deg']=normal
                    for key in POSE_FIELDS+[f'R{a}{b}' for a in range(3) for b in range(3)]:row[key]=''
                    continue
                row['normal_step_deg']=normal;row['rotation_step_deg']=rotation_angle(old['R'],current['R'])
                row['position_step_mm']=float(np.linalg.norm(old['t']-current['t'])*1000)
                row['source_handoff']=int(old['row']['source_cameras']!=row['source_cameras'])
            previous=(ts,current)
    write_csv(output/'poses.csv',rows)
    per_camera=[]
    for row in rows:
        group=groups[int(row['group_index'])]
        for cam,fi in group['frames'].items():
            per_camera.append(dict(camera=cam,frame_index=fi,pc_timestamp_ns=times[cam][fi],
                corrected_image_timestamp_ns=adjusted[cam][fi],image_minus_pose_time_ms=(adjusted[cam][fi]-int(row['timestamp_ns']))/1e6,
                detected_in_this_camera=row[cam+'_detected'],used_for_this_pose=int(cam in row['source_cameras'].split('|') and row['pose_valid']=='1'),**row))
    per_camera.sort(key=lambda row:(CAMERAS.index(row['camera']),int(row['frame_index']),int(row['marker_id'])))
    write_csv(output/'per_camera_poses.csv',per_camera)
    meta=json.loads((output/'pose_metadata.json').read_text())
    meta['temporal_branch_rule']='Offline dynamic-programming selection among measured IPPE/LM candidate poses. Multi-view fits are fixed anchors. Rotation transition scale=max(5deg,100deg/s*dt), position scale=max(5mm,0.5m/s*dt). Unary cost=RMS_px squared. Chains break after 250ms without any candidate. No smoothing, interpolation or pose extrapolation. Reject normal changes >45deg across <=100ms as abrupt/uncertain.'
    meta['orientation_warning']='Single-view tracked poses use a continuity assumption. Unanchored tracks (including marker 1) cannot resolve absolute planar orientation from geometry; both raw candidate branches are retained. A smooth trajectory is not proof of accurate orientation.'
    for mid in meta['marker_ids']:
        rr=[r for r in rows if int(r['marker_id'])==mid];valid=[r for r in rr if r['pose_valid']=='1'];steps=[float(r['normal_step_deg']) for r in valid if r.get('normal_step_deg') not in (None,'')]
        meta['per_marker'][str(mid)]=dict(status_counts=dict(Counter(r['pose_status'] for r in rr)),valid=len(valid),
            median_rms_px=float(np.median([float(r['reprojection_rms_px']) for r in valid])) if valid else None,
            normal_steps_over_45_deg=sum(s>45 for s in steps),max_normal_step_deg=max(steps) if steps else None)
    (output/'pose_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta['per_marker'],indent=2),flush=True)


if __name__=='__main__':track(Path(sys.argv[1]).resolve())
