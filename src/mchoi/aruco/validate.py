"""Validate pose exports, known synthetic geometry, and every encoded RGB timestamp."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

from mchoi.aruco.poses import CAMERAS,Geometry,dataset,POSE_FIELDS
from mchoi.common import rotation_angle


def validate(output,video=False):
    metadata=json.loads((output/'pose_metadata.json').read_text())
    take=Path(metadata['source_recording']);calibration=Path(metadata['source_calibration'])
    saved,serials,times,adjusted,obs,ids,counts,groups,corrections=dataset(take,calibration,output)
    geometry=Geometry(saved)
    assert hashlib.sha256(calibration.read_bytes()).digest()==hashlib.sha256((output/'calibration_used.json').read_bytes()).digest()
    rows=list(csv.DictReader((output/'poses.csv').open()))
    assert len(rows)==len(groups)*len(ids)
    assert [(int(r['group_index']),int(r['marker_id'])) for r in rows]==[(g['group_index'],mid) for g in groups for mid in ids]
    seen=Counter((cam,fi) for g in groups for cam,fi in g['frames'].items())
    assert set(seen)=={(cam,i) for cam in CAMERAS for i in range(len(times[cam]))}
    assert all(n==1 for n in seen.values())
    per_camera=list(csv.DictReader((output/'per_camera_poses.csv').open()))
    assert [(r['camera'],int(r['frame_index']),int(r['marker_id'])) for r in per_camera]==[(c,i,mid) for c in CAMERAS for i in range(len(times[c])) for mid in ids]
    lookup={(int(r['group_index']),int(r['marker_id'])):r for r in rows}
    for row in per_camera:
        c,i=row['camera'],int(row['frame_index'])
        assert int(row['pc_timestamp_ns'])==times[c][i]
        assert int(row['corrected_image_timestamp_ns'])==adjusted[c][i]
        assert abs(float(row['image_minus_pose_time_ms'])-(adjusted[c][i]-int(row['timestamp_ns']))/1e6)<1e-8
        assert abs(float(row['image_minus_pose_time_ms']))<=20
        original=lookup[int(row['group_index']),int(row['marker_id'])]
        assert all(row.get(k)==original.get(k) for k in POSE_FIELDS+['pose_valid','pose_status','source_cameras'])
    largest_error=0.;valid=0;big_steps=[];exclusions=0
    for row in rows:
        group=groups[int(row['group_index'])];mid=int(row['marker_id']);assert int(row['timestamp_ns'])==group['timestamp_ns']
        if row['pose_valid']!='1':
            assert all(not row.get(k) for k in POSE_FIELDS)
            continue
        valid+=1
        R=np.array([[float(row[f'R{i}{j}']) for j in range(3)] for i in range(3)])
        x=np.array([float(row[k]) for k in ('rx_rad','ry_rad','rz_rad','tx_m','ty_m','tz_m')])
        assert np.max(np.abs(R.T@R-np.eye(3)))<1e-8 and abs(np.linalg.det(R)-1)<1e-8
        assert np.max(np.abs(cv2.Rodrigues(x[:3])[0]-R))<1e-8
        sources=row['source_cameras'].split('|');residuals=[]
        for cam in sources:
            fi=group['frames'][cam];q=obs[cam,fi,mid]
            K,D,T=geometry.cal[cam]
            camera_points=(geometry.objects@R.T+x[3:])@T[:3,:3].T+T[:3,3]
            assert camera_points[:,2].min()>0
            projected=cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),K,D)[0].reshape(4,2)
            errors=np.linalg.norm(projected-q,axis=1);residuals.extend(errors)
            assert abs(np.sqrt(np.mean(errors**2))-float(row[cam+'_reprojection_rms_px']))<1e-6
        rms=float(np.sqrt(np.mean(np.array(residuals)**2)))
        assert abs(rms-float(row['reprojection_rms_px']))<1e-6 and rms<=2.000001 and max(residuals)<=5.000001
        largest_error=max(largest_error,max(residuals))
        if row.get('normal_step_deg') and float(row['normal_step_deg'])>45:big_steps.append({k:row[k] for k in ['group_index','elapsed_seconds','marker_id','pose_status','normal_step_deg','source_cameras']})
        exclusions+=bool(row['excluded_cameras'])
    # Exact synthetic observations from independently perturbed reference poses,
    # with one camera removed and one corrupted, test scale, transforms and outlier handling.
    selected=[r for r in rows if r['pose_status']=='multiview' and len(r['source_cameras'].split('|'))>=3][::250]
    rng=np.random.default_rng(314);synthetic=[]
    for row in selected[:12]:
        x=np.array([float(row[k]) for k in ('rx_rad','ry_rad','rz_rad','tx_m','ty_m','tz_m')]);x[3:]+=rng.normal(0,.003,3)
        cameras=row['source_cameras'].split('|')
        exact={c:geometry.project(x,c) for c in cameras}
        pose,_=geometry.solve(exact,None,0)
        assert pose is not None and np.linalg.norm(pose['x'][3:]-x[3:])<1e-6 and rotation_angle(pose['R'],cv2.Rodrigues(x[:3])[0])<.001
        noisy={c:q+rng.normal(0,.15,q.shape) for c,q in exact.items()}
        pose,_=geometry.solve(noisy,None,0)
        assert pose is not None and np.linalg.norm(pose['x'][3:]-x[3:])<.002 and rotation_angle(pose['R'],cv2.Rodrigues(x[:3])[0])<5
        corrupted={c:q.copy() for c,q in exact.items()};bad=cameras[-1];corrupted[bad]+=[80,-65]
        pose,_=geometry.solve(corrupted,None,0)
        assert pose is not None and bad not in pose['sources'] and np.linalg.norm(pose['x'][3:]-x[3:])<1e-6
        missing=dict(exact);del missing[cameras[-1]]
        pose,_=geometry.solve(missing,None,0)
        assert pose is not None and np.linalg.norm(pose['x'][3:]-x[3:])<1e-6
        synthetic.append(dict(group_index=int(row['group_index']),cameras=cameras,exact_and_noisy_and_corrupted_and_missing='passed'))
    report=dict(pose_rows=len(rows),per_camera_pose_rows=len(per_camera),valid_poses=valid,all_native_frames_covered_once=True,source_calibration_copied_verbatim=True,
                transforms_and_reprojection_verified=True,max_accepted_corner_residual_px=largest_error,
                poses_with_excluded_views=exclusions,normal_steps_over_45_deg=big_steps,synthetic_cases=synthetic)
    (output/'pose_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('synthetic_cases','normal_steps_over_45_deg')},indent=2),flush=True)
    print('Large normal steps:',json.dumps(big_steps[:30]),'total',len(big_steps),flush=True)
    if video:
        video_report={}
        dimensions=tuple(metadata.get('video_dimensions',[1920,1380]))
        for cam in (*CAMERAS,'all_cameras'):
            expected=times['zed_left' if cam=='all_cameras' else cam]
            path=output/(cam+'_annotated.mp4');count=0
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    assert count<len(expected)
                    assert (frame.width,frame.height)==dimensions
                    assert round(frame.pts*frame.time_base*1_000_000)==(expected[count]-expected[0])//1000
                    count+=1
            assert count==len(expected)
            video_report[cam]=dict(frames=count,all_native_timestamps_verified=True,size_bytes=path.stat().st_size,dimensions=list(dimensions))
            print('Verified video',cam,count,flush=True)
        (output/'video_validation.json').write_text(json.dumps(video_report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--video',action='store_true')
    a=p.parse_args();validate(a.output.resolve(),a.video)
