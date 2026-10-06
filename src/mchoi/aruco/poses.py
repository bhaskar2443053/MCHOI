"""Recover independent marker poses across all four calibrated RGB views.

Uses native corner measurements, calibrated optical transforms, recorded PC times,
and an explicitly saved content-based RGB timing correction. Never fills missing
observations. Monocular branch selection through continuity is explicitly labeled.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import time

import av
import cv2
import numpy as np
from scipy.optimize import least_squares

from mchoi.recorder.devices import load_zed
from mchoi.common import square_points, roll_pitch_yaw
from mchoi.common import rotation_angle
from mchoi.io.realsense import ColorReader

CAMERAS = ('zed_left','zed_right','d435','d455')
POSE_FIELDS = ['tx_m','ty_m','tz_m','rx_rad','ry_rad','rz_rad','roll_deg','pitch_deg','yaw_deg']


def panel_height(marker_count):
    return max(300,140+40*marker_count)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def dataset(take, calibration_path, output):
    saved=json.loads(calibration_path.read_text())
    if saved['status']!='calibrated' or saved['reference_camera']!='zed_left':raise ValueError('Invalid calibration')
    serials={key:saved['cameras'][key]['device']['serial'] for key in ('d435','d455')}
    cam_to_source={**{c:('zed',c.split('_')[1]) for c in CAMERAS[:2]},**{c:('realsense_'+s,'color') for c,s in serials.items()}}
    source_to_cam={v:k for k,v in cam_to_source.items()}
    times={c:[] for c in CAMERAS}
    for row in csv.DictReader((take/'frame_timestamps.csv').open()):
        cam=source_to_cam.get((row['camera'],row['stream']))
        if cam is not None:
            if int(row['frame_index'])!=len(times[cam]):raise ValueError('Non-contiguous native frames')
            times[cam].append(int(row['pc_timestamp_ns']))
    corrections={c:0 for c in CAMERAS}
    timing=json.loads((output/'timing_alignment.json').read_text())
    if Path(timing['source_calibration']).resolve()!=calibration_path.resolve():raise ValueError('Timing calibration mismatch')
    corrections.update({c:v['rs_minus_zed_lag_ns'] for c,v in timing['cameras'].items()})
    adjusted={c:[t-corrections[c] for t in ts] for c,ts in times.items()}
    obs={}; counts=Counter(); coverage=set(); duplicates=set()
    paths=[take/'aruco_annotated/detections.csv']+[take/'aruco_annotated'/f'realsense_{s}_detections.csv' for s in serials.values()]
    for path in paths:
        for row in csv.DictReader(path.open()):
            cam='zed_'+row['view'] if 'view' in row else source_to_cam[(row['camera'],'color')]
            frame=int(row['frame_index']);coverage.add((cam,frame))
            if int(row['timestamp_ns'])!=times[cam][frame]:raise ValueError('Detection timestamp mismatch')
            if row['detected']!='1':continue
            mid=int(row['marker_id']);key=(cam,frame,mid)
            if key in obs:duplicates.add(key)
            obs[key]=np.array([[float(row[f'corner_{i}_{a}_px']) for a in ('x','y')] for i in range(4)])
            counts[mid]+=1
    for key in duplicates:del obs[key]
    if coverage!={(c,i) for c in CAMERAS for i in range(len(times[c]))}:raise ValueError('Missing detection frames')
    # Repeated IDs are retained; all rare decoder candidates remain in original detection CSVs.
    selection=output/'marker_selection.json'
    ids=sorted(json.loads(selection.read_text())['marker_ids']) if selection.exists() else sorted(mid for mid,count in counts.items() if count>=10)
    if not ids or any(mid not in counts for mid in ids):raise ValueError('Marker selection contains no observations')
    used={c:set() for c in CAMERAS}; groups=[]
    for i,timestamp in enumerate(adjusted['zed_left']):
        frames={'zed_left':i,'zed_right':i};used['zed_left'].add(i);used['zed_right'].add(i)
        for cam in CAMERAS[2:]:
            j=bisect_left(adjusted[cam],timestamp)
            candidates=[k for k in (j-1,j) if 0<=k<len(adjusted[cam]) and k not in used[cam]]
            if candidates:
                k=min(candidates,key=lambda k:abs(adjusted[cam][k]-timestamp))
                if abs(adjusted[cam][k]-timestamp)<=20_000_000:frames[cam]=k;used[cam].add(k)
        groups.append(dict(timestamp_ns=timestamp,frames=frames))
    # Preserve every extra RGB frame. These events are not included in the ZED-anchored overview.
    for cam in CAMERAS[2:]:
        for i,timestamp in enumerate(adjusted[cam]):
            if i not in used[cam]:groups.append(dict(timestamp_ns=timestamp,frames={cam:i}));used[cam].add(i)
    groups.sort(key=lambda g:g['timestamp_ns'])
    for i,g in enumerate(groups):g['group_index']=i
    return saved,serials,times,adjusted,obs,ids,counts,groups,corrections


class Geometry:
    def __init__(self,saved,side_mm=24):
        self.objects=square_points(side_mm/1000)
        self.cal={}
        for cam in CAMERAS:
            c=saved['cameras'][cam]
            if c['image_size']!=[1920,1080]:raise ValueError('Calibration size mismatch')
            self.cal[cam]=(np.array(c['camera_matrix']),np.array(c['distortion_coefficients']),np.array(c['T_camera_from_reference']))

    def project(self,x,cam):
        K,D,T=self.cal[cam]
        R=T[:3,:3]@cv2.Rodrigues(x[:3])[0]
        t=T[:3,:3]@x[3:]+T[:3,3]
        return cv2.projectPoints(self.objects,cv2.Rodrigues(R)[0],t,K,D)[0].reshape(4,2)

    def pose(self,x,observations):
        if not np.isfinite(x).all():return None
        R=cv2.Rodrigues(x[:3])[0];world=self.objects@R.T+x[3:]
        # A printed marker is visible only from its front half-space.
        for cam in observations:
            T=self.cal[cam][2];points=world@T[:3,:3].T+T[:3,3]
            normal=T[:3,:3]@R[:,2];center=points.mean(axis=0)
            if points[:,2].min()<=0 or np.dot(normal,center)>=0:return None
        errors={cam:np.linalg.norm(self.project(x,cam)-q,axis=1) for cam,q in observations.items()}
        return dict(x=x,R=R,rms=float(np.sqrt(np.mean(np.concatenate(list(errors.values()))**2))),
                    max_error=float(max(e.max() for e in errors.values())),errors={c:float(np.sqrt(np.mean(e**2))) for c,e in errors.items()},sources=list(observations))

    def singles(self,cam,q):
        if abs(cv2.contourArea(q.astype(np.float32)))<16:return []
        K,D,T=self.cal[cam];inv=np.linalg.inv(T)
        _,rv,tv,_=cv2.solvePnPGeneric(self.objects,q,K,D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        poses=[]
        for r,t in zip(rv,tv):
            r,t=cv2.solvePnPRefineLM(self.objects,q,K,D,r.copy(),t.copy())
            R=inv[:3,:3]@cv2.Rodrigues(r)[0];translation=inv[:3,:3]@t.ravel()+inv[:3,3]
            x=np.r_[cv2.Rodrigues(R)[0].ravel(),translation]
            pose=self.pose(x,{cam:q})
            if pose and pose['rms']<=2 and pose['max_error']<=5 and not any(rotation_angle(p['R'],R)<1 for p in poses):poses.append(pose)
        return sorted(poses,key=lambda p:p['rms'])

    def epipolar(self,a,b,qa,qb):
        ka,da,ta=self.cal[a];kb,db,tb=self.cal[b];T=tb@np.linalg.inv(ta)
        x,y,z=T[:3,3];E=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@T[:3,:3]
        ua,ub=[cv2.undistortPoints(q.reshape(4,1,2),K,D).reshape(4,2) for q,K,D in [(qa,ka,da),(qb,kb,db)]]
        ua,ub=np.c_[ua,np.ones(4)],np.c_[ub,np.ones(4)]
        ea=np.abs(np.sum((ub@E)*ua,axis=1))/np.linalg.norm((ub@E)[:,:2],axis=1)*ka[1,1]
        eb=np.abs(np.sum((ua@E.T)*ub,axis=1))/np.linalg.norm((ua@E.T)[:,:2],axis=1)*kb[1,1]
        return float(np.sqrt(np.mean(np.r_[ea,eb]**2)))

    def fit(self,observations,singles):
        starts=[]
        for cam in observations:
            for p in singles[cam]:
                if not any(rotation_angle(cv2.Rodrigues(x[:3])[0],p['R'])<5 and np.linalg.norm(x[3:]-p['x'][3:])<.01 for x in starts):starts.append(p['x'])
        # Multi-view triangulation supplies a branch-independent geometric initialization.
        a,b=max(combinations(observations,2),key=lambda pair:np.linalg.norm(np.linalg.inv(self.cal[pair[0]][2])[:3,3]-np.linalg.inv(self.cal[pair[1]][2])[:3,3]))
        normalized=[cv2.undistortPoints(observations[c].reshape(4,1,2),*self.cal[c][:2]).reshape(4,2) for c in (a,b)]
        h=cv2.triangulatePoints(self.cal[a][2][:3],self.cal[b][2][:3],normalized[0].T,normalized[1].T)
        if np.min(np.abs(h[3]))>1e-12:
            world=(h[:3]/h[3]).T;t=world.mean(axis=0)
            u,_,vt=np.linalg.svd(self.objects.T@(world-t));R=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
            starts.append(np.r_[cv2.Rodrigues(R)[0].ravel(),t])
        def residual(x):return np.concatenate([(self.project(x,c)-q).ravel() for c,q in observations.items()])
        candidates=[]
        for start in starts:
            fit=least_squares(residual,start,method='lm',max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            if not fit.success:continue
            p=self.pose(fit.x,observations)
            if p and not any(rotation_angle(old['R'],p['R'])<1 for old in candidates):candidates.append(p)
        candidates.sort(key=lambda p:p['rms'])
        if not candidates:return None
        best=candidates[0]
        best['ambiguous']=any(rotation_angle(best['R'],p['R'])>5 and p['rms']-best['rms']<.25 for p in candidates[1:])
        if best['rms']>2 or best['max_error']>5 or best['ambiguous']:return None
        return best

    def solve(self,observations,previous,timestamp):
        singles={c:self.singles(c,q) for c,q in observations.items()}
        pair_errors={frozenset((a,b)):self.epipolar(a,b,observations[a],observations[b]) for a,b in combinations(observations,2)}
        for size in range(len(observations),1,-1):
            fits=[]
            for subset in combinations(observations,size):
                if any(pair_errors[frozenset(pair)]>3 for pair in combinations(subset,2)):continue
                fit=self.fit({c:observations[c] for c in subset},singles)
                if fit:fit['status']='multiview';fits.append(fit)
            if fits:
                # Prefer the most precise agreeing subset; disagreement is retained in the source fields.
                best=min(fits,key=lambda p:p['rms'])
                return best,singles
        reliable=[];temporal=[]
        for cam,candidates in singles.items():
            if not candidates:continue
            best=candidates[0]
            if len(candidates)==1 or candidates[1]['rms']-best['rms']>=.25:
                best['status']='single_view';reliable.append(best)
            elif previous and 0<timestamp-previous[0]<=250_000_000:
                # Choose a measured branch, never smooth its values or extrapolate through absence.
                matches=sorted((rotation_angle(previous[1]['R'],p['R']),j) for j,p in enumerate(candidates))
                angle,j=matches[0]
                if angle<=25 and matches[1][0]-angle>=10 and np.linalg.norm(candidates[j]['x'][3:]-previous[1]['x'][3:])<=.05:
                    p=candidates[j];p['status']='single_view_temporal';temporal.append(p)
        pool=reliable or temporal
        if pool:
            # With disagreement, keep one measured camera pose rather than average incompatible views.
            if previous and timestamp-previous[0]<=250_000_000:
                best=min(pool,key=lambda p:rotation_angle(previous[1]['R'],p['R'])/30+np.linalg.norm(p['x'][3:]-previous[1]['x'][3:])/.03+p['rms'])
            else:best=min(pool,key=lambda p:p['rms'])
            return best,singles
        return None,singles


def pose_values(p):
    values=np.r_[p['x'][3:],p['x'][:3],roll_pitch_yaw(p['R'])]
    row={k:float(v) for k,v in zip(POSE_FIELDS,values)}
    row.update({f'R{i}{j}':float(p['R'][i,j]) for i in range(3) for j in range(3)})
    return row


def prepare(take,calibration_path,output):
    if (output/'poses.csv').exists():raise FileExistsError(output/'poses.csv')
    saved,serials,times,adjusted,obs,ids,counts,groups,corrections=dataset(take,calibration_path,output)
    geometry=Geometry(saved);rows=[];candidate_rows=[];previous={};alignment=[];start=time.monotonic()
    origin=times['zed_left'][0]
    for group in groups:
        gi=group['group_index'];ts=group['timestamp_ns'];frames=group['frames']
        align=dict(group_index=gi,timestamp_ns=ts,elapsed_seconds=(ts-origin)/1e9)
        for cam in CAMERAS:
            fi=frames.get(cam)
            align.update({cam+'_frame_index':fi,cam+'_pc_timestamp_ns':times[cam][fi] if fi is not None else None,
                          cam+'_corrected_offset_ms':(adjusted[cam][fi]-ts)/1e6 if fi is not None else None})
        alignment.append(align)
        for mid in ids:
            observations={c:obs[c,fi,mid] for c,fi in frames.items() if (c,fi,mid) in obs}
            p,singles=geometry.solve(observations,previous.get(mid),ts) if observations else (None,{})
            row=dict(group_index=gi,timestamp_ns=ts,elapsed_seconds=(ts-origin)/1e9,marker_id=mid,
                     pose_valid=int(p is not None),pose_status=p['status'] if p else ('orientation_ambiguous_or_bad_fit' if observations else 'not_detected'),
                     source_cameras='|'.join(p['sources']) if p else '',detected_cameras='|'.join(observations),
                     excluded_cameras='|'.join(c for c in observations if not p or c not in p['sources']),
                     reference_frame='zed_left_optical',reprojection_rms_px=p['rms'] if p else None,max_corner_error_px=p['max_error'] if p else None)
            for cam in CAMERAS:
                row[cam+'_frame_index']=frames.get(cam);row[cam+'_detected']=int(cam in observations)
                row[cam+'_reprojection_rms_px']=p['errors'].get(cam) if p else None
            if p:
                row.update(pose_values(p))
                old=previous.get(mid)
                if old and 0<ts-old[0]<=100_000_000:
                    row['rotation_step_deg']=rotation_angle(old[1]['R'],p['R'])
                    row['normal_step_deg']=float(np.degrees(np.arccos(np.clip(old[1]['R'][:,2]@p['R'][:,2],-1,1))))
                    row['position_step_mm']=float(np.linalg.norm(old[1]['x'][3:]-p['x'][3:])*1000)
                    row['source_handoff']=int(old[1]['sources']!=p['sources'])
                previous[mid]=(ts,p)
            rows.append(row)
            for cam,candidates in singles.items():
                for branch,pc in enumerate(candidates):
                    candidate_rows.append(dict(group_index=gi,camera=cam,frame_index=frames[cam],pc_timestamp_ns=times[cam][frames[cam]],
                        corrected_timestamp_ns=adjusted[cam][frames[cam]],marker_id=mid,branch=branch,reprojection_rms_px=pc['rms'],**pose_values(pc)))
        if (gi+1)%400==0:print(f'Multicamera pose groups: {gi+1}/{len(groups)} | {time.monotonic()-start:.1f}s',flush=True)
    write_csv(output/'poses.csv',rows);write_csv(output/'monocular_candidates.csv',candidate_rows);write_csv(output/'aligned_frames.csv',alignment)
    # The supplied calibration is copied verbatim for reproducibility; no intrinsic/extrinsic changes.
    shutil.copy2(calibration_path,output/'calibration_used.json')
    metadata=dict(source_recording=str(take),source_calibration=str(calibration_path),camera_serials=serials,
        dictionary='DICT_4X4_50',marker_black_square_side_m=.024,marker_ids=ids,all_decoder_id_counts=dict(counts),
        excluded_rare_decoder_ids={mid:n for mid,n in counts.items() if mid not in ids},
        frame_counts={c:len(t) for c,t in times.items()},groups=len(groups),csv_rows=len(rows),video_dimensions=[1920,1080+panel_height(len(ids))],
        video_encoder=os.environ.get('ARUCO_VIDEO_ENCODER','libx264'),
        time_correction_subtracted_ns=corrections,alignment_tolerance_ms=20,hardware_synchronized=False,
        reference_frame='ZED left native unrectified optical; X right, Y down, Z forward',
        marker_frame='Black-square center; X decoded right, Y decoded up, Z out of printed face',
        transform='p_reference = R @ p_marker + t; column vectors; translation metres',
        rotation='Rodrigues vector radians; Euler degrees R=Rz(yaw)@Ry(pitch)@Rx(roll); matrix columns are marker axes in reference frame',
        method='Joint rigid-square reprojection fit across largest agreeing subset; triangulation plus monocular IPPE branches as starts; otherwise labeled monocular pose',
        acceptance='RMS <=2px, max corner residual <=5px, positive depth and front-facing; multi-view symmetric epipolar RMS <=3px; reject distinct rotations within 0.25px RMS',
        temporal_branch_rule='For ambiguous monocular measurements only: prior <=250ms old, rotation <=25deg, alternative at least 10deg farther, translation <=50mm. Output is the measured branch, without smoothing.',
        no_interpolation=True,no_pose_without_detection=True,calibration_refined=False,
        limitations='A single flat marker has ambiguous orientation in some views. Continuity-selected monocular poses are less reliable than multi-view estimates. Timing is estimated from image content, not synchronized exposure. Calibration consistency and low residual do not establish absolute metrology accuracy.',
        per_marker={})
    for mid in ids:
        rr=[r for r in rows if r['marker_id']==mid];valid=[r for r in rr if r['pose_valid']]
        steps=[r['normal_step_deg'] for r in valid if 'normal_step_deg' in r]
        metadata['per_marker'][mid]=dict(status_counts=dict(Counter(r['pose_status'] for r in rr)),valid=len(valid),
            median_rms_px=float(np.median([r['reprojection_rms_px'] for r in valid])) if valid else None,
            normal_steps_over_45_deg=sum(s>45 for s in steps),max_normal_step_deg=max(steps) if steps else None)
    (output/'pose_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata['per_marker'],indent=2),flush=True)


def text(image,string,xy,scale=.7,color=(230,235,240),thickness=1):
    cv2.putText(image,string,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+3,cv2.LINE_AA)
    cv2.putText(image,string,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def axes(image,p,cam,geometry):
    K,D,T=geometry.cal[cam];R=T[:3,:3]@cv2.Rodrigues(p[:3])[0];t=T[:3,:3]@p[3:]+T[:3,3]
    points=np.array([[0,0,0],[.016,0,0],[0,.016,0],[0,0,.016]])
    proj=cv2.projectPoints(points,cv2.Rodrigues(R)[0],t,K,D)[0].reshape(4,2)
    if np.isfinite(proj).all() and np.max(np.abs(proj))<100000:
        for i,color in enumerate([(0,0,255),(0,255,0),(255,100,20)],1):
            a,b=tuple(np.rint(proj[0]).astype(int)),tuple(np.rint(proj[i]).astype(int))
            ok,a,b=cv2.clipLine((0,0,image.shape[1],image.shape[0]),a,b)
            if ok:cv2.line(image,a,b,color,3,cv2.LINE_AA)


def overlay(image,cam,fi,ts,origin,rows,obs,geometry,serials):
    image=image.copy()
    for row in rows:
        mid=int(row['marker_id']);q=obs.get((cam,fi,mid))
        if q is not None:
            included=cam in row['source_cameras'].split('|')
            color=(60,240,90) if included else (30,180,255)
            cv2.polylines(image,[np.rint(q).astype(np.int32)],True,color,2,cv2.LINE_AA)
            text(image,f'ID {mid}',tuple(np.rint(q[0]+[0,-10]).astype(int)),.65,color,2)
            if row['pose_valid']=='1' and included:
                p=np.array([float(row[k]) for k in ('rx_rad','ry_rad','rz_rad','tx_m','ty_m','tz_m')]);axes(image,p,cam,geometry)
    title=cam.upper() if cam.startswith('zed') else 'RealSense '+serials[cam]+' ('+cam+')'
    cv2.rectangle(image,(0,0),(1920,76),(25,22,20),-1)
    text(image,title,(22,30),.85)
    text(image,f'Native frame {fi} | recorded time {((ts-origin)/1e9):.3f}s | 4x4_50 | 24 mm',(22,62),.65)
    panel=np.full((panel_height(len(rows)),1920,3),(26,23,20),np.uint8)
    text(panel,'POSES IN ZED LEFT FRAME | X/Y/Z: mm | roll/pitch/yaw: deg | axes: X red, Y green, Z blue',(20,30),.70)
    text(panel,'ID         X           Y           Z           Roll      Pitch       Yaw       Fit px    Method / contributing cameras',(20,68),.68)
    for i,row in enumerate(rows):
        y=106+i*40;mid=int(row['marker_id'])
        if row['pose_valid']=='1':
            values=[float(row[k])*1000 for k in ('tx_m','ty_m','tz_m')]+[float(row[k]) for k in ('roll_deg','pitch_deg','yaw_deg')]
            for s,x in [(str(mid),25)]+[(f'{v:+.1f}',x) for v,x in zip(values,[140,285,430,595,740,890])]+[(f'{float(row["reprojection_rms_px"]):.2f}',1050)]:text(panel,s,(x,y),.75,(110,235,150))
            method={'multiview':'MULTI','single_view':'MONO','single_view_temporal':'MONO continuity',
                    'single_view_tracked':'MONO tracked','single_view_tracked_unanchored':'MONO uncertain'}[row['pose_status']]
            text(panel,method+' / '+row['source_cameras'],(1170,y),.60,(110,235,150) if method=='MULTI' else (70,200,255))
        else:text(panel,f'{mid}       '+('Not detected' if row['pose_status']=='not_detected' else 'Orientation ambiguous / fit rejected'),(25,y),.70,(70,200,255))
    offset=(ts-int(rows[0]['timestamp_ns']))/1e6
    text(panel,f'Group {rows[0]["group_index"]} | group time {float(rows[0]["elapsed_seconds"]):.3f}s | native PC time minus group: {offset:+.1f} ms',(20,panel.shape[0]-14),.64)
    return cv2.vconcat([image,panel])


class VideoWriter:
    def __init__(self,path,width,height):
        if path.exists():raise FileExistsError(path)
        self.container=av.open(str(path),'w',options={'movflags':'+faststart'})
        encoder=os.environ.get('ARUCO_VIDEO_ENCODER','libx264')
        self.stream=self.container.add_stream(encoder,rate=30)
        self.stream.width,self.stream.height,self.stream.pix_fmt=width,height,'yuv420p'
        self.clock=Fraction(1,1_000_000)
        self.stream.time_base=self.stream.codec_context.time_base=self.clock
        self.stream.codec_context.thread_count=3
        self.stream.options=({'preset':'p4','tune':'hq','rc':'vbr','cq':'18','b':'0'} if encoder=='h264_nvenc' else {'crf':'19','preset':'veryfast'})
    def add(self,image,timestamp):
        frame=av.VideoFrame.from_ndarray(image,format='bgr24');frame.pts=timestamp//1000;frame.time_base=self.clock
        for packet in self.stream.encode(frame):self.container.mux(packet)
    def close(self):
        for packet in self.stream.encode():self.container.mux(packet)
        self.container.close()


def render(take,calibration_path,output):
    saved,serials,times,adjusted,obs,ids,counts,groups,corrections=dataset(take,calibration_path,output)
    geometry=Geometry(saved);bygroup=defaultdict(list);byframe={};origin=times['zed_left'][0]
    for row in csv.DictReader((output/'poses.csv').open()):bygroup[int(row['group_index'])].append(row)
    for group in groups:
        for cam,fi in group['frames'].items():byframe[cam,fi]=bygroup[group['group_index']]
    # Full native-frame per-camera exports; montage is made in a second pass over compact H264 files.
    sl=load_zed();camera=sl.Camera();init=sl.InitParameters();init.set_from_svo_file(str(take/'zed.svo2'))
    init.svo_real_time_mode=False;init.depth_mode=sl.DEPTH_MODE.NONE;init.sdk_verbose=0
    if camera.open(init)>sl.ERROR_CODE.SUCCESS:raise RuntimeError('Cannot open SVO')
    height=1080+panel_height(len(ids))
    writers={c:VideoWriter(output/(c+'_annotated.mp4'),1920,height) for c in CAMERAS[:2]};mat=sl.Mat();fi=0;start=time.monotonic()
    try:
        while True:
            status=camera.grab()
            if status==sl.ERROR_CODE.END_OF_SVOFILE_REACHED:break
            if status>sl.ERROR_CODE.SUCCESS:raise RuntimeError(status)
            ts=int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if fi>=len(times['zed_left']) or ts!=times['zed_left'][fi]:raise ValueError('SVO timestamp mismatch')
            for cam,view in [('zed_left',sl.VIEW.LEFT_UNRECTIFIED),('zed_right',sl.VIEW.RIGHT_UNRECTIFIED)]:
                if camera.retrieve_image(mat,view)>sl.ERROR_CODE.SUCCESS:raise RuntimeError('SVO image failed')
                im=cv2.cvtColor(mat.get_data(),cv2.COLOR_BGRA2BGR)
                im=overlay(im,cam,fi,ts,origin,byframe[cam,fi],obs,geometry,serials)
                writers[cam].add(im,ts-times[cam][0])
                if fi in (0,770,1540,len(times[cam])-1):cv2.imwrite(str(output/f'{cam}_preview_{fi:04d}.jpg'),im)
            fi+=1
            if fi%400==0:print(f'ZED annotated videos: {fi}/{len(times["zed_left"])} | {time.monotonic()-start:.1f}s',flush=True)
        if fi!=len(times['zed_left']):raise ValueError('Incomplete ZED video')
    finally:
        camera.close()
        for writer in writers.values():writer.close()
    for cam in CAMERAS[2:]:
        reader=ColorReader(take/('realsense_'+serials[cam]+'.db3'));writer=VideoWriter(output/(cam+'_annotated.mp4'),1920,height)
        try:
            if reader.timestamps!=times[cam]:raise ValueError('RGB timestamp mismatch')
            for fi,ts in enumerate(times[cam]):
                im=overlay(reader.read(fi),cam,fi,ts,origin,byframe[cam,fi],obs,geometry,serials)
                writer.add(im,ts-times[cam][0])
                if fi in (0,799,1598,len(times[cam])-1):cv2.imwrite(str(output/f'{cam}_preview_{fi:04d}.jpg'),im)
                if (fi+1)%400==0:print(f'{cam} annotated video: {fi+1}/{len(times[cam])}',flush=True)
        finally:reader.close();writer.close()
    montage(output,groups,times)


def montage(output,groups,times):
    readers={c:av.open(str(output/(c+'_annotated.mp4'))) for c in CAMERAS}
    iterators={c:iter(reader.decode(video=0)) for c,reader in readers.items()};indices={c:-1 for c in CAMERAS}
    height=json.loads((output/'pose_metadata.json').read_text()).get('video_dimensions',[1920,1380])[1]
    frames={};writer=VideoWriter(output/'all_cameras_annotated.mp4',1920,height)
    try:
        for group in groups:
            if 'zed_left' not in group['frames']:continue
            tiles=[]
            for cam in CAMERAS:
                target=group['frames'].get(cam)
                if target is None:
                    tile=np.zeros((540,960,3),np.uint8);text(tile,cam+' | no aligned frame',(40,60),.8)
                else:
                    while indices[cam]<target:
                        indices[cam]+=1;f=next(iterators[cam]);frames[cam]=f
                        if round(f.pts*f.time_base*1_000_000)!=(times[cam][indices[cam]]-times[cam][0])//1000:raise ValueError('Encoded time mismatch')
                    native=frames[cam].to_ndarray(format='bgr24')
                    if cam=='zed_left':panel=native[1080:].copy()
                    tile=cv2.resize(native[:1080],(960,540),interpolation=cv2.INTER_AREA)
                tiles.append(tile)
            combined=cv2.vconcat([cv2.hconcat(tiles[:2]),cv2.hconcat(tiles[2:]),panel])
            writer.add(combined,group['timestamp_ns']-times['zed_left'][0])
            fi=group['frames']['zed_left']
            if fi in (0,770,1540,2310):cv2.imwrite(str(output/f'overview_preview_{fi:04d}.jpg'),combined)
            if (fi+1)%500==0:print(f'Four-view overview: {fi+1}/{len(times["zed_left"])}',flush=True)
    finally:
        writer.close()
        for reader in readers.values():reader.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take',type=Path);parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--data-only',action='store_true');modes.add_argument('--render-only',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(4);args.output.mkdir(parents=True,exist_ok=True)
    if not args.render_only:
        if not (args.output/'timing_alignment.json').exists():
            from mchoi.aruco.timing import estimate
            estimate(args.take.resolve(),args.calibration.resolve(),args.output.resolve())
        prepare(args.take.resolve(),args.calibration.resolve(),args.output.resolve())
        from mchoi.aruco.track import track
        track(args.output.resolve())
    if not args.data_only:render(args.take.resolve(),args.calibration.resolve(),args.output.resolve())
