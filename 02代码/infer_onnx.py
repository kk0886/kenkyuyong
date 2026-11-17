"""Portable CPU ONNX reference. No server, API or uploads. Run from package root."""
from pathlib import Path
import argparse,json,time
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps

def infer(model,image,conf=.25,nms_iou=.7):
 session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
 # ImageOps.exif_transpose ensures a phone photo's displayed rotation is used.
 rgb=np.asarray(ImageOps.exif_transpose(Image.open(image)).convert('RGB'));h,w=rgb.shape[:2]
 H,W=session.get_inputs()[0].shape[2:];ratio=min(W/w,H/h);nw,nh=round(w*ratio),round(h*ratio)
 left=round((W-nw)/2-.1);top=round((H-nh)/2-.1)
 canvas=np.full((H,W,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
 x=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)/255
 t=time.perf_counter();outputs=session.run(None,{session.get_inputs()[0].name:x});ms=(time.perf_counter()-t)*1000
 pred=next(o for o in outputs if o.ndim==3)[0].T;proto=next(o for o in outputs if o.ndim==4)[0]
 nm=proto.shape[0];nc=pred.shape[1]-4-nm
 if nc!=1:raise ValueError('This reference supports the packaged one-class models only')
 scores=pred[:,4];sel=scores>=conf;pred=pred[sel];scores=scores[sel];union=np.zeros((h,w),bool);detections=[]
 if len(pred):
  xywh=pred[:,:4].copy();boxes=xywh.copy();boxes[:,:2]-=xywh[:,2:]/2
  keep=cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),conf,nms_iou)
  for idx in np.asarray(keep).reshape(-1):
   row=pred[idx];cx,cy,bw,bh=row[:4];xyxy=np.array([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],float)
   xyxy[[0,2]]=(xyxy[[0,2]]-left)/ratio;xyxy[[1,3]]=(xyxy[[1,3]]-top)/ratio;xyxy[[0,2]]=np.clip(xyxy[[0,2]],0,w);xyxy[[1,3]]=np.clip(xyxy[[1,3]],0,h)
   logits=(row[4+nc:]@proto.reshape(nm,-1)).reshape(proto.shape[1:])
   # Remove letterbox padding in prototype space, then resize logits to original.
   ph,pw=logits.shape;scale=min(pw/w,ph/h);padw=(pw-w*scale)/2;padh=(ph-h*scale)/2
   x0,y0=round(padw-.1),round(padh-.1);x1,y1=pw-round(padw+.1),ph-round(padh+.1)
   resized=cv2.resize(logits[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_LINEAR)
   mask=resized>0 # sigmoid(logit)>0.5 is equivalent to logit>0
   xx=np.arange(w)[None,:];yy=np.arange(h)[:,None];mask&=(xx>=xyxy[0])&(xx<xyxy[2])&(yy>=xyxy[1])&(yy<xyxy[3]);union|=mask
   detections.append({'class_id':0,'class_name':'tactile_paving','confidence':float(scores[idx]),'xyxy_original_pixels':xyxy.tolist(),'mask_pixels':int(mask.sum())})
 overlay=rgb.copy();overlay[union]=(overlay[union]*.45+np.array([0,220,70])*.55).astype(np.uint8)
 return union,overlay,{'onnx_run_ms_excludes_session_load_and_preprocess':ms,'original_size':[w,h],'input_shape':list(x.shape),'detections':detections,'conf_threshold':conf,'nms_iou':nms_iou,'mask_threshold':.5,'note':'No navigation or safety decision; image segmentation only'}

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--image',required=True);a.add_argument('--out',default='本次检测结果');args=a.parse_args();out=Path(args.out)
 if out.exists():raise SystemExit('Output exists. Choose a new --out directory to preserve results.')
 out.mkdir(parents=True);mask,overlay,info=infer(args.model,args.image);Image.fromarray(mask.astype(np.uint8)*255).save(out/'mask.png');Image.fromarray(overlay).save(out/'overlay.jpg');(out/'result.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(info,ensure_ascii=False))
