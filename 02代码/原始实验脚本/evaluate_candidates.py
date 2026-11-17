from pathlib import Path
import json,time,hashlib,os
os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from ultralytics import YOLO
p=Path(__file__).resolve().parent;src=p/'真实条件开发评价候选';out=src/'两模型逐图诊断';out.mkdir(exist_ok=False)
rows=json.loads((src/'manifest.json').read_text(encoding='utf-8'));conditions={Path(r['source']).stem:r['AI_conditions'] for r in json.loads((src/'AI_condition_candidates.json').read_text(encoding='utf-8'))['rows']}
tp=Path('C:/Users/okiki/Documents/Codex/2026-10-13/referenced-chatgpt-conversation-this-is-an/outputs/navigation-study/public_data/TP-Dataset')
preds={};timings={};weights={}
for arm in ['A_control','B_shadow']:
 weight=p/'runs'/arm/'weights/best.pt';weights[arm]=hashlib.sha256(weight.read_bytes()).hexdigest();model=YOLO(str(weight));preds[arm]=[];timings[arm]=[]
 for r in rows:
  im=Image.open(r['source']).convert('RGB');t=time.perf_counter();result=model.predict(im,imgsz=416,conf=.25,iou=.7,device=0,retina_masks=True,verbose=False)[0];timings[arm].append((time.perf_counter()-t)*1000)
  mask=np.zeros((im.height,im.width),dtype=bool)
  if result.masks is not None:mask=result.masks.data.cpu().numpy().max(axis=0)>.5
  assert mask.shape==(im.height,im.width);preds[arm].append(mask)
def stat(a,g):
 tp=int((a&g).sum());fp=int((a&~g).sum());fn=int((~a&g).sum());return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn) if tp+fp+fn else None,'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
records=[];body='<meta charset="utf-8"><style>body{max-width:1300px;margin:25px auto;font:18px sans-serif}img{width:100%}</style><h1>12张真实场景开发诊断</h1><p>同图：原图、作者标签、A预测、B预测。历史开发图，条件仅AI草案，未经人工确认；不是独立测试或真实阴影分组成绩。固定conf=.25，416输入。IoU为对作者掩膜的像素重叠，不能与此前mAP数字直接比较。</p>'
for i,r in enumerate(rows):
 f=Path(r['source']);part,num=f.stem.split('_');im=Image.open(f).convert('RGB');m=tp/'GroundTruth'/part/(num+'.png');g=np.array(Image.open(m).resize(im.size,Image.Resampling.NEAREST))>0
 info={'stem':f.stem,'AI_conditions':conditions[f.stem],'human_conditions':None,'author_mask_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'evaluation':'historical development image; author reference resized nearest to development image grid','metrics':{arm:stat(preds[arm][i],g) for arm in preds}}
 info['B_minus_A_iou']=info['metrics']['B_shadow']['iou']-info['metrics']['A_control']['iou'];records.append(info)
 np.savez_compressed(out/(f.stem+'.npz'),reference=g,A=preds['A_control'][i],B=preds['B_shadow'][i])
 panel=Image.new('RGB',(1280,350),'white')
 for j,(name,mask) in enumerate([('Original',None),('Author reference',g),('A control',preds['A_control'][i]),('B shadow',preds['B_shadow'][i])]):
  arr=np.array(im).copy()
  if mask is not None:arr[mask]=(arr[mask]*.4+np.array([0,200,80])*.6).astype('uint8')
  panel.paste(ImageOps.contain(Image.fromarray(arr),(320,315)),(320*j,25));ImageDraw.Draw(panel).text((320*j+4,5),name,fill='black')
 name=f.stem+'.jpg';panel.save(out/name);body+='<h2>'+f.stem+'</h2><p>A IoU '+format(info['metrics']['A_control']['iou'],'.3f')+' / B IoU '+format(info['metrics']['B_shadow']['iou'],'.3f')+'</p><img src="'+name+'">'
summary={'count':len(records),'AI_candidate_conditions_not_human_truth':True,'independent_test':False,'A_mean_image_iou':float(np.mean([r['metrics']['A_control']['iou'] for r in records])),'B_mean_image_iou':float(np.mean([r['metrics']['B_shadow']['iou'] for r in records])),'improved_images':sum(r['B_minus_A_iou']>0 for r in records),'regressed_images':sum(r['B_minus_A_iou']<0 for r in records),'weights_sha256':weights,'API_ms_per_image_including_first_warmup':timings,'note':'12 selected images only; no dataset-wide or subgroup gain claim; fixed author reference may contain semantic ambiguity'}
(out/'results.json').write_text(json.dumps({'summary':summary,'images':records},ensure_ascii=False,indent=2),encoding='utf-8');(out/'打开这里_12图诊断.html').write_text(body,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
