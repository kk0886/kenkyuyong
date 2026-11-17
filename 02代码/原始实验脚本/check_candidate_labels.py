from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageOps
p=Path(__file__).resolve().parent;src=p/'真实条件开发评价候选';out=src/'原始标签核对';out.mkdir(exist_ok=False)
tp=Path('C:/Users/okiki/Documents/Codex/2026-10-13/referenced-chatgpt-conversation-this-is-an/outputs/navigation-study/public_data/TP-Dataset')
rows=json.loads((src/'manifest.json').read_text(encoding='utf-8'));res=[];tiles=[]
for r in rows:
 f=Path(r['source']);part,num=f.stem.split('_');orig=tp/'JPEGImages'/part/(num+'.jpg');mp=tp/'GroundTruth'/part/(num+'.png');label=Path(str(f).replace('\\images\\','\\labels\\').replace('/images/','/labels/')).with_suffix('.txt')
 im=Image.open(f).convert('RGB');mask=Image.open(mp);raw=np.array(mask);assert raw.ndim==2 and mask.size==Image.open(orig).size
 ref=np.array(mask.resize(im.size,Image.Resampling.NEAREST))>0;poly=Image.new('L',im.size);d=ImageDraw.Draw(poly)
 for line in label.read_text().splitlines():
  v=list(map(float,line.split()));coords=v[1:];pts=[(coords[j]*im.width,coords[j+1]*im.height) for j in range(0,len(coords),2)];d.polygon(pts,fill=255)
 a=np.array(poly)>0;inter=int((a&ref).sum());union=int((a|ref).sum());iou=inter/union if union else None
 info={'stem':f.stem,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'author_image_sha256':hashlib.sha256(orig.read_bytes()).hexdigest(),'author_mask_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'converted_label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'author_mask_values':np.unique(raw).tolist(),'author_mask_nonempty':bool(ref.any()),'polygon_author_iou_at_development_image_size':iou,'reference_resize':'nearest','polygon_rasterizer':'PIL for diagnostic, not full Ultralytics dataloader','human_label_quality':None,'new_model_inference':False}
 res.append(info)
 panel=Image.new('RGB',(900,320),'white')
 for j,(title,m) in enumerate([('Original',None),('Author mask',ref),('Converted polygon',a)]):
  pic=im.copy()
  if m is not None:
   arr=np.array(pic);arr[m]=(arr[m]*.45+np.array([0,220,70])*.55).astype('uint8');pic=Image.fromarray(arr)
  panel.paste(ImageOps.contain(pic,(300,285)),(j*300,25));ImageDraw.Draw(panel).text((j*300+4,5),title,fill='black')
 panel.save(out/(f.stem+'.jpg'));tiles.append(panel)
sheet=Image.new('RGB',(1800,320*6),'white')
for i,t in enumerate(tiles):sheet.paste(t,((i%2)*900,(i//2)*320))
sheet.save(out/'contact.jpg');(out/'results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print([(r['stem'],round(r['polygon_author_iou_at_development_image_size'],4)) for r in res])
