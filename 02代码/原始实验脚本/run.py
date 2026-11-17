from pathlib import Path
import os,json,time,hashlib,shutil,random,traceback
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
from PIL import Image,ImageDraw,ImageFilter
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'yolo_development_pilot_20261109_run_002'
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();initial=json.loads((OLD/'frozen_plan.json').read_text(encoding='utf-8'))['initialization']
 assert sha(initial)=='55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152'
 if (ROOT/'runs').exists():raise RuntimeError('Existing trained experiment; do not overwrite/retrain')
 plan={'authorization':'2026-11-16 user requests newly trained demonstration and automatic next steps','purpose':'development demonstration; no independent or shadow subgroup accuracy claim','train':859,'val':527,'source':'existing TP grouped development manifest; no Axonix samples mixed','historically_exposed':True,'initialization':initial,'initialization_sha256':sha(initial),'epochs_per_arm_cap':10,'time_hours_per_arm_cap':0.9,'arms':['A_control','B_shadow'],'only_changed_factor':'deterministic extra soft polygon darkening on ~50% training images in B; A/B PNG re-encoding identical','synthetic_shadow_is_real_shadow':False,'seed':42,'imgsz':416,'batch':8,'workers':0,'amp':False,'mask_ratio':4,'overlap_mask':True,'optimizer':'AdamW','lr0':0.001,'new_downloads':False,'final_test':None,'human_condition_truth':None}
 save('frozen_plan.json',plan);save('status.json',{'phase':'prepare','started':start})
 sources=[Path(s) for s in (OLD/'train.txt').read_text().splitlines() if s.strip()];assert len(sources)==859
 manifest=[]
 for arm in plan['arms']:
  (ROOT/arm/'images/train').mkdir(parents=True,exist_ok=True);(ROOT/arm/'labels/train').mkdir(parents=True,exist_ok=True)
 for i,src in enumerate(sources):
  label=Path(str(src).replace('/images/','/labels/')).with_suffix('.txt')
  if not label.exists():label=Path(str(src).replace('\\images\\','\\labels\\')).with_suffix('.txt')
  pic=Image.open(src).convert('RGB');rng=random.Random('shadow_ab_20261116_'+src.stem);apply=rng.random()<0.5
  w,h=pic.size;mask=Image.new('L',(w,h));draw=ImageDraw.Draw(mask)
  x=rng.uniform(-.2,.6)*w;span=rng.uniform(.25,.6)*w
  draw.polygon([(x,0),(x+span,0),(x+span+.25*w,h),(x+.25*w,h)],fill=255)
  mask=mask.filter(ImageFilter.GaussianBlur(max(2,w*.012)))
  strength=rng.uniform(.25,.5)
  dark=Image.fromarray(np.clip(np.asarray(pic,dtype=np.float32)*(1-strength*np.asarray(mask,dtype=np.float32)[...,None]/255),0,255).astype('uint8')) if apply else pic
  row={'source':str(src),'source_sha256':sha(src),'label_sha256':sha(label),'extra_shadow':apply,'strength':strength if apply else None}
  for arm,im in zip(plan['arms'],[pic,dark]):
   target=ROOT/arm/'images/train'/(src.stem+'.png');im.save(target);shutil.copy2(label,ROOT/arm/'labels/train'/label.name);row[arm+'_image_sha256']=sha(target)
  manifest.append(row)
 save('training_manifest.json',manifest)
 val=(OLD/'val.txt').read_text()
 for arm in plan['arms']:
  base=ROOT/arm;(base/'val.txt').write_text(val,encoding='utf-8')
  (base/'data.yaml').write_text('path: '+base.as_posix()+'\ntrain: images/train\nval: val.txt\nnames:\n  0: tactile_paving\n',encoding='utf-8')
 save('preparation.json',{'extra_shadow_images':sum(r['extra_shadow'] for r in manifest),'same_labels':True,'validation_list_sha256':sha(OLD/'val.txt'),'elapsed_s':time.time()-start})
 from ultralytics import YOLO
 for arm in plan['arms']:
  save('status.json',{'phase':'training','arm':arm,'elapsed_s':time.time()-start});print('START '+arm,flush=True)
  model=YOLO(initial)
  arm_start=time.time()
  def bounded_epoch(trainer):
   if time.time()-arm_start>=3240:trainer.stop=True
  model.add_callback('on_train_epoch_end',bounded_epoch)
  model.train(data=str(ROOT/arm/'data.yaml'),epochs=10,imgsz=416,batch=8,device=0,workers=0,amp=False,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,mask_ratio=4,overlap_mask=True,project=str(ROOT/'runs'),name=arm,exist_ok=False,plots=True,save=True,patience=100,cache=False)
  del model
 save('status.json',{'phase':'training_complete','elapsed_s':time.time()-start,'next':'compare development results and create desktop demonstration; no further training'})
 from finalize import finish
 desktop=finish()
 save('status.json',{'phase':'delivered','elapsed_s':time.time()-start,'desktop':desktop})
if __name__=='__main__':
 try:main()
 except Exception:
  save('failure.json',{'traceback':traceback.format_exc()});raise
