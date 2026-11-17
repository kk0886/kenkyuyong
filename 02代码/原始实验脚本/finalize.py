from pathlib import Path
import csv,json,shutil,html,time
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
def finish():
 dest=Path('E:/桌面/盲道阴影训练演示_20261116');dest.mkdir(exist_ok=True)
 summaries=[]
 for arm in ['A_control','B_shadow']:
  run=ROOT/'runs'/arm;rows=list(csv.DictReader((run/'results.csv').open(encoding='utf-8-sig')))
  rows=[{k.strip():v for k,v in r.items()} for r in rows]
  best=max(rows,key=lambda r:float(r.get('metrics/mAP50-95(M)',0)))
  summaries.append({'arm':arm,'epochs_completed':len(rows),'best_development_mask_mAP50_95':float(best.get('metrics/mAP50-95(M)',0)),'best_development_mask_mAP50':float(best.get('metrics/mAP50(M)',0)),'best_metric_epoch':best.get('epoch'),'note':'best CSV mask metric epoch may differ from checkpoint combined fitness; not pixel IoU or independent test'})
  target=dest/arm;target.mkdir(exist_ok=True)
  for n in ['results.csv','results.png','args.yaml']:
   if (run/n).exists():shutil.copy2(run/n,target/n)
  (target/'weights').mkdir(exist_ok=True)
  for n in ['best.pt','last.pt']:
   if (run/'weights'/n).exists():shutil.copy2(run/'weights'/n,target/'weights'/n)
 for n in ['frozen_plan.json','preparation.json','training_manifest.json']:
  shutil.copy2(ROOT/n,dest/n)
 (ROOT/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
 shutil.copy2(ROOT/'comparison.json',dest/'comparison.json')
 body='<meta charset="utf-8"><title>盲道阴影训练演示</title><style>body{font:18px sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}img{width:100%}</style><h1>盲道阴影增强：新训练开发演示</h1><p>A：普通微调；B：同批训练图增加合成柔和阴影。通用YOLO11n-seg初始化，859训练、527开发评价，416分辨率，每组最多10轮。</p><p><strong>这是开发实验，数据已被接触；没有独立真实阴影分组成绩。合成阴影不等于真实阴影，不能据此宣称最终精度提升。</strong></p><pre>'+html.escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre>'
 for a in ['A_control','B_shadow']:body+='<h2>'+a+'</h2><p>模型文件：'+a+'/weights/best.pt</p><img src="'+a+'/results.png">'
 # Same selected development examples for both freshly trained models, no accuracy claims.
 from ultralytics import YOLO
 manifest=json.loads((ROOT/'training_manifest.json').read_text(encoding='utf-8'))
 wanted=['Part01_0003','Part01_0019','Part02_0286','Part04_0717','Part05_0737','Part06_1480','Part07_0873','Part09_1232']
 sources={Path(r['source']).stem:r['source'] for r in manifest}
 imgs=[Path(sources[k]) for k in wanted if k in sources]
 preds={}
 for arm in ['A_control','B_shadow']:
  model=YOLO(str(ROOT/'runs'/arm/'weights/best.pt'))
  preds[arm]=model.predict([Image.open(p).convert('RGB') for p in imgs],imgsz=416,device=0,verbose=False,save=False)
 body+='<h2>同图分割演示</h2><p>左原图，中A，右B。训练侧选例，仅演示，未按真实阴影人工分组，不计算独立准确率。</p>'
 for i,p in enumerate(imgs):
  original=Image.open(p).convert('RGB');panels=[original,Image.fromarray(preds['A_control'][i].plot()[...,::-1]),Image.fromarray(preds['B_shadow'][i].plot()[...,::-1])]
  canvas=Image.new('RGB',(960,350),'white');draw=ImageDraw.Draw(canvas)
  for j,im in enumerate(panels):im.thumbnail((320,320));canvas.paste(im,(j*320,25));draw.text((j*320+5,5),['Original','A control','B shadow'][j],fill='black')
  name=p.stem+'_comparison.jpg';canvas.save(dest/name);body+='<h3>'+p.stem+'</h3><img src="'+name+'">'
 (dest/'打开这里_新训练结果.html').write_text(body,encoding='utf-8')
 (ROOT/'delivery.json').write_text(json.dumps({'desktop':str(dest),'summary':summaries,'completed_at':time.time()},ensure_ascii=False,indent=2),encoding='utf-8')
 return str(dest)
if __name__=='__main__':print(finish())
