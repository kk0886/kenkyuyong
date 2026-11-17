from pathlib import Path
import argparse
from PIL import Image,ImageOps
from ultralytics import YOLO
import os
os.environ['YOLO_AUTOINSTALL']='false'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--out',default='PT检测结果');a=p.parse_args();out=Path(a.out)
 if out.exists():raise SystemExit('Output exists, choose another --out')
 out.mkdir(parents=True);im=ImageOps.exif_transpose(Image.open(a.image)).convert('RGB');r=YOLO(a.model).predict(im,imgsz=416,rect=False,conf=.25,iou=.7,retina_masks=True,device='cpu',verbose=False)[0];Image.fromarray(r.plot()[...,::-1]).save(out/'overlay.jpg');r.save_txt(str(out/'predictions.txt'),save_conf=True)
