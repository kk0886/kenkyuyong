import argparse,os
os.environ['YOLO_AUTOINSTALL']='false'
from pathlib import Path
from ultralytics import YOLO
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();target=Path(a.model).with_suffix('.onnx')
 if target.exists():raise SystemExit('ONNX exists; preserve it. Export a separate copy if needed.')
 YOLO(a.model).export(format='onnx',imgsz=416,batch=1,opset=17,dynamic=False,simplify=False,nms=False,half=False,device='cpu')
