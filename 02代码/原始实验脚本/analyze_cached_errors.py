from pathlib import Path
import json,hashlib,shutil
import numpy as np
p=Path(__file__).resolve().parent/'真实条件开发评价候选/两模型逐图诊断'
j=json.loads((p/'results.json').read_text(encoding='utf-8'));records=[]
for r in j['images']:
 f=p/(r['stem']+'.npz')
 with np.load(f) as z:
  g=z['reference'];m={a:z[k] for a,k in [('A_control','A'),('B_shadow','B')]}
  for arm,a in m.items():
   actual={'tp':int((a&g).sum()),'fp':int((a&~g).sum()),'fn':int((~a&g).sum())}
   assert all(actual[k]==r['metrics'][arm][k] for k in actual)
  a,b=m['A_control'],m['B_shadow']
  rec={'stem':r['stem'],'npz_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'iou_delta':r['B_minus_A_iou'],'new_correct_foreground':int((~a&b&g).sum()),'lost_correct_foreground':int((a&~b&g).sum()),'removed_false_foreground':int((a&~b&~g).sum()),'new_false_foreground':int((~a&b&~g).sum())}
  rec['A']=r['metrics']['A_control'];rec['B']=r['metrics']['B_shadow'];records.append(rec)
def total(rs):
 out={}
 for arm in ['A','B']:
  t={k:sum(r[arm][k] for r in rs) for k in ['tp','fp','fn']};t['foreground_recall']=t['tp']/(t['tp']+t['fn']);t['foreground_precision']=t['tp']/(t['tp']+t['fp']);t['pixel_aggregate_iou']=t['tp']/(t['tp']+t['fp']+t['fn']);out[arm]=t
 return out
result={'all12':total(records),'without_influential_0572':total([r for r in records if r['stem']!='Part03_0572']),'images':records,'note':'pixel-weighted cached diagnostic relative to author reference; not equal-image mean, instance mAP, human truth, independent or confirmed shadow subgroup'}
(p/'cached_error_tradeoff.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 老师追问：到底少漏了，还是少误认了？','','以下仅复用12张开发图预测缓存，计数与原结果逐图核验一致，没有重跑模型。参照作者掩膜，不是新人工裁决。','','| 范围/指标 | A普通微调 | B额外阴影 |','|---|---:|---:|']
for scope,title in [('all12','全部12图'),('without_influential_0572','去最大影响例的11图（仅敏感性）')]:
 a,b=result[scope]['A'],result[scope]['B']
 for key,label in [('foreground_recall','盲道像素召回率'),('foreground_precision','盲道像素精确率'),('pixel_aggregate_iou','汇总像素IoU')]:lines.append(f'| {title}：{label} | {a[key]*100:.2f}% | {b[key]*100:.2f}% |')
lines+=['','这些是按像素汇总的诊断，大面积图片影响较大，不等于逐图平均；不能直接与完整527图实例mAP比较。','', '逐图发生了什么：','', '| 图像 | B−A IoU百分点 | 少漏的像素净量 | 少误认的像素净量 |','|---|---:|---:|---:|']
for r in records:lines.append(f"| {r['stem']} | {r['iou_delta']*100:+.2f} | {r['new_correct_foreground']-r['lost_correct_foreground']:+d} | {r['removed_false_foreground']-r['new_false_foreground']:+d} |")
lines+=['','正值代表相对A改善，负值代表恶化；不同尺寸不能用绝对像素量排难度。','', '汇报口径：完整开发mAP略降，小批像素重叠有升有降；主要改善例是A未检出、B检出的0572。不能据此认定真实阴影专项改善，也不能只展示有利例。','', '下一步真正缺的是可靠的条件人工确认和按来源隔离的真实评价，而不是再调一轮参数追分。现有全部失败例与权重保留。']
(p/'老师追问_漏分与误分.md').write_text('\n'.join(lines),encoding='utf-8')
d=Path('E:/桌面/盲道阴影训练演示_20261116/12张真实场景开发诊断')
for name in ['cached_error_tradeoff.json','老师追问_漏分与误分.md']:shutil.copy2(p/name,d/name)
print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False))
