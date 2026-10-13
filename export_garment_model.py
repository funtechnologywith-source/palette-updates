"""
Converts the free garment detector (RF-DETR-Seg-Small fine-tuned on Fashionpedia,
Apache-2.0, https://huggingface.co/resoa/garment-detector-seg) into a small int8
ONNX file the phone can run, then checks it on the flat-lay fixtures.

Runs in CI (.github/workflows/export-garment-model.yml), never on the dev PC.
Usage: python scripts/export_garment_model.py <checkpoint.pth> <resolution> <out_dir>
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

# Fashionpedia categories in dataset order (Jia et al. 2020, CC BY 4.0).
CLASSES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants',
    'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
    'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
    'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead',
    'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel',
]

ckpt, res, out_dir = os.path.abspath(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
assert res % 24 == 0, 'RF-DETR resolutions must be a multiple of 24'
os.makedirs(out_dir, exist_ok=True)

from rfdetr import RFDETRSegSmall  # noqa: E402  (heavy import after arg checks)

try:
    model = RFDETRSegSmall(pretrain_weights=ckpt, resolution=res)
except Exception as e:  # older checkpoints hold training args, which the safe loader refuses
    print('safe load failed, loading the pinned checkpoint as trusted:', e)
    model = RFDETRSegSmall(pretrain_weights=ckpt, resolution=res, trust_checkpoint=True)
model.export(output_dir='export_tmp')
# The file name differs between rfdetr versions (inference_model.onnx, rfdetr-seg-small.onnx).
found = sorted(glob.glob(os.path.join('export_tmp', '*.onnx')))
assert found, os.listdir('export_tmp')
fp32 = found[0]

int8 = os.path.join(out_dir, f'garment_rfdetr_seg_s_{res}_q8.onnx')
# Per-channel keeps the int8 weights close to fp32; plain per-tensor reordered detections.
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)

m = onnx.load(int8)
io = {
    'inputs': [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
    'outputs': [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
}
print('IO', json.dumps(io))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(path):
    im = Image.open(path).convert('RGB').resize((res, res), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


# Ids 0-26 are whole pieces (shirt ... umbrella); 27+ are parts (sleeve, pocket ...) the app ignores.
GARMENT = 27


def top(sess, x, k=6):
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))
    logits = next(v for v in outs.values() if v.ndim == 3 and v.shape[-1] > 4)
    boxes = next(v for v in outs.values() if v.ndim == 3 and v.shape[-1] == 4)
    p = 1 / (1 + np.exp(-logits[0][:, :GARMENT]))
    cls, score = p.argmax(-1), p.max(-1)
    order = score.argsort()[::-1][:k]
    return [(int(cls[i]), float(score[i]), boxes[0][i].round(3).tolist()) for i in order], outs


def name(c):
    return CLASSES[c] if c < len(CLASSES) else '?'


def iou(a, b):  # boxes are cx, cy, w, h
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    i = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def matched(a, b, th=0.3):
    """How many confident fp32 pieces int8 also finds (same class, box IoU >= 0.7)."""
    want = [d for d in a if d[1] >= th]
    got = [d for d in b if d[1] >= th * 0.8]
    return sum(1 for c, _, bx in want if any(c == c2 and iou(bx, bx2) >= 0.7 for c2, _, bx2 in got)), len(want)


s32 = ort.InferenceSession(fp32, providers=['CPUExecutionProvider'])
s8 = ort.InferenceSession(int8, providers=['CPUExecutionProvider'])
report = []
totals = [0, 0]
for path in sorted(glob.glob('e2e/fixtures/flatlay/*.jpg')):
    x = prep(path)
    a, outs = top(s32, x, 10)
    b, _ = top(s8, x, 20)
    hit, want = matched(a, b)
    totals[0] += hit
    totals[1] += want
    print(f'\n{os.path.basename(path)}  int8 finds {hit}/{want} of the fp32 pieces')
    for c, s, bx in a:
        if s >= 0.3:
            print(f'   {s:.2f}  [{c}] {name(c)}  box={bx}')
    report.append({'image': os.path.basename(path), 'fp32': a, 'int8': b, 'shapes': {k: list(v.shape) for k, v in outs.items()}})

print(f'\nAGREE int8 finds {totals[0]}/{totals[1]} fp32 pieces across fixtures')
with open(int8, 'rb') as f:
    digest = hashlib.sha256(f.read()).hexdigest()
meta = {
    'file': os.path.basename(int8),
    'bytes': os.path.getsize(int8),
    'sha256': digest,
    'resolution': res,
    'io': io,
    'classes': CLASSES,
    'source': 'https://huggingface.co/resoa/garment-detector-seg (Apache-2.0), trained on Fashionpedia (CC BY 4.0)',
}
json.dump(meta, open(os.path.join(out_dir, 'garment_model.json'), 'w'), indent=2)
json.dump(report, open(os.path.join(out_dir, 'fixture_report.json'), 'w'), indent=1)
shutil.copy(__file__, os.path.join(out_dir, 'export_garment_model.py'))
print('\nMETA', json.dumps({k: meta[k] for k in ('file', 'bytes', 'sha256', 'resolution')}))
