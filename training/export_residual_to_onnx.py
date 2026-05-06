#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--artifact', required=True, help='JSON artifact from train_board_cnn.py')
p.add_argument('--out', required=True, help='Output .onnx path')
p.add_argument('--opset', type=int, default=17)
p.add_argument('--verify', action='store_true', help='Run onnxruntime shape smoke if available')
args=p.parse_args()

try:
  import torch
  import torch.nn as nn
except Exception as e:
  raise SystemExit('PyTorch is required. Install with: pip install -r requirements-onnx.txt') from e

class ResidualStudent(nn.Module):
  def __init__(self, artifact: dict):
    super().__init__(); self.artifact=artifact
    c1w=torch.tensor(artifact['c1_weight'], dtype=torch.float32); c1b=torch.tensor(artifact['c1_bias'], dtype=torch.float32)
    in_planes=c1w.shape[1]; channels=c1w.shape[0]
    self.stem=nn.Conv2d(in_planes, channels, 3, padding=1)
    self.stem.weight.data.copy_(c1w); self.stem.bias.data.copy_(c1b)
    self.blocks=nn.ModuleList()
    for b in artifact.get('residual_blocks', []):
      c1=nn.Conv2d(channels, channels, 3, padding=1); c2=nn.Conv2d(channels, channels, 3, padding=1)
      c1.weight.data.copy_(torch.tensor(b['c1_weight'], dtype=torch.float32)); c1.bias.data.copy_(torch.tensor(b['c1_bias'], dtype=torch.float32))
      c2.weight.data.copy_(torch.tensor(b['c2_weight'], dtype=torch.float32)); c2.bias.data.copy_(torch.tensor(b['c2_bias'], dtype=torch.float32))
      self.blocks.append(nn.ModuleList([c1,c2]))
    policy_w=torch.tensor(artifact['policy_weight'], dtype=torch.float32); policy_b=torch.tensor(artifact['policy_bias'], dtype=torch.float32)
    wdl_w=torch.tensor(artifact['wdl_weight'], dtype=torch.float32); wdl_b=torch.tensor(artifact['wdl_bias'], dtype=torch.float32)
    self.policy=nn.Linear(policy_w.shape[1], policy_w.shape[0]); self.policy.weight.data.copy_(policy_w); self.policy.bias.data.copy_(policy_b)
    self.wdl=nn.Linear(wdl_w.shape[1], wdl_w.shape[0]); self.wdl.weight.data.copy_(wdl_w); self.wdl.bias.data.copy_(wdl_b)
    self.spatial=artifact.get('policy_head') == 'spatial'
  def forward(self, x):
    h=torch.relu(self.stem(x))
    for c1,c2 in self.blocks: h=torch.relu(c2(torch.relu(c1(h))) + h)
    pooled=h.mean(dim=(2,3)); pf=h.flatten(1) if self.spatial else pooled
    return self.policy(pf), self.wdl(pooled)

artifact=json.loads(Path(args.artifact).read_text())
if artifact.get('architecture') != 'residual_tower' and artifact.get('kind') != 'tiny_board_residual_student':
  raise SystemExit('Expected residual_tower / tiny_board_residual_student artifact')
model=ResidualStudent(artifact).eval()
in_planes=int(artifact.get('input_planes') or torch.tensor(artifact['c1_weight']).shape[1])
dummy=torch.zeros(1, in_planes, 8, 8, dtype=torch.float32)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, dummy, args.out, input_names=['planes'], output_names=['policy_logits','wdl_logits'], dynamic_axes={'planes':{0:'batch'},'policy_logits':{0:'batch'},'wdl_logits':{0:'batch'}}, opset_version=args.opset)
print(f'METRIC onnx_input_planes={in_planes}')
print(f'METRIC onnx_policy_size={len(artifact["moves"])}')
print(f'METRIC onnx_blocks={len(artifact.get("residual_blocks", []))}')
if args.verify:
  try:
    import onnxruntime as ort
    sess=ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    outs=sess.run(None, {'planes': dummy.numpy()})
    print(f'METRIC onnx_verify_policy_shape={outs[0].shape[1]}')
    print(f'METRIC onnx_verify_wdl_shape={outs[1].shape[1]}')
  except Exception as e:
    raise SystemExit(f'ONNX export wrote file, but verify failed: {e}') from e
