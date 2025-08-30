#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    import timm
except Exception as e:
    timm = None


@dataclass
class LayerSpec:
    name: str
    kind: str  # qkv, q, k, v, proj, fc1, fc2, other
    block: Optional[int]
    shape: Tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline spectral analysis for timm models")
    p.add_argument('--model', type=str, required=True, help='timm model name, e.g. vit_base_patch16_224')
    p.add_argument('--pretrained', action='store_true', help='Load pretrained weights from timm')
    p.add_argument('--checkpoint', type=str, default='', help='Optional local checkpoint (.pth) to load')
    p.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    p.add_argument('--targets', type=str, default='attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
                   help='Comma-separated substrings to select named_modules for analysis')
    p.add_argument('--topk', type=int, default=8, help='Top-k and last-k singular components to keep')
    p.add_argument('--full', action='store_true', help='Save full singular spectrum (values and optionally vectors)')
    p.add_argument('--save-dir', type=str, default='', help='Directory to save outputs; if empty prints only summary')
    p.add_argument('--format', type=str, default='json', choices=['json', 'npz', 'pt'],
                   help='Output format for saved results (vectors require npz or pt)')
    p.add_argument('--save-vectors', action='store_true', help='Also save singular vectors (U/V)')
    p.add_argument('--svd-on-cpu', action='store_true', help='Force SVD on CPU regardless of model device')
    p.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'],
                   help='Compute dtype for SVD')
    p.add_argument('--split-qkv', action='store_true', help='Split fused qkv linear into separate q/k/v analyses')
    p.add_argument('--max-layers', type=int, default=0, help='Optional limit on number of layers to analyze (>0)')
    return p.parse_args()


def load_model(args: argparse.Namespace) -> nn.Module:
    assert timm is not None, "timm is required: pip install timm"
    model = timm.create_model(args.model, pretrained=args.pretrained)
    model.eval()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")
    model.to(args.device)
    return model


def kind_from_name(name: str) -> str:
    if 'attn.qkv' in name:
        return 'qkv'
    if 'attn.proj' in name:
        return 'proj'
    if 'mlp.fc1' in name:
        return 'fc1'
    if 'mlp.fc2' in name:
        return 'fc2'
    # Common variants in other timm models
    if '.q.' in name or name.endswith('.q'):
        return 'q'
    if '.k.' in name or name.endswith('.k'):
        return 'k'
    if '.v.' in name or name.endswith('.v'):
        return 'v'
    return 'other'


def block_index_from_name(name: str) -> Optional[int]:
    # Expect patterns like 'blocks.0.attn.qkv'
    m = re.search(r'blocks\.(\d+)\.', name)
    if m:
        return int(m.group(1))
    return None


def enumerate_linear_targets(model: nn.Module, patterns: List[str], split_qkv: bool) -> List[Tuple[LayerSpec, nn.Linear, Optional[slice]]]:
    """Return list of (LayerSpec, module, rows_slice) tuples.

    If split_qkv and a fused qkv is detected (out_features divisible by 3), returns three entries
    with slices selecting rows [0:e], [e:2e], [2e:3e]. Otherwise rows_slice is None (use full weight).
    """
    targets: List[Tuple[LayerSpec, nn.Linear, Optional[slice]]] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if patterns and not any(p in name for p in patterns):
            continue
        kind = kind_from_name(name)
        blk = block_index_from_name(name)
        out_f, in_f = m.weight.shape
        if split_qkv and 'attn.qkv' in name and out_f % 3 == 0:
            e = out_f // 3
            parts = [('q', slice(0, e)), ('k', slice(e, 2 * e)), ('v', slice(2 * e, 3 * e))]
            for suf, sl in parts:
                spec = LayerSpec(name=f"{name}/{suf}", kind=suf, block=blk, shape=(sl.stop - sl.start, in_f))
                targets.append((spec, m, sl))
        else:
            spec = LayerSpec(name=name, kind=kind, block=blk, shape=(out_f, in_f))
            targets.append((spec, m, None))
    return targets


@torch.no_grad()
def svd_weight(W: torch.Tensor, device: torch.device, dtype: torch.dtype, svd_on_cpu: bool):
    Wd = W.detach()
    if Wd.is_sparse:
        Wd = Wd.to_dense()
    Wd = Wd.to(dtype)
    if svd_on_cpu:
        Wd = Wd.cpu()
    else:
        Wd = Wd.to(device)
    # full_matrices=False yields compact U, Vh
    U, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
    return U, S, Vh


def to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def analyze_model(args: argparse.Namespace) -> Dict[str, dict]:
    model = load_model(args)
    compute_dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    patterns = [p.strip() for p in args.targets.split(',') if p.strip()]
    targets = enumerate_linear_targets(model, patterns, split_qkv=args.split_qkv)
    if args.max_layers > 0:
        targets = targets[:args.max_layers]

    results: Dict[str, dict] = {}

    for spec, mod, rows_sl in targets:
        W = mod.weight
        if rows_sl is not None:
            W = W[rows_sl, :]
        U, S, Vh = svd_weight(W, device=torch.device(args.device), dtype=compute_dtype, svd_on_cpu=args.svd_on_cpu)
        k = min(args.topk, S.shape[0])

        # Top-k
        Uk = U[:, :k].contiguous() if k > 0 else U[:, :0]
        Vk = Vh[:k, :].transpose(0, 1).contiguous() if k > 0 else Vh[:0, :].transpose(0, 1)
        Sk = S[:k].contiguous() if k > 0 else S[:0]
        # Last-k (tail)
        Ul = U[:, -k:].contiguous() if k > 0 else U[:, :0]
        Vl = Vh[-k:, :].transpose(0, 1).contiguous() if k > 0 else Vh[:0, :].transpose(0, 1)
        Sl = S[-k:].contiguous() if k > 0 else S[:0]

        entry = {
            'meta': asdict(spec),
            'sigma_max': float(S[0].item()) if S.numel() else float('nan'),
            'sigma_min': float(S[-1].item()) if S.numel() else float('nan'),
            'cond_number': float((S[0] / S[-1]).item()) if S.numel() and S[-1] > 0 else float('inf'),
            'sv_topk': to_numpy(Sk).tolist(),
            # sv_last1 is the smallest singular value
            'sv_lastk': [float(x) for x in to_numpy(S[-k:])] if k > 0 else [],
        }

        if args.full:
            entry['singular_values'] = to_numpy(S).tolist()
            if args.save_vectors:
                entry['U'] = to_numpy(U)
                entry['V'] = to_numpy(Vh.transpose(0, 1))
        else:
            if args.save_vectors:
                entry['U_topk'] = to_numpy(Uk)
                entry['V_topk'] = to_numpy(Vk)
                entry['U_lastk'] = to_numpy(Ul)
                entry['V_lastk'] = to_numpy(Vl)

        results[spec.name] = entry

    return results


def save_results(results: Dict[str, dict], args: argparse.Namespace) -> None:
    if not args.save_dir:
        return
    os.makedirs(args.save_dir, exist_ok=True)
    # Split numeric small fields to JSON; large arrays to npz/pt when present
    base = os.path.join(args.save_dir, f"{args.model}_spectral")
    # Create a light JSON with meta + numeric scalars and lists (no large arrays)
    light: Dict[str, dict] = {}
    tensor_payload: Dict[str, torch.Tensor] = {}
    for name, entry in results.items():
        light_entry = {
            'meta': entry['meta'],
            'sigma_max': entry.get('sigma_max'),
            'sigma_min': entry.get('sigma_min'),
            'cond_number': entry.get('cond_number'),
            'sv_topk': entry.get('sv_topk', []),
            'sv_lastk': entry.get('sv_lastk', []),
        }
        # collect tensors if present
        for key in ('U', 'V', 'U_topk', 'V_topk', 'U_lastk', 'V_lastk'):
            if key in entry:
                tensor_payload[f"{name}/{key}"] = torch.from_numpy(entry[key])
        if 'singular_values' in entry and isinstance(entry['singular_values'], list):
            light_entry['singular_values'] = entry['singular_values']
        light[name] = light_entry

    json_path = base + '.json'
    with open(json_path, 'w') as f:
        json.dump(light, f, indent=2)
    print(f"[save] wrote {json_path}")

    if tensor_payload and args.format in ('npz', 'pt'):
        if args.format == 'npz':
            # Save as npz with flattened keys; convert to numpy
            import numpy as np
            npz_payload = {k: v.detach().cpu().numpy() for k, v in tensor_payload.items()}
            npz_path = base + '.npz'
            import numpy as _np
            _np.savez_compressed(npz_path, **npz_payload)
            print(f"[save] wrote {npz_path}")
        else:
            pt_path = base + '.pt'
            torch.save(tensor_payload, pt_path)
            print(f"[save] wrote {pt_path}")
    elif tensor_payload and args.format == 'json':
        print('[warn] vectors present but format=json; vectors not saved. Use --format npz or --format pt.')


def print_summary(results: Dict[str, dict]) -> None:
    # Per-layer brief
    rows = []
    for name, entry in results.items():
        meta = entry['meta']
        rows.append((meta['block'], meta['kind'], name, meta['shape'], entry['sigma_max'], entry['sigma_min']))
    rows.sort(key=lambda x: (x[0] if x[0] is not None else 10_000, x[1], x[2]))
    print('\nPer-layer summary:')
    for blk, kind, name, shape, smax, smin in rows:
        bstr = f"block{blk}" if blk is not None else "-"
        cond = float('inf') if (smin == 0 or smin is None) else (smax / smin)
        print(f"  [{bstr:>7}] {kind:>4} | {name:<60} | {str(tuple(shape)):<18} | sigma_max={smax:.4g} sigma_min={smin:.4g} cond={cond:.4g}")

    # Aggregated by kind
    agg: Dict[str, List[float]] = {}
    for entry in results.values():
        kind = entry['meta']['kind']
        agg.setdefault(kind, []).append(entry['sigma_max'])
    print('\nAggregated sigma_max by kind:')
    for kind, vals in sorted(agg.items()):
        import math as _m
        mean = sum(vals) / max(1, len(vals))
        std = (_m.fsum((v - mean) ** 2 for v in vals) / max(1, len(vals))) ** 0.5
        print(f"  {kind:>4}: n={len(vals):3d} mean={mean:.4g} std={std:.4g}")


def main():
    args = parse_args()
    results = analyze_model(args)
    print_summary(results)
    save_results(results, args)


if __name__ == '__main__':
    main()

