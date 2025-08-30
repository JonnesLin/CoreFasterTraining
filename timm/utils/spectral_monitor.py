import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


def _principal_angles_deg(U_prev: torch.Tensor, U_cur: torch.Tensor) -> Tuple[float, float]:
    """Compute principal angles (max, mean) in degrees between two column-orthonormal bases.

    Both inputs are (..., d, k) with orthonormal columns. We compute SVD of U_prev^T U_cur and
    derive cosines of principal angles from singular values in [0, 1].
    """
    if U_prev.numel() == 0 or U_cur.numel() == 0:
        return float('nan'), float('nan')
    M = U_prev.transpose(-2, -1) @ U_cur
    sv = torch.linalg.svdvals(M)
    sv = torch.clamp(sv, -1.0, 1.0)
    ang = torch.acos(sv) * (180.0 / math.pi)
    return ang.max().item(), ang.mean().item()


def _principal_cosines(U_prev: torch.Tensor, U_cur: torch.Tensor) -> Tuple[float, float]:
    """Compute principal cosines (max, mean) between two column-orthonormal bases.

    Returns the maximum and mean singular values of U_prev^T U_cur, which are cosines of
    the principal angles. Values are in [0, 1].
    """
    if U_prev.numel() == 0 or U_cur.numel() == 0:
        return float('nan'), float('nan')
    M = U_prev.transpose(-2, -1) @ U_cur
    sv = torch.linalg.svdvals(M)
    sv = torch.clamp(sv, 0.0, 1.0)
    return sv.max().item(), sv.mean().item()


def _principal_cosines_all(U_prev: torch.Tensor, U_cur: torch.Tensor) -> torch.Tensor:
    """Return all principal cosines (singular values of U_prev^T U_cur), sorted desc.

    Output shape: (k,), values in [0, 1]. Assumes columns are orthonormal.
    """
    if U_prev.numel() == 0 or U_cur.numel() == 0:
        return torch.tensor([], dtype=torch.float32)
    M = U_prev.transpose(-2, -1) @ U_cur
    sv = torch.linalg.svdvals(M)
    sv = torch.clamp(sv, 0.0, 1.0)
    # torch.linalg.svdvals returns values sorted desc for CPU/GPU backends
    return sv


@dataclass
class SpectralModuleState:
    name: str
    module: nn.Module
    k: int
    # store previous top-k and last-k subspaces separately to limit memory
    prev_U_top: Optional[torch.Tensor] = None
    prev_V_top: Optional[torch.Tensor] = None
    prev_U_last: Optional[torch.Tensor] = None
    prev_V_last: Optional[torch.Tensor] = None
    prev_S_top: Optional[torch.Tensor] = None


class WeightSpectralMonitor:
    """Monitor top-k singular spectrum and subspace drift for selected linear modules.

    - Every `every` updates, computes SVD of each target module's weight `W`.
    - Logs top-k singular values, relative change vs previous step.
    - Logs subspace drift via principal cosines; per-index cosines `cos_u1..k`, `cos_v1..k` and summary max/mean.
    - Intended for rectangular weight matrices; uses SVD rather than eigen on W^T W.
    """

    def __init__(
        self,
        model: nn.Module,
        patterns: Sequence[str] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        topk: int = 8,
        every: int = 100,
        device: Optional[torch.device] = None,
        use_cpu: bool = False,
    ) -> None:
        self.topk = topk
        self.every = every
        self.device = device
        self.use_cpu = use_cpu
        self.targets: List[SpectralModuleState] = []

        # Collect target nn.Linear modules by name pattern substring match
        for name, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if any(p in name for p in patterns):
                self.targets.append(SpectralModuleState(name=name, module=m, k=topk))

    def _get_weight(self, m: nn.Linear) -> torch.Tensor:
        W = m.weight
        if W.is_sparse:
            W = W.to_dense()
        if self.use_cpu:
            return W.detach().float().cpu()
        if self.device is not None:
            return W.detach().to(self.device, dtype=torch.float32)
        return W.detach().float()

    @torch.no_grad()
    def compute_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for t in self.targets:
            W = self._get_weight(t.module)
            # SVD
            # full_matrices=False returns shapes: U (m, min(m,n)), S (min), Vh (min, n)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(t.k, S.shape[0])
            # top-k
            Uk = U[:, :k] if k > 0 else U[:, :0]
            Vk = Vh[:k, :].transpose(0, 1) if k > 0 else Vh[:0, :].transpose(0, 1)
            Sk = S[:k] if k > 0 else S[:0]
            # last-k (tail)
            Ul = U[:, -k:] if k > 0 else U[:, :0]
            Vl = Vh[-k:, :].transpose(0, 1) if k > 0 else Vh[:0, :].transpose(0, 1)
            Sl = S[-k:] if k > 0 else S[:0]

            # values
            base = f"spec/{t.name}"
            if k > 0:
                metrics[f"{base}/sigma_max"] = Sk[0].item()
            else:
                metrics[f"{base}/sigma_max"] = float("nan")
            for i in range(k):
                metrics[f"{base}/sv{i+1}"] = Sk[i].item()
                # last-k singular values: sv_last1 = smallest, increasing index = next smallest
                metrics[f"{base}/sv_last{i+1}"] = S[-(i + 1)].item()

            # relative change of singular values if prev available
            if t.prev_S_top is not None and len(t.prev_S_top) >= k:
                prev = t.prev_S_top[:k]
                denom = torch.norm(prev) + 1e-12
                delta = torch.norm(Sk - prev) / denom
                metrics[f"{base}/delta_sv_rel"] = delta.item()
            else:
                metrics[f"{base}/delta_sv_rel"] = float("nan")

            # principal angles for U/V subspaces vs previous (top-k)
            if t.prev_U_top is not None and t.prev_U_top.shape[0] == Uk.shape[0]:
                # per-index principal cosines (k values)
                u_cos = _principal_cosines_all(t.prev_U_top[:, :k], Uk)
                for i in range(min(k, u_cos.shape[0])):
                    metrics[f"{base}/cos_u{i+1}"] = float(u_cos[i].item())
                # summary for convenience
                metrics[f"{base}/cos_u_max"] = float(u_cos.max().item()) if u_cos.numel() else float('nan')
                metrics[f"{base}/cos_u_mean"] = float(u_cos.mean().item()) if u_cos.numel() else float('nan')
            else:
                for i in range(k):
                    metrics[f"{base}/cos_u{i+1}"] = float("nan")
                metrics[f"{base}/cos_u_max"] = float("nan")
                metrics[f"{base}/cos_u_mean"] = float("nan")

            if t.prev_V_top is not None and t.prev_V_top.shape[0] == Vk.shape[0]:
                v_cos = _principal_cosines_all(t.prev_V_top[:, :k], Vk)
                for i in range(min(k, v_cos.shape[0])):
                    metrics[f"{base}/cos_v{i+1}"] = float(v_cos[i].item())
                metrics[f"{base}/cos_v_max"] = float(v_cos.max().item()) if v_cos.numel() else float('nan')
                metrics[f"{base}/cos_v_mean"] = float(v_cos.mean().item()) if v_cos.numel() else float('nan')
            else:
                for i in range(k):
                    metrics[f"{base}/cos_v{i+1}"] = float("nan")
                metrics[f"{base}/cos_v_max"] = float("nan")
                metrics[f"{base}/cos_v_mean"] = float("nan")

            # principal cosines for last-k subspaces
            if t.prev_U_last is not None and t.prev_U_last.shape[0] == Ul.shape[0]:
                u_cos_last = _principal_cosines_all(t.prev_U_last[:, :k], Ul)
                for i in range(min(k, u_cos_last.shape[0])):
                    metrics[f"{base}/cos_u_last{i+1}"] = float(u_cos_last[i].item())
            else:
                for i in range(k):
                    metrics[f"{base}/cos_u_last{i+1}"] = float("nan")

            if t.prev_V_last is not None and t.prev_V_last.shape[0] == Vl.shape[0]:
                v_cos_last = _principal_cosines_all(t.prev_V_last[:, :k], Vl)
                for i in range(min(k, v_cos_last.shape[0])):
                    metrics[f"{base}/cos_v_last{i+1}"] = float(v_cos_last[i].item())
            else:
                for i in range(k):
                    metrics[f"{base}/cos_v_last{i+1}"] = float("nan")

            # update state
            t.prev_U_top = Uk
            t.prev_V_top = Vk
            t.prev_U_last = Ul
            t.prev_V_last = Vl
            t.prev_S_top = Sk

        return metrics

    def maybe_log(self, step: int, log_fn) -> Optional[Dict[str, float]]:
        """If at logging step, compute metrics and pass to `log_fn(dict)`.

        Returns metrics when executed, else None.
        """
        if self.every <= 0 or step % self.every != 0:
            return None
        metrics = self.compute_metrics()
        if metrics and log_fn is not None:
            log_fn(metrics)
        return metrics
