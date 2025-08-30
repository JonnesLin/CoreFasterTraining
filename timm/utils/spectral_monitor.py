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
        interval: int = 1,
        # logging controls
        log_cos_per_index: bool = True,
        log_lastk: bool = True,
        log_sv_list: bool = True,
        log_cos_summaries: bool = True,
        log_delta: bool = True,
    ) -> None:
        self.topk = topk
        self.every = every
        self.device = device
        self.use_cpu = use_cpu
        self.interval = max(1, int(interval))
        self.log_cos_per_index = log_cos_per_index
        self.log_lastk = log_lastk
        self.log_sv_list = log_sv_list
        self.log_cos_summaries = log_cos_summaries
        self.log_delta = log_delta
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
            r = S.shape[0]
            # effective k considering interval and available rank
            k = min(t.k, (r + self.interval - 1) // self.interval)
            # indices for top-k with spacing (0-based indices)
            if k > 0:
                top_idx = torch.arange(0, k * self.interval, step=self.interval, device=U.device)
                top_idx = top_idx[top_idx < r]
            else:
                top_idx = torch.tensor([], dtype=torch.long, device=U.device)
            # indices for last-k with spacing (from end)
            if k > 0:
                last_start = r - 1
                last_idx = torch.arange(last_start, last_start - k * self.interval - 1, step=-self.interval, device=U.device)
                last_idx = last_idx[last_idx >= 0][:k]
            else:
                last_idx = torch.tensor([], dtype=torch.long, device=U.device)

            # top-k (spaced)
            Uk = U.index_select(1, top_idx) if top_idx.numel() else U[:, :0]
            # Vh is (r, n), take rows by top_idx then transpose to (n, k)
            Vk = Vh.index_select(0, top_idx).transpose(0, 1) if top_idx.numel() else Vh[:0, :].transpose(0, 1)
            Sk = S.index_select(0, top_idx) if top_idx.numel() else S[:0]
            # last-k (spaced from tail)
            Ul = U.index_select(1, last_idx) if last_idx.numel() else U[:, :0]
            Vl = Vh.index_select(0, last_idx).transpose(0, 1) if last_idx.numel() else Vh[:0, :].transpose(0, 1)
            Sl = S.index_select(0, last_idx) if last_idx.numel() else S[:0]

            # values
            base = f"spec/{t.name}"
            if k > 0:
                metrics[f"{base}/sigma_max"] = Sk[0].item()
            else:
                metrics[f"{base}/sigma_max"] = float("nan")
            if self.log_sv_list:
                for i in range(k):
                    metrics[f"{base}/sv{i+1}"] = Sk[i].item()
                for i in range(k):
                    # last-k singular values with spacing
                    metrics[f"{base}/sv_last{i+1}"] = Sl[i].item() if i < Sl.numel() else float('nan')

            # relative change of singular values if prev available
            if self.log_delta:
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
                if self.log_cos_per_index:
                    for i in range(min(k, u_cos.shape[0])):
                        metrics[f"{base}/cos_u{i+1}"] = float(u_cos[i].item())
                # summary for convenience
                if self.log_cos_summaries:
                    metrics[f"{base}/cos_u_max"] = float(u_cos.max().item()) if u_cos.numel() else float('nan')
                    metrics[f"{base}/cos_u_mean"] = float(u_cos.mean().item()) if u_cos.numel() else float('nan')
            else:
                if self.log_cos_per_index:
                    for i in range(k):
                        metrics[f"{base}/cos_u{i+1}"] = float("nan")
                if self.log_cos_summaries:
                    metrics[f"{base}/cos_u_max"] = float("nan")
                    metrics[f"{base}/cos_u_mean"] = float("nan")

            if t.prev_V_top is not None and t.prev_V_top.shape[0] == Vk.shape[0]:
                v_cos = _principal_cosines_all(t.prev_V_top[:, :k], Vk)
                if self.log_cos_per_index:
                    for i in range(min(k, v_cos.shape[0])):
                        metrics[f"{base}/cos_v{i+1}"] = float(v_cos[i].item())
                if self.log_cos_summaries:
                    metrics[f"{base}/cos_v_max"] = float(v_cos.max().item()) if v_cos.numel() else float('nan')
                    metrics[f"{base}/cos_v_mean"] = float(v_cos.mean().item()) if v_cos.numel() else float('nan')
            else:
                if self.log_cos_per_index:
                    for i in range(k):
                        metrics[f"{base}/cos_v{i+1}"] = float("nan")
                if self.log_cos_summaries:
                    metrics[f"{base}/cos_v_max"] = float("nan")
                    metrics[f"{base}/cos_v_mean"] = float("nan")

            # principal cosines for last-k subspaces
            if self.log_lastk:
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
            if self.log_lastk:
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
