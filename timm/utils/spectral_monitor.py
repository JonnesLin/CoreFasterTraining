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
    # Guard empty or 1-d cases
    if U_prev.numel() == 0 or U_cur.numel() == 0:
        return float('nan'), float('nan')
    # Shape: (k_prev, k_cur)
    M = U_prev.transpose(-2, -1) @ U_cur
    # Only need singular values
    sv = torch.linalg.svdvals(M)
    sv = torch.clamp(sv, -1.0, 1.0)
    ang = torch.acos(sv) * (180.0 / math.pi)
    return ang.max().item(), ang.mean().item()


@dataclass
class SpectralModuleState:
    name: str
    module: nn.Module
    k: int
    prev_U: Optional[torch.Tensor] = None
    prev_V: Optional[torch.Tensor] = None
    prev_S: Optional[torch.Tensor] = None


class WeightSpectralMonitor:
    """Monitor top-k singular spectrum and subspace drift for selected linear modules.

    - Every `every` updates, computes SVD of each target module's weight `W`.
    - Logs top-k singular values, relative change vs previous step, and principal angles
      between previous and current left/right subspaces.
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
            Uk = U[:, :k]
            Vk = Vh[:k, :].transpose(0, 1)
            Sk = S[:k]

            # values
            base = f"spec/{t.name}"
            metrics[f"{base}/sigma_max"] = Sk[0].item()
            for i in range(k):
                metrics[f"{base}/sv{i+1}"] = Sk[i].item()

            # relative change of singular values if prev available
            if t.prev_S is not None and len(t.prev_S) >= k:
                prev = t.prev_S[:k]
                denom = torch.norm(prev) + 1e-12
                delta = torch.norm(Sk - prev) / denom
                metrics[f"{base}/delta_sv_rel"] = delta.item()
            else:
                metrics[f"{base}/delta_sv_rel"] = float("nan")

            # principal angles for U/V subspaces vs previous
            if t.prev_U is not None and t.prev_U.shape[0] == Uk.shape[0]:
                u_max, u_mean = _principal_angles_deg(t.prev_U[:, :k], Uk)
                metrics[f"{base}/angle_u_max_deg"] = u_max
                metrics[f"{base}/angle_u_mean_deg"] = u_mean
            else:
                metrics[f"{base}/angle_u_max_deg"] = float("nan")
                metrics[f"{base}/angle_u_mean_deg"] = float("nan")

            if t.prev_V is not None and t.prev_V.shape[0] == Vk.shape[0]:
                v_max, v_mean = _principal_angles_deg(t.prev_V[:, :k], Vk)
                metrics[f"{base}/angle_v_max_deg"] = v_max
                metrics[f"{base}/angle_v_mean_deg"] = v_mean
            else:
                metrics[f"{base}/angle_v_max_deg"] = float("nan")
                metrics[f"{base}/angle_v_mean_deg"] = float("nan")

            # update state
            t.prev_U = Uk
            t.prev_V = Vk
            t.prev_S = Sk

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

