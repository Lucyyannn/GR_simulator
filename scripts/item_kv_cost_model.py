#!/usr/bin/env python3
"""Explicit implementation of the paper's storage-aware item-KV cost model.

Only effective hardware rates appear as calibrated quantities.  All byte,
FLOP, Vector-op, and HBM-to-core movement counts are computed directly from
the equations in the paper and are never multiplied by empirical workload
correction factors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SaturatingHardwarePath:
    """One hardware path with saturation and per-request startup overhead.

    ``peak_rate`` is the configured rate after its hardware-only rate factor.
    A positive ``saturation_work`` models the clean, monotone approach to that
    rate as one request grows.  Zero disables saturation and recovers the
    original paper denominator exactly.
    """

    peak_rate: float
    saturation_work: float = 0.0
    startup_s: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_rate <= 0.0:
            raise ValueError("peak_rate must be positive")
        if self.saturation_work < 0.0:
            raise ValueError("saturation_work must be non-negative")
        if self.startup_s < 0.0:
            raise ValueError("startup_s must be non-negative")

    def achieved_rate(self, work_per_request: float) -> float:
        """R_eff(x) = R_peak * (1 - exp(-x/x_sat))."""

        if self.saturation_work == 0.0:
            return self.peak_rate
        utilization = -math.expm1(-work_per_request / self.saturation_work)
        return self.peak_rate * max(utilization, 1e-15)

    def time_s(self, total_work: float, request_count: int) -> tuple[float, float, float]:
        """Return total, transfer/compute, and startup time for this path."""

        if total_work <= 0.0 or request_count <= 0:
            return 0.0, 0.0, 0.0
        work_per_request = total_work / request_count
        body_s = total_work / self.achieved_rate(work_per_request)
        startup_s = request_count * self.startup_s
        return body_s + startup_s, body_s, startup_s


@dataclass(frozen=True)
class ItemKVCostWorkload:
    """Workload symbols in the paper, extended with an independent-user batch."""

    original_history_tokens: int  # S
    item_kv_after_akr: int  # S_i
    action_kv_after_akr: int  # S_a
    candidates_per_user: int  # C
    hidden: int  # H
    layers: int  # L
    bytes_per_element: float  # s
    batch_size: int = 1  # N, independent users in one batch
    ar_reduces_attention_compute: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "original_history_tokens": self.original_history_tokens,
            "item_kv_after_akr": self.item_kv_after_akr,
            "action_kv_after_akr": self.action_kv_after_akr,
            "candidates_per_user": self.candidates_per_user,
            "hidden": self.hidden,
            "layers": self.layers,
            "batch_size": self.batch_size,
        }
        for name, value in integer_fields.items():
            if value < 0 or (name in {"hidden", "layers", "batch_size"} and value == 0):
                raise ValueError(f"{name} must be positive where required, got {value}")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")

    @property
    def total_kv_after_akr(self) -> int:
        """S_h = S_i + S_a."""

        return self.item_kv_after_akr + self.action_kv_after_akr

    @property
    def attention_history_tokens(self) -> int:
        """S_att: logical history rows charged by QK/AV attention."""

        if self.ar_reduces_attention_compute:
            return self.total_kv_after_akr
        return self.original_history_tokens


@dataclass(frozen=True)
class ItemKVHardwareRates:
    """Raw hardware inputs and hardware-only achieved/reference-rate factors.

    The effective denominator used by each paper equation is ``rate * eta``.
    An eta describes an achieved/reference rate for a chip, storage tier, and
    batch/concurrency point.  It may exceed one when the supplied raw rate is a
    nominal reference rather than a theoretical peak, but it must not depend
    on model dimensions, sequence length, recompute ratio, or a numerator.
    """

    b_kv: float  # B_kv, bytes/s
    b_emb: float  # B_emb, bytes/s
    b_core: float  # B_core, bytes/s
    f_cube: float  # F_cube, FLOP/s
    f_vec: float  # F_vec, Vector operations/s
    eta_kv: float = 1.0
    eta_emb: float = 1.0
    eta_core: float = 1.0
    eta_cube: float = 1.0
    eta_vec: float = 1.0
    kv_saturation_bytes: float = 0.0
    emb_saturation_bytes: float = 0.0
    core_saturation_bytes: float = 0.0
    cube_saturation_flops: float = 0.0
    vec_saturation_ops: float = 0.0
    kv_startup_s: float = 0.0
    emb_startup_s: float = 0.0
    core_startup_s: float = 0.0
    cube_startup_s: float = 0.0
    vec_startup_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("b_kv", "b_emb", "b_core", "f_cube", "f_vec"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("eta_kv", "eta_emb", "eta_core", "eta_cube", "eta_vec"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "kv_saturation_bytes", "emb_saturation_bytes",
            "core_saturation_bytes", "cube_saturation_flops",
            "vec_saturation_ops", "kv_startup_s", "emb_startup_s",
            "core_startup_s", "cube_startup_s", "vec_startup_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def effective(self) -> dict[str, float]:
        return {
            "B_kv": self.b_kv * self.eta_kv,
            "B_emb": self.b_emb * self.eta_emb,
            "B_core": self.b_core * self.eta_core,
            "F_cube": self.f_cube * self.eta_cube,
            "F_vec": self.f_vec * self.eta_vec,
        }

    @property
    def paths(self) -> dict[str, SaturatingHardwarePath]:
        """The five clean hardware paths used by the paper equations."""

        effective = self.effective
        return {
            "kv": SaturatingHardwarePath(
                effective["B_kv"], self.kv_saturation_bytes, self.kv_startup_s
            ),
            "emb": SaturatingHardwarePath(
                effective["B_emb"], self.emb_saturation_bytes, self.emb_startup_s
            ),
            "core": SaturatingHardwarePath(
                effective["B_core"], self.core_saturation_bytes, self.core_startup_s
            ),
            "cube": SaturatingHardwarePath(
                effective["F_cube"], self.cube_saturation_flops, self.cube_startup_s
            ),
            "vec": SaturatingHardwarePath(
                effective["F_vec"], self.vec_saturation_ops, self.vec_startup_s
            ),
        }


def item_kv_cost_terms(
    k: int,
    workload: ItemKVCostWorkload,
    hardware: ItemKVHardwareRates,
) -> dict[str, float | int]:
    """Evaluate the paper equations for one integer recompute count ``k``.

    ``N`` extends the single-user equations to a batch of independent users.
    It multiplies each user's theoretical work; importantly, the causal term
    is ``N * k^2`` rather than ``(N*k)^2``, so users never attend across one
    another.
    """

    if not 0 <= k <= workload.item_kv_after_akr:
        raise ValueError(
            f"k={k} is outside [0, {workload.item_kv_after_akr}]"
        )

    Si = workload.item_kv_after_akr
    Sa = workload.action_kv_after_akr
    Sh = workload.total_kv_after_akr
    Satt = workload.attention_history_tokens
    C = workload.candidates_per_user
    H = workload.hidden
    L = workload.layers
    s = workload.bytes_per_element
    N = workload.batch_size
    paths = hardware.paths
    layer_user_requests = L * N

    # Paper Eq. (KV memory):
    #   T_mem(k) = 2 L N [S_a + (S_i-k)] H s / B_kv
    #            + N k H s / B_emb.
    kv_bytes = 2.0 * L * N * (Sa + (Si - k)) * H * s
    embedding_bytes = N * k * H * s
    kv_time_s, kv_body_s, kv_startup_s = paths["kv"].time_s(
        kv_bytes, layer_user_requests
    )
    embedding_time_s, embedding_body_s, embedding_startup_s = paths["emb"].time_s(
        embedding_bytes, N
    )
    t_mem_s = kv_time_s + embedding_time_s

    # Paper Eq. (Cube):
    #   T_cube(k) = L N [8 C H^2 + 4 C S_att H
    #                    + 8 k H^2 + 4 k^2 H]
    #               / F_cube.
    # With AR attention reduction enabled, S_att=S_h=S_i+S_a; otherwise
    # S_att=S. This keeps the formula visibly aligned with the trace model.
    cube_flops = L * N * (
        8.0 * C * H * H
        + 4.0 * C * Satt * H
        + 8.0 * k * H * H
        + 4.0 * k * k * H
    )
    t_cube_s, cube_body_s, cube_startup_s = paths["cube"].time_s(
        cube_flops, layer_user_requests
    )

    # Paper Eq. (Vector):
    #   T_vec(k) = L N [2(C S_att+k^2) + 2(C+k)H] / F_vec.
    vector_ops = L * N * (
        2.0 * (C * Satt + k * k)
        + 2.0 * (C + k) * H
    )
    t_vec_s, vec_body_s, vec_startup_s = paths["vec"].time_s(
        vector_ops, layer_user_requests
    )

    # Paper Eq. (HBM-to-core movement):
    #   T_core(k) = L N s [(C+k)H + 2 S_h H + 4 k H] / B_core.
    core_bytes = L * N * s * (
        (C + k) * H
        + 2.0 * Sh * H
        + 4.0 * k * H
    )
    t_core_s, core_body_s, core_startup_s = paths["core"].time_s(
        core_bytes, layer_user_requests
    )

    # Paper critical path and decision objective.
    t_npu_s = t_cube_s + t_vec_s + t_core_s
    t_s = max(t_mem_s, t_npu_s)

    return {
        "k": k,
        "recompute_ratio": k / Si if Si else 0.0,
        "attention_history_tokens": Satt,
        "kv_bytes": kv_bytes,
        "embedding_bytes": embedding_bytes,
        "cube_flops": cube_flops,
        "vector_ops": vector_ops,
        "core_bytes": core_bytes,
        "kv_time_s": kv_time_s,
        "kv_body_s": kv_body_s,
        "kv_startup_s": kv_startup_s,
        "embedding_time_s": embedding_time_s,
        "embedding_body_s": embedding_body_s,
        "embedding_startup_s": embedding_startup_s,
        "cube_body_s": cube_body_s,
        "cube_startup_s": cube_startup_s,
        "vec_body_s": vec_body_s,
        "vec_startup_s": vec_startup_s,
        "core_body_s": core_body_s,
        "core_startup_s": core_startup_s,
        "T_mem_s": t_mem_s,
        "T_cube_s": t_cube_s,
        "T_vec_s": t_vec_s,
        "T_core_s": t_core_s,
        "T_npu_s": t_npu_s,
        "T_s": t_s,
    }


def select_optimal_item_recompute(
    workload: ItemKVCostWorkload,
    hardware: ItemKVHardwareRates,
) -> dict[str, float | int]:
    """Return the exact integer argmin over every ``k`` in ``[0, S_i]``.

    This is deliberately not a 0.1-ratio grid search.  With thousands of item
    rows it can select ratios such as 0.13 (represented by the nearest valid
    integer k), as required by the offline per-user decision.
    """

    best: dict[str, float | int] | None = None
    for k in range(workload.item_kv_after_akr + 1):
        candidate = item_kv_cost_terms(k, workload, hardware)
        if best is None or (candidate["T_s"], k) < (best["T_s"], best["k"]):
            best = candidate
    assert best is not None
    return best
