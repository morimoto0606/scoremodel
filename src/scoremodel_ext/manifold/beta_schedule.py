"""Tensor-safe Brownian beta schedules for the S2 experiments.

The implementations deliberately use only Python arithmetic and PyTorch tensor
operators.  They therefore preserve the input kind (``float`` or
``torch.Tensor``), device, and dtype, and are compatible with ``torch.func``
transforms such as ``vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch


TimeValue = TypeVar("TimeValue", float, torch.Tensor)


@dataclass(frozen=True)
class LegacyUnitBetaSchedule:
    """The legacy Brownian clock ``beta(t)=1`` and ``tau(t)=t``.

    Earthquake runs created before beta schedules were introduced use this
    clock.  Callers that require bitwise replay should still take their legacy
    branch rather than recomputing existing quantities through this object.
    """

    t0: float = 0.0
    tf: float = 1.0

    def beta_t(self, t: TimeValue) -> TimeValue:
        return t * 0.0 + 1.0

    def rescale_t(self, t: TimeValue) -> TimeValue:
        return t

    def interval_brownian_time(
        self,
        t_start: TimeValue,
        t_end: TimeValue,
    ) -> TimeValue:
        return t_end - t_start


@dataclass(frozen=True)
class LinearBetaSchedule:
    r"""De Bortoli linear beta schedule.

    With ``u=(t-t0)/(tf-t0)``, the upstream implementation is

    ``beta_t(t) = beta_0 + u * (beta_f-beta_0)``

    and

    ``rescale_t(t) = beta_0*u + 0.5*(beta_f-beta_0)*u**2``.

    Thus ``rescale_t`` is the normalized Brownian time

    .. math::

        \tau(t)=\frac{1}{t_f-t_0}\int_{t_0}^{t}\beta(r)\,dr.

    It equals the unnormalized physical-time integral when ``tf-t0 == 1``, as
    in the upstream Earthquake configuration.
    """

    beta_0: float = 0.001
    beta_f: float = 5.0
    t0: float = 0.0
    tf: float = 1.0

    def __post_init__(self) -> None:
        if self.tf <= self.t0:
            raise ValueError("beta schedule requires tf > t0")
        if self.beta_0 <= 0.0 or self.beta_f <= 0.0:
            raise ValueError("beta schedule endpoints must be positive")

    def _normalized_time(self, t: TimeValue) -> TimeValue:
        return (t - self.t0) / (self.tf - self.t0)

    def beta_t(self, t: TimeValue) -> TimeValue:
        u = self._normalized_time(t)
        return self.beta_0 + u * (self.beta_f - self.beta_0)

    def rescale_t(self, t: TimeValue) -> TimeValue:
        u = self._normalized_time(t)
        return self.beta_0 * u + 0.5 * (self.beta_f - self.beta_0) * u**2

    def interval_brownian_time(
        self,
        t_start: TimeValue,
        t_end: TimeValue,
    ) -> TimeValue:
        return self.rescale_t(t_end) - self.rescale_t(t_start)
