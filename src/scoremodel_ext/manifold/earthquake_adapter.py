"""Adapters for generating S2 teacher targets from empirical initial samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import torch

from .s2_malliavin import (
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_varadhan_score,
)


Tensor = torch.Tensor
TeacherKind = Literal["malliavin", "heat", "varadhan"]


@dataclass
class S2TeacherBatch:
    initial_point: Tensor
    time: Tensor
    noise: Tensor
    endpoint: Tensor
    score_target: Tensor
    directional_score_target: Optional[Tensor] = None
    skorokhod: Optional[Tensor] = None

    def as_training_dict(self) -> Dict[str, Tensor]:
        dataset = {
            "initial_point": self.initial_point,
            "time": self.time,
            "noise": self.noise,
            "endpoint": self.endpoint,
            "score_target": self.score_target,
        }
        if self.directional_score_target is not None:
            dataset["directional_score_weight"] = self.directional_score_target
        if self.skorokhod is not None:
            dataset["skorokhod"] = self.skorokhod
            dataset["score_weight"] = self.score_target
        return dataset


def load_earthquake_points(
    csv_path: Path,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
) -> Tensor:
    """Load earthquake latitude/longitude data and embed it on S2."""

    latlon_deg = np.genfromtxt(csv_path, delimiter=",", skip_header=4)
    lat = np.deg2rad(latlon_deg[:, 0])
    lon = np.deg2rad(latlon_deg[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    points = np.stack([x, y, z], axis=1)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return torch.tensor(points, dtype=dtype, device=device)


def evaluate_s2_score_model(model, dataset: Dict[str, Tensor]) -> float:
    """Return mean squared error of a direct score model on one dataset."""

    with torch.no_grad():
        prediction = model(dataset["time"], dataset["endpoint"])
    return float(torch.mean((prediction - dataset["score_target"]) ** 2))


def s2_rbf_mmd(
    samples: Tensor,
    reference: Tensor,
    *,
    sigma: float = 1.0,
    n_sub: int = 2000,
    seed: int = 0,
) -> float:
    """Unbiased RBF MMD on S2 using ambient chordal distance."""

    samples_np = samples.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    sample_index = rng.choice(len(samples_np), min(n_sub, len(samples_np)), replace=False)
    reference_index = rng.choice(len(reference_np), min(n_sub, len(reference_np)), replace=False)
    sample_subset = samples_np[sample_index]
    reference_subset = reference_np[reference_index]

    def gram(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance_sq = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-distance_sq / (2.0 * sigma ** 2))

    k_xx = gram(sample_subset, sample_subset)
    k_yy = gram(reference_subset, reference_subset)
    k_xy = gram(sample_subset, reference_subset)
    n_x = len(sample_subset)
    n_y = len(reference_subset)
    return float(
        (k_xx.sum() - np.trace(k_xx)) / (n_x * max(n_x - 1, 1))
        + (k_yy.sum() - np.trace(k_yy)) / (n_y * max(n_y - 1, 1))
        - 2.0 * k_xy.mean()
    )


def nearest_neighbor_geodesic_summary(
    samples: Tensor,
    reference: Tensor,
    *,
    n_sub: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Nearest-neighbor geodesic summary from samples to a reference cloud on S2."""

    sample_generator = torch.Generator(device=samples.device)
    sample_generator.manual_seed(seed)
    reference_generator = torch.Generator(device=reference.device)
    reference_generator.manual_seed(seed + 1)
    if samples.shape[0] > n_sub:
        sample_index = torch.randperm(samples.shape[0], generator=sample_generator, device=samples.device)[:n_sub]
        samples = samples[sample_index]
    if reference.shape[0] > n_sub:
        reference_index = torch.randperm(reference.shape[0], generator=reference_generator, device=reference.device)[:n_sub]
        reference = reference[reference_index]

    normalized_samples = samples / torch.linalg.vector_norm(samples, dim=1, keepdim=True)
    normalized_reference = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    cosine = torch.clamp(normalized_samples @ normalized_reference.transpose(0, 1), -1.0, 1.0)
    distances = torch.arccos(cosine)
    nearest = distances.min(dim=1).values
    return {
        "mean": float(nearest.mean()),
        "median": float(nearest.median()),
        "max": float(nearest.max()),
    }


class S2TeacherProvider:
    """Sample training triples ``(time, endpoint, score_target)`` on S2."""

    def __init__(
        self,
        initial_points: Tensor,
        *,
        n_steps: int,
        covariance_regularization: float = 1e-6,
        n_heat_terms: int = 80,
        vectorize_jacobian: bool = True,
    ) -> None:
        if initial_points.ndim != 2 or initial_points.shape[1] != 3:
            raise ValueError("initial_points must have shape [n_samples, 3]")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        normalized = initial_points / torch.linalg.vector_norm(
            initial_points,
            dim=1,
            keepdim=True,
        )
        self.initial_points = normalized
        self.n_steps = n_steps
        self.covariance_regularization = covariance_regularization
        self.n_heat_terms = n_heat_terms
        self.vectorize_jacobian = vectorize_jacobian

    @classmethod
    def from_earthquake_csv(
        cls,
        csv_path: Path,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
        n_steps: int,
        covariance_regularization: float = 1e-6,
        n_heat_terms: int = 80,
        vectorize_jacobian: bool = True,
    ) -> "S2TeacherProvider":
        return cls(
            load_earthquake_points(csv_path, dtype=dtype, device=device),
            n_steps=n_steps,
            covariance_regularization=covariance_regularization,
            n_heat_terms=n_heat_terms,
            vectorize_jacobian=vectorize_jacobian,
        )

    def sample_batch(
        self,
        batch_size: int,
        *,
        teacher: TeacherKind,
        minimum_time: float,
        maximum_time: float,
        seed: int,
    ) -> S2TeacherBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if minimum_time <= 0 or maximum_time <= 0:
            raise ValueError("minimum_time and maximum_time must be positive")
        if minimum_time > maximum_time:
            raise ValueError("minimum_time must be <= maximum_time")
        if teacher not in {"malliavin", "heat", "varadhan"}:
            raise ValueError("teacher must be one of {'malliavin', 'heat', 'varadhan'}")

        generator = torch.Generator(device=self.initial_points.device)
        generator.manual_seed(seed)
        sample_index = torch.randint(
            0,
            self.initial_points.shape[0],
            (batch_size,),
            generator=generator,
            device=self.initial_points.device,
        )
        initial_points = self.initial_points[sample_index]
        terminal_times = torch.empty(
            batch_size,
            dtype=self.initial_points.dtype,
            device=self.initial_points.device,
        ).uniform_(minimum_time, maximum_time, generator=generator)

        noises = []
        endpoints = []
        score_targets = []
        directional_targets = []
        skorokhod_targets = []

        for initial_point, terminal_time in zip(initial_points, terminal_times):
            noise = torch.randn(
                self.n_steps,
                3,
                dtype=self.initial_points.dtype,
                device=self.initial_points.device,
                generator=generator,
            )
            terminal_time_float = float(terminal_time.detach().cpu())
            noises.append(noise)

            if teacher == "malliavin":
                teacher_state = s2_discrete_malliavin_teacher(
                    initial_point,
                    noise,
                    terminal_time_float,
                    covariance_regularization=self.covariance_regularization,
                    vectorize_jacobian=self.vectorize_jacobian,
                )
                endpoints.append(teacher_state.endpoint)
                score_targets.append(teacher_state.score_weight)
                directional_targets.append(teacher_state.directional_score_weight)
                skorokhod_targets.append(teacher_state.skorokhod)
            elif teacher == "heat":
                endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
                endpoints.append(endpoint)
                score_targets.append(
                    s2_heat_kernel_score(
                        initial_point,
                        endpoint,
                        terminal_time_float,
                        n_terms=self.n_heat_terms,
                    )
                )
            else:
                endpoint = s2_grw_endpoint(initial_point, noise, terminal_time_float)
                endpoints.append(endpoint)
                score_targets.append(s2_varadhan_score(initial_point, endpoint, terminal_time_float))

        directional_score_target = None
        skorokhod = None
        if directional_targets:
            directional_score_target = torch.stack(directional_targets)
            skorokhod = torch.stack(skorokhod_targets)
        return S2TeacherBatch(
            initial_point=initial_points,
            time=terminal_times,
            noise=torch.stack(noises),
            endpoint=torch.stack(endpoints),
            score_target=torch.stack(score_targets),
            directional_score_target=directional_score_target,
            skorokhod=skorokhod,
        )

    def sample_dataset(
        self,
        n_paths: int,
        *,
        teacher: TeacherKind,
        minimum_time: float,
        maximum_time: float,
        seed: int,
    ) -> Dict[str, Tensor]:
        return self.sample_batch(
            n_paths,
            teacher=teacher,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
            seed=seed,
        ).as_training_dict()


MalliavinScoreProvider = S2TeacherProvider