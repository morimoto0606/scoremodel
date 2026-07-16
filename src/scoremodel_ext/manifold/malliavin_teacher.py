"""Backend for discrete Malliavin--Skorokhod score teachers.

The implementation deliberately depends only on an endpoint map

    Z -> F(Z),

where ``Z`` is a finite collection of independent standard Gaussian noise
variables and ``F(Z)`` is the endpoint of a differentiable SDE integrator.
It therefore does not assume that a Euclidean endpoint score can be lifted to
the manifold.  The same routine can be used with a base-manifold endpoint
``F=X_t`` or a horizontal frame-bundle endpoint ``F=U_t``.

This is an exact finite-dimensional Gaussian integration-by-parts
calculation.  It approximates the continuous Malliavin weight only through
the chosen time discretisation of the SDE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


Tensor = torch.Tensor
EndpointMap = Callable[[Tensor], Tensor]
TargetFields = Callable[[Tensor], Tensor]
FieldDivergence = Callable[[Tensor], Tensor]


@dataclass
class DiscreteMalliavinTeacher:
    """All quantities produced for one discretised diffusion path.

    Attributes
    ----------
    endpoint:
        The endpoint ``F(Z)``, flattened to ambient/local coordinates.
    endpoint_jacobian:
        Jacobian ``dF/dZ`` with shape ``[endpoint_dim, noise_dim]``.
    covariance:
        Discrete Malliavin covariance ``J J^T``.
    covariance_eigenvalues:
        Eigenvalues of the symmetrised covariance, useful for diagnostics.
    covering:
        Minimum-energy regularised controls, shape
        ``[noise_dim, n_fields]``.
    skorokhod:
        Directional Skorokhod integrals ``delta(u^a)``.
    directional_score_weight:
        Per-path weights whose conditional expectations are the directional
        score components ``V_a log p``.
    score_weight:
        Reconstructed endpoint-coordinate tangent score weight.  Its
        conditional expectation is the score when the target fields span the
        endpoint tangent space.
    """

    endpoint: Tensor
    endpoint_jacobian: Tensor
    covariance: Tensor
    covariance_eigenvalues: Tensor
    covering: Tensor
    skorokhod: Tensor
    directional_score_weight: Tensor
    score_weight: Tensor


def _jacobian(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    *,
    create_graph: bool,
    vectorize: bool,
) -> Tensor:
    """Compatibility wrapper around ``torch.autograd.functional.jacobian``."""

    return torch.autograd.functional.jacobian(
        function,
        value,
        create_graph=create_graph,
        strict=False,
        vectorize=vectorize,
    )


def discrete_malliavin_skorokhod_teacher(
    endpoint_fn: EndpointMap,
    standard_noise: Tensor,
    target_fields_fn: TargetFields,
    *,
    field_divergence_fn: Optional[FieldDivergence] = None,
    covariance_regularization: float = 1e-6,
    reconstruction_rtol: float = 1e-7,
    vectorize_jacobian: bool = True,
) -> DiscreteMalliavinTeacher:
    r"""Compute a discrete Malliavin--Skorokhod teacher for one path.

    Let ``Z ~ N(0, I_q)`` and let ``F(Z)`` be an endpoint in an ambient or
    local coordinate space of dimension ``m``.  For endpoint vector fields
    ``V=[V_1,...,V_k]`` this function computes

    .. math::

        J = \partial_Z F,\qquad C=JJ^\top,\qquad
        U=J^\top(C+\lambda I)^{-1}V.

    The finite-dimensional Skorokhod divergence is

    .. math::

        \delta(U_a)=U_a^\top Z-\operatorname{div}_Z U_a.

    With density defined relative to the reference manifold volume,

    .. math::

        V_a\log p(F)
        =-\mathbb E[\delta(U_a)\mid F]-\operatorname{div}V_a(F).

    Parameters
    ----------
    endpoint_fn:
        Differentiable map accepting ``standard_noise`` with its original
        shape and returning one endpoint tensor.
    standard_noise:
        Independent *standard* Gaussian variables.  The endpoint integrator
        is responsible for multiplying them by ``sqrt(dt)`` and the diffusion
        coefficient.
    target_fields_fn:
        Returns a matrix of endpoint vector fields with shape
        ``[endpoint_dim, n_fields]``.  For an embedded sphere this can be the
        projected ambient basis ``I - x x^T``.
    field_divergence_fn:
        Returns ``[n_fields]`` containing divergences with respect to the
        reference manifold volume.  Omit only for divergence-free fields.
    covariance_regularization:
        Tikhonov regularisation used when solving the covariance system.
    reconstruction_rtol:
        Relative tolerance for reconstructing a tangent vector from possibly
        redundant directional components.
    vectorize_jacobian:
        Passed to PyTorch's Jacobian implementation.  Disable on a backend
        for which vmap coverage is incomplete.
    """

    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")
    if not standard_noise.is_floating_point():
        raise TypeError("standard_noise must be a floating-point tensor")

    noise_shape = standard_noise.shape
    noise = standard_noise.detach().clone().requires_grad_(True)

    def flat_endpoint(z: Tensor) -> Tensor:
        return endpoint_fn(z.reshape(noise_shape)).reshape(-1)

    def covering_from_noise(z: Tensor) -> Tensor:
        endpoint = flat_endpoint(z)
        jacobian = _jacobian(
            flat_endpoint,
            z,
            create_graph=True,
            vectorize=vectorize_jacobian,
        ).reshape(endpoint.numel(), z.numel())
        fields = target_fields_fn(endpoint).reshape(endpoint.numel(), -1)
        covariance = jacobian @ jacobian.transpose(0, 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eye = torch.eye(
            covariance.shape[0],
            dtype=covariance.dtype,
            device=covariance.device,
        )
        coefficients = torch.linalg.solve(
            covariance + covariance_regularization * eye,
            fields,
        )
        return jacobian.transpose(0, 1) @ coefficients

    z_flat = noise.reshape(-1)
    endpoint = flat_endpoint(z_flat)
    endpoint_jacobian = _jacobian(
        flat_endpoint,
        z_flat,
        create_graph=True,
        vectorize=vectorize_jacobian,
    ).reshape(endpoint.numel(), z_flat.numel())
    covariance = endpoint_jacobian @ endpoint_jacobian.transpose(0, 1)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    fields = target_fields_fn(endpoint).reshape(endpoint.numel(), -1)
    covering = covering_from_noise(z_flat)

    covering_jacobian = _jacobian(
        covering_from_noise,
        z_flat,
        create_graph=False,
        vectorize=vectorize_jacobian,
    ).reshape(z_flat.numel(), fields.shape[1], z_flat.numel())
    diagonal_indices = torch.arange(z_flat.numel(), device=z_flat.device)
    covering_divergence = covering_jacobian[
        diagonal_indices, :, diagonal_indices
    ].sum(dim=0)

    gaussian_pairing = covering.transpose(0, 1) @ z_flat
    skorokhod = gaussian_pairing - covering_divergence

    if field_divergence_fn is None:
        field_divergence = torch.zeros_like(skorokhod)
    else:
        field_divergence = field_divergence_fn(endpoint).reshape(-1)
        if field_divergence.shape != skorokhod.shape:
            raise ValueError(
                "field_divergence_fn returned shape "
                f"{tuple(field_divergence.shape)}, expected {tuple(skorokhod.shape)}"
            )

    directional_score_weight = -skorokhod - field_divergence

    # directional_score_weight[a] = <score_weight, V_a>.  The pseudoinverse
    # handles redundant generating fields, e.g. the three projected ambient
    # fields spanning the two-dimensional tangent space of S^2.
    score_weight = torch.linalg.pinv(
        fields.transpose(0, 1), rtol=reconstruction_rtol
    ) @ directional_score_weight

    return DiscreteMalliavinTeacher(
        endpoint=endpoint,
        endpoint_jacobian=endpoint_jacobian,
        covariance=covariance,
        covariance_eigenvalues=torch.linalg.eigvalsh(covariance),
        covering=covering,
        skorokhod=skorokhod,
        directional_score_weight=directional_score_weight,
        score_weight=score_weight,
    )
