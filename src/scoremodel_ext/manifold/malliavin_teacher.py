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
    condition_number:
        Spectral condition number of the symmetrised covariance.
    covering:
        Minimum-energy regularised controls, shape
        ``[noise_dim, n_fields]``.
    right_inverse_residual:
        Residual norms for ``J U = V`` after regularised inversion.
    gaussian_pairing:
        Gaussian pairing term ``U^T Z`` for each field.
    divergence_term:
        Divergence term ``div_Z U`` for each field.
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
    condition_number: Tensor
    covering: Tensor
    right_inverse_residual: Tensor
    gaussian_pairing: Tensor
    divergence_term: Tensor
    skorokhod: Tensor
    directional_score_weight: Tensor
    score_weight: Tensor


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(0, 1))


def _condition_number_from_eigenvalues(eigenvalues: Tensor) -> Tensor:
    absolute = eigenvalues.abs()
    largest = absolute.max()
    smallest = absolute.min()
    if bool((smallest <= 0).detach().cpu()):
        return torch.full_like(largest, float("inf"))
    return largest / smallest


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


def tangent_malliavin_skorokhod_teacher(
    endpoint_fn: EndpointMap,
    standard_noise: Tensor,
    tangent_basis_fn: Callable[[Tensor], Tensor],
    target_fields_fn: Callable[[Tensor], Tensor],
    *,
    field_divergence_fn: Optional[FieldDivergence] = None,
    covariance_regularization: float = 1e-6,
    reconstruction_rtol: float = 1e-7,
    vectorize_jacobian: bool = True,
) -> DiscreteMalliavinTeacher:
    r"""Compute a tangent-space discrete Malliavin teacher.

    The endpoint map ``F(Z)`` is assumed to live in a manifold embedded in
    Euclidean space.  The tangent basis and endpoint vector fields are
    projected onto the tangent space before the finite-dimensional Malliavin
    covariance is inverted.
    """

    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")
    if not standard_noise.is_floating_point():
        raise TypeError("standard_noise must be a floating-point tensor")

    noise_shape = standard_noise.shape
    noise = standard_noise.detach().clone().requires_grad_(True)

    def flat_endpoint(z: Tensor) -> Tensor:
        return endpoint_fn(z.reshape(noise_shape)).reshape(-1)

    def _prepare_tangent_state(z: Tensor, *, create_graph: bool):
        endpoint = flat_endpoint(z)
        tangent_basis = tangent_basis_fn(endpoint)
        if tangent_basis.ndim != 2:
            raise ValueError("tangent_basis_fn must return a matrix")
        if tangent_basis.shape[0] != endpoint.numel():
            raise ValueError(
                "tangent_basis_fn returned shape "
                f"{tuple(tangent_basis.shape)}, expected first dimension {endpoint.numel()}"
            )
        ambient_fields = target_fields_fn(endpoint).reshape(endpoint.numel(), -1)
        tangent_fields = tangent_basis.transpose(0, 1) @ ambient_fields
        endpoint_jacobian = _jacobian(
            flat_endpoint,
            z,
            create_graph=create_graph,
            vectorize=vectorize_jacobian,
        ).reshape(endpoint.numel(), z.numel())
        tangent_jacobian = tangent_basis.transpose(0, 1) @ endpoint_jacobian
        return endpoint, tangent_basis, tangent_fields, endpoint_jacobian, tangent_jacobian

    def _covering_from_tangent_state(tangent_jacobian: Tensor, tangent_fields: Tensor) -> Tensor:
        covariance = tangent_jacobian @ tangent_jacobian.transpose(0, 1)
        covariance = _symmetrize(covariance)
        eye = torch.eye(
            covariance.shape[0],
            dtype=covariance.dtype,
            device=covariance.device,
        )
        coefficients = torch.linalg.solve(
            covariance + covariance_regularization * eye,
            tangent_fields,
        )
        return tangent_jacobian.transpose(0, 1) @ coefficients

    def covering_from_noise(z: Tensor) -> Tensor:
        _, _, tangent_fields, _, tangent_jacobian = _prepare_tangent_state(
            z,
            create_graph=True,
        )
        return _covering_from_tangent_state(tangent_jacobian, tangent_fields)

    z_flat = noise.reshape(-1)
    (
        endpoint,
        tangent_basis,
        tangent_fields,
        endpoint_jacobian,
        tangent_jacobian,
    ) = _prepare_tangent_state(z_flat, create_graph=False)
    covariance = tangent_jacobian @ tangent_jacobian.transpose(0, 1)
    covariance = _symmetrize(covariance)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    covering = _covering_from_tangent_state(tangent_jacobian, tangent_fields)

    covering_jacobian = _jacobian(
        covering_from_noise,
        z_flat,
        create_graph=False,
        vectorize=vectorize_jacobian,
    ).reshape(z_flat.numel(), tangent_fields.shape[1], z_flat.numel())
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
    tangent_score_weight = torch.linalg.pinv(
        tangent_fields.transpose(0, 1), rtol=reconstruction_rtol
    ) @ directional_score_weight
    score_weight = tangent_basis @ tangent_score_weight
    right_inverse_residual = torch.linalg.vector_norm(
        tangent_jacobian @ covering - tangent_fields,
        dim=0,
    )

    return DiscreteMalliavinTeacher(
        endpoint=endpoint,
        endpoint_jacobian=endpoint_jacobian,
        covariance=covariance,
        covariance_eigenvalues=covariance_eigenvalues,
        condition_number=_condition_number_from_eigenvalues(covariance_eigenvalues),
        covering=covering,
        right_inverse_residual=right_inverse_residual,
        gaussian_pairing=gaussian_pairing,
        divergence_term=covering_divergence,
        skorokhod=skorokhod,
        directional_score_weight=directional_score_weight,
        score_weight=score_weight,
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

    def _prepare_endpoint_state(z: Tensor, *, create_graph: bool):
        endpoint = flat_endpoint(z)
        jacobian = _jacobian(
            flat_endpoint,
            z,
            create_graph=create_graph,
            vectorize=vectorize_jacobian,
        ).reshape(endpoint.numel(), z.numel())
        fields = target_fields_fn(endpoint).reshape(endpoint.numel(), -1)
        return endpoint, jacobian, fields

    def _covering_from_endpoint_state(jacobian: Tensor, fields: Tensor) -> Tensor:
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

    def covering_from_noise(z: Tensor) -> Tensor:
        _, jacobian, fields = _prepare_endpoint_state(z, create_graph=True)
        return _covering_from_endpoint_state(jacobian, fields)

    z_flat = noise.reshape(-1)
    endpoint, endpoint_jacobian, fields = _prepare_endpoint_state(
        z_flat,
        create_graph=False,
    )
    covariance = endpoint_jacobian @ endpoint_jacobian.transpose(0, 1)
    covariance = _symmetrize(covariance)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    covering = _covering_from_endpoint_state(endpoint_jacobian, fields)

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
    right_inverse_residual = torch.linalg.vector_norm(
        endpoint_jacobian @ covering - fields,
        dim=0,
    )

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
        covariance_eigenvalues=covariance_eigenvalues,
        condition_number=_condition_number_from_eigenvalues(covariance_eigenvalues),
        covering=covering,
        right_inverse_residual=right_inverse_residual,
        gaussian_pairing=gaussian_pairing,
        divergence_term=covering_divergence,
        skorokhod=skorokhod,
        directional_score_weight=directional_score_weight,
        score_weight=score_weight,
    )


def _pure_single_malliavin_skorokhod_teacher(
    endpoint_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    initial_point: Tensor,
    standard_noise: Tensor,
    terminal_time: Tensor,
    target_fields_fn: TargetFields,
    *,
    field_divergence_fn: Optional[FieldDivergence],
    covariance_regularization: float,
    reconstruction_rtol: float,
) -> tuple[Tensor, ...]:
    """Pure ``torch.func`` form of the exact single-sample teacher.

    This intentionally mirrors :func:`discrete_malliavin_skorokhod_teacher`.
    It returns tensors rather than a dataclass so that ``torch.func.vmap`` can
    add the sample dimension without registering a custom pytree.
    """

    noise_shape = standard_noise.shape
    z_flat = standard_noise.reshape(-1)

    def flat_endpoint(z: Tensor) -> Tensor:
        return endpoint_fn(
            initial_point,
            z.reshape(noise_shape),
            terminal_time,
        ).reshape(-1)

    def prepare_endpoint_state(z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        endpoint = flat_endpoint(z)
        jacobian = torch.func.jacrev(flat_endpoint)(z).reshape(
            endpoint.numel(), z.numel()
        )
        fields = target_fields_fn(endpoint).reshape(endpoint.numel(), -1)
        return endpoint, jacobian, fields

    def covering_from_endpoint_state(jacobian: Tensor, fields: Tensor) -> Tensor:
        covariance = _symmetrize(jacobian @ jacobian.transpose(0, 1))
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

    def covering_from_noise(z: Tensor) -> Tensor:
        _, jacobian, fields = prepare_endpoint_state(z)
        return covering_from_endpoint_state(jacobian, fields)

    endpoint, endpoint_jacobian, fields = prepare_endpoint_state(z_flat)
    covariance = _symmetrize(endpoint_jacobian @ endpoint_jacobian.transpose(0, 1))
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    covering = covering_from_endpoint_state(endpoint_jacobian, fields)

    covering_jacobian = torch.func.jacrev(covering_from_noise)(z_flat).reshape(
        z_flat.numel(), fields.shape[1], z_flat.numel()
    )
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
    directional_score_weight = -skorokhod - field_divergence
    right_inverse_residual = torch.linalg.vector_norm(
        endpoint_jacobian @ covering - fields,
        dim=0,
    )
    score_weight = torch.linalg.pinv(
        fields.transpose(0, 1), rtol=reconstruction_rtol
    ) @ directional_score_weight

    absolute_eigenvalues = covariance_eigenvalues.abs()
    largest = absolute_eigenvalues.max()
    smallest = absolute_eigenvalues.min()
    condition_number = torch.where(
        smallest <= 0,
        torch.full_like(largest, torch.inf),
        largest / smallest,
    )
    return (
        endpoint,
        endpoint_jacobian,
        covariance,
        covariance_eigenvalues,
        condition_number,
        covering,
        right_inverse_residual,
        gaussian_pairing,
        covering_divergence,
        skorokhod,
        directional_score_weight,
        score_weight,
    )


def batched_discrete_malliavin_skorokhod_teacher(
    endpoint_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    initial_points: Tensor,
    standard_noises: Tensor,
    terminal_times: Tensor,
    target_fields_fn: TargetFields,
    *,
    field_divergence_fn: Optional[FieldDivergence] = None,
    covariance_regularization: float = 1e-6,
    reconstruction_rtol: float = 1e-7,
) -> DiscreteMalliavinTeacher:
    """Exact sample-batched teacher built from ``vmap(jacrev(...))``.

    No random variables are generated here.  The leading dimensions of
    ``initial_points``, ``standard_noises``, and ``terminal_times`` identify
    the same ordered samples returned in the output dataclass.
    """

    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")
    if initial_points.ndim != 2:
        raise ValueError("initial_points must have shape [batch, endpoint_dim]")
    if standard_noises.ndim < 2:
        raise ValueError("standard_noises must have a leading batch dimension")
    batch_size = initial_points.shape[0]
    if standard_noises.shape[0] != batch_size:
        raise ValueError("standard_noises batch dimension does not match")
    if terminal_times.shape != (batch_size,):
        raise ValueError("terminal_times must have shape [batch]")
    if not standard_noises.is_floating_point():
        raise TypeError("standard_noises must be floating point")
    if (
        initial_points.dtype != standard_noises.dtype
        or terminal_times.dtype != standard_noises.dtype
    ):
        raise ValueError("batched teacher inputs must have one dtype")

    def single_teacher(
        initial_point: Tensor,
        standard_noise: Tensor,
        terminal_time: Tensor,
    ) -> tuple[Tensor, ...]:
        return _pure_single_malliavin_skorokhod_teacher(
            endpoint_fn,
            initial_point,
            standard_noise,
            terminal_time,
            target_fields_fn,
            field_divergence_fn=field_divergence_fn,
            covariance_regularization=covariance_regularization,
            reconstruction_rtol=reconstruction_rtol,
        )

    outputs = torch.func.vmap(single_teacher)(
        initial_points,
        standard_noises,
        terminal_times,
    )
    return DiscreteMalliavinTeacher(
        endpoint=outputs[0],
        endpoint_jacobian=outputs[1],
        covariance=outputs[2],
        covariance_eigenvalues=outputs[3],
        condition_number=outputs[4],
        covering=outputs[5],
        right_inverse_residual=outputs[6],
        gaussian_pairing=outputs[7],
        divergence_term=outputs[8],
        skorokhod=outputs[9],
        directional_score_weight=outputs[10],
        score_weight=outputs[11],
    )
