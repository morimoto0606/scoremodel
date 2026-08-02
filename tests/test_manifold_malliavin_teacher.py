"""Tests for the generic discrete Malliavin--Skorokhod backend.

These tests are intended to run on the GPU server.  The exact Skorokhod
divergence differentiates through a Jacobian and can be slow on a laptop.
"""

import math
import numpy as np

import torch

from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    build_s2_reference_score_functions,
    compare_s2_reverse_generators,
    generate_s2_fixed_start_marginal_teacher_dataset,
    generate_s2_mixture_marginal_teacher_dataset,
    generate_s2_teacher_dataset,
    summarize_s2_score_comparison,
    train_s2_score_model,
    train_s2_marginal_score,
)
from scoremodel_ext.manifold.earthquake_adapter import (
    S2TeacherProvider,
    evaluate_s2_score_model,
    nearest_neighbor_geodesic_summary,
    s2_rbf_mmd,
)
from scoremodel_ext.manifold import earthquake_adapter as earthquake_adapter_module
from scripts import reproduce_earthquake_s2_malliavin as earthquake_script
from scoremodel_ext.malliavin.models import MirafzaliSkorokhodNet
from scoremodel_ext.manifold.malliavin_teacher import (
    discrete_malliavin_skorokhod_teacher,
    tangent_malliavin_skorokhod_teacher,
)
from scoremodel_ext.manifold.s2_malliavin import (
    S2SkorokhodScoreModel,
    s2_discrete_malliavin_teacher,
    s2_grw_endpoint,
    s2_heat_kernel_score,
    s2_projector,
    s2_reconstruct_score_vector,
    s2_reverse_grw,
    s2_tangent_basis,
    s2_tangent_malliavin_teacher,
    s2_varadhan_score,
)


DTYPE = torch.float64


def _centered_difference_jacobian(function, value, *, step=1e-6):
    value = value.clone()
    base = function(value)
    jacobian = torch.empty(
        base.numel(),
        value.numel(),
        dtype=base.dtype,
        device=base.device,
    )
    for index in range(value.numel()):
        perturbation = torch.zeros_like(value)
        perturbation[index] = step
        forward = function(value + perturbation)
        backward = function(value - perturbation)
        jacobian[:, index] = ((forward - backward) / (2.0 * step)).reshape(-1)
    return jacobian


def _centered_difference_trace(vector_field, value, *, step=1e-6):
    value = value.clone()
    sample = vector_field(value)
    if sample.ndim == 1:
        sample = sample[:, None]
    trace = torch.zeros(sample.shape[1], dtype=sample.dtype, device=sample.device)
    for index in range(value.numel()):
        perturbation = torch.zeros_like(value)
        perturbation[index] = step
        forward = vector_field(value + perturbation)
        backward = vector_field(value - perturbation)
        if forward.ndim == 1:
            forward = forward[:, None]
            backward = backward[:, None]
        partial = (forward - backward) / (2.0 * step)
        trace = trace + partial[index]
    return trace


def _phase1_s2_case(*, covariance_regularization=1e-8, field_index=None):
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    teacher = s2_tangent_malliavin_teacher(
        x0,
        noise,
        terminal_time=0.2,
        field_index=field_index,
        covariance_regularization=covariance_regularization,
    )
    return x0, noise, teacher


def test_mirafzali_network_supports_distinct_input_and_teacher_dimensions():
    network = MirafzaliSkorokhodNet(
        x_dim=5,
        out_dim=2,
        hidden=16,
        n_blocks=1,
        num_frequencies=4,
    )
    output = network(torch.rand(3), torch.rand(3, 5))
    assert output.shape == (3, 2)


def test_euclidean_additive_noise_recovers_exact_path_weight():
    """For X_t=x0+sqrt(t)Z the score weight is exactly -Z/sqrt(t)."""

    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    time = 0.6
    identity = torch.eye(2, dtype=DTYPE)

    teacher = discrete_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(time) * noise,
        z,
        lambda endpoint: identity,
        field_divergence_fn=lambda endpoint: torch.zeros(2, dtype=DTYPE),
        covariance_regularization=1e-12,
    )

    expected = -z / math.sqrt(time)
    torch.testing.assert_close(teacher.skorokhod, -expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(teacher.score_weight, expected, rtol=1e-8, atol=1e-8)


def test_s2_grw_endpoint_remains_on_sphere():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3], [0.5, 0.2, -0.6]],
        dtype=DTYPE,
    )
    endpoint = s2_grw_endpoint(x0, noise, terminal_time=0.2)
    torch.testing.assert_close(
        torch.linalg.vector_norm(endpoint),
        torch.ones((), dtype=DTYPE),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tangent_malliavin_teacher_matches_simple_additive_noise():
    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    time = 0.6
    tangent_basis = torch.eye(2, dtype=DTYPE)

    teacher = tangent_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(time) * noise,
        z,
        lambda endpoint: tangent_basis,
        lambda endpoint: tangent_basis,
        field_divergence_fn=lambda endpoint: torch.zeros(2, dtype=DTYPE),
        covariance_regularization=1e-12,
    )

    expected = -z / math.sqrt(time)
    torch.testing.assert_close(teacher.skorokhod, -expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        teacher.covering,
        torch.eye(2, dtype=DTYPE) / math.sqrt(time),
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(teacher.score_weight, expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(teacher.gaussian_pairing, z / math.sqrt(time), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(teacher.divergence_term, torch.zeros(2, dtype=DTYPE), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        teacher.right_inverse_residual,
        torch.zeros(2, dtype=DTYPE),
        rtol=1e-7,
        atol=1e-7,
    )


def test_tangent_malliavin_teacher_handles_redundant_fields():
    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    time = 0.6
    tangent_basis = torch.eye(2, dtype=DTYPE)

    teacher = tangent_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(time) * noise,
        z,
        lambda endpoint: tangent_basis,
        lambda endpoint: torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=DTYPE
        ),
        field_divergence_fn=lambda endpoint: torch.zeros(3, dtype=DTYPE),
        covariance_regularization=1e-12,
    )

    expected = -z / math.sqrt(time)
    torch.testing.assert_close(teacher.score_weight, expected, rtol=1e-8, atol=1e-8)
    assert teacher.covering.shape == (2, 3)
    assert teacher.directional_score_weight.shape == (3,)
    assert teacher.gaussian_pairing.shape == (3,)
    assert teacher.divergence_term.shape == (3,)
    assert teacher.right_inverse_residual.shape == (3,)


def test_s2_tangent_basis_and_reconstruction_are_tangent():
    endpoint = torch.tensor([0.2, -0.3, 0.9327379053], dtype=DTYPE)
    endpoint = endpoint / torch.linalg.vector_norm(endpoint)
    basis = s2_tangent_basis(endpoint)
    directional_scores = torch.tensor([0.4, -0.1, 0.3], dtype=DTYPE)

    reconstructed = s2_reconstruct_score_vector(directional_scores, endpoint)
    expected = torch.linalg.pinv(s2_projector(endpoint).transpose(0, 1), rtol=1e-7) @ directional_scores

    assert basis.shape == (3, 2)
    assert abs(float(torch.dot(reconstructed, endpoint))) < 1e-10
    torch.testing.assert_close(
        reconstructed,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_s2_endpoint_jacobian_matches_centered_finite_difference():
    x0, noise, teacher = _phase1_s2_case(covariance_regularization=1e-8)

    def endpoint_fn(flat_noise):
        return s2_grw_endpoint(x0, flat_noise.reshape_as(noise), terminal_time=0.2)

    finite_difference = _centered_difference_jacobian(
        endpoint_fn,
        noise.reshape(-1),
        step=1e-6,
    )
    torch.testing.assert_close(
        teacher.endpoint_jacobian,
        finite_difference,
        rtol=2e-5,
        atol=2e-6,
    )


def test_s2_tangent_covering_relative_residual_is_small():
    _, _, teacher = _phase1_s2_case(covariance_regularization=1e-8)
    basis = s2_tangent_basis(teacher.endpoint)
    projected_fields = s2_projector(teacher.endpoint)
    tangent_fields = basis.transpose(0, 1) @ projected_fields
    tangent_jacobian = basis.transpose(0, 1) @ teacher.endpoint_jacobian

    residual = torch.linalg.vector_norm(
        tangent_jacobian @ teacher.covering - tangent_fields,
        dim=0,
    )
    scale = torch.clamp(torch.linalg.vector_norm(tangent_fields, dim=0), min=1e-12)
    relative_residual = residual / scale
    torch.testing.assert_close(
        relative_residual,
        teacher.right_inverse_residual / scale,
        rtol=1e-8,
        atol=1e-10,
    )
    assert float(relative_residual.max()) < 5e-5


def test_s2_exact_divergence_matches_centered_difference_trace():
    x0, noise, teacher = _phase1_s2_case(covariance_regularization=1e-8)

    def covering_from_noise(flat_noise):
        reshaped_noise = flat_noise.reshape_as(noise)
        endpoint = s2_grw_endpoint(x0, reshaped_noise, terminal_time=0.2)
        basis = s2_tangent_basis(endpoint)
        tangent_jacobian = basis.transpose(0, 1) @ torch.autograd.functional.jacobian(
            lambda local_flat_noise: s2_grw_endpoint(
                x0,
                local_flat_noise.reshape_as(noise),
                terminal_time=0.2,
            ).reshape(-1),
            flat_noise,
            create_graph=False,
            strict=False,
            vectorize=True,
        ).reshape(3, flat_noise.numel())
        tangent_fields = basis.transpose(0, 1) @ s2_projector(endpoint)
        covariance = tangent_jacobian @ tangent_jacobian.transpose(0, 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eye = torch.eye(2, dtype=DTYPE)
        coefficients = torch.linalg.solve(covariance + 1e-8 * eye, tangent_fields)
        return tangent_jacobian.transpose(0, 1) @ coefficients

    finite_difference_trace = _centered_difference_trace(
        covering_from_noise,
        noise.reshape(-1),
        step=1e-6,
    )
    torch.testing.assert_close(
        teacher.divergence_term,
        finite_difference_trace,
        rtol=2e-4,
        atol=2e-5,
    )


def test_s2_tangent_teacher_is_tangent_and_covariance_has_two_tangent_modes():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    teacher = s2_tangent_malliavin_teacher(
        x0,
        noise,
        terminal_time=0.2,
        field_index=0,
        covariance_regularization=1e-8,
    )

    tangent_residual = torch.dot(teacher.endpoint, teacher.score_weight)
    assert abs(float(tangent_residual)) < 1e-7
    assert teacher.covariance_eigenvalues.shape == (2,)
    assert teacher.covariance_eigenvalues[0] > 1e-6
    assert teacher.covariance_eigenvalues[1] > 1e-6


def test_s2_tangent_teacher_supports_all_projected_ambient_fields():
    _, _, teacher = _phase1_s2_case(covariance_regularization=1e-8)

    tangent_residual = torch.dot(teacher.endpoint, teacher.score_weight)
    assert abs(float(tangent_residual)) < 1e-7
    assert teacher.directional_score_weight.shape == (3,)
    assert teacher.gaussian_pairing.shape == (3,)
    assert teacher.divergence_term.shape == (3,)
    assert teacher.right_inverse_residual.shape == (3,)
    assert teacher.covering.shape[1] == 3


def test_s2_regularization_sweep_reports_stable_diagnostics():
    regularizations = [1e-10, 1e-8, 1e-6, 1e-4]
    diagnostics = []
    for regularization in regularizations:
        _, _, teacher = _phase1_s2_case(covariance_regularization=regularization)
        diagnostics.append(
            {
                "regularization": regularization,
                "condition_number": float(teacher.condition_number),
                "max_right_inverse_residual": float(teacher.right_inverse_residual.max()),
            }
        )

    baseline_condition_number = diagnostics[0]["condition_number"]
    for diagnostic in diagnostics:
        assert math.isfinite(diagnostic["condition_number"]), diagnostics
        assert math.isfinite(diagnostic["max_right_inverse_residual"]), diagnostics
        assert diagnostic["max_right_inverse_residual"] < 5e-3, diagnostics
        assert abs(diagnostic["condition_number"] - baseline_condition_number) < 1e-8, diagnostics


def test_s2_single_field_api_remains_backward_compatible():
    _, _, single_field_teacher = _phase1_s2_case(
        covariance_regularization=1e-8,
        field_index=0,
    )
    _, _, all_fields_teacher = _phase1_s2_case(covariance_regularization=1e-8)

    torch.testing.assert_close(
        single_field_teacher.covering.reshape(-1),
        all_fields_teacher.covering[:, 0],
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        single_field_teacher.gaussian_pairing.reshape(-1),
        all_fields_teacher.gaussian_pairing[:1],
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        single_field_teacher.divergence_term.reshape(-1),
        all_fields_teacher.divergence_term[:1],
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        single_field_teacher.skorokhod.reshape(-1),
        all_fields_teacher.skorokhod[:1],
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        single_field_teacher.directional_score_weight.reshape(-1),
        all_fields_teacher.directional_score_weight[:1],
        rtol=1e-8,
        atol=1e-8,
    )


def test_s2_teacher_is_tangent_and_covariance_has_two_tangent_modes():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    noise = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    teacher = s2_discrete_malliavin_teacher(
        x0,
        noise,
        terminal_time=0.2,
        covariance_regularization=1e-8,
    )

    tangent_residual = torch.dot(teacher.endpoint, teacher.score_weight)
    assert abs(float(tangent_residual)) < 1e-7
    assert teacher.covariance_eigenvalues.shape == (3,)
    assert teacher.covariance_eigenvalues[-1] > 1e-6
    assert teacher.covariance_eigenvalues[-2] > 1e-6
    assert abs(float(teacher.covariance_eigenvalues[0])) < 1e-7


def test_s2_heat_kernel_and_varadhan_scores_are_tangent():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    endpoint = torch.tensor([0.2, -0.3, 0.9327379053], dtype=DTYPE)
    endpoint = endpoint / torch.linalg.vector_norm(endpoint)

    exact = s2_heat_kernel_score(x0, endpoint, terminal_time=0.3, n_terms=60)
    asymptotic = s2_varadhan_score(x0, endpoint, terminal_time=0.3)
    projector = s2_projector(endpoint)

    torch.testing.assert_close(projector @ exact, exact, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        projector @ asymptotic, asymptotic, rtol=1e-9, atol=1e-9
    )


def test_s2_heat_kernel_score_points_towards_initial_point():
    x0 = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    endpoint = torch.tensor([0.4, 0.0, math.sqrt(0.84)], dtype=DTYPE)
    score = s2_heat_kernel_score(x0, endpoint, terminal_time=0.4, n_terms=60)
    direction_to_x0 = s2_projector(endpoint) @ x0
    assert torch.dot(score, direction_to_x0) > 0


def test_s2_skorokhod_wrapper_projects_network_output():
    class ConstantDelta(torch.nn.Module):
        def forward(self, t, x):
            return torch.tensor([1.0, -2.0, 0.5], dtype=x.dtype).expand_as(x)

    model = S2SkorokhodScoreModel(ConstantDelta())
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
    )
    score = model(torch.tensor([0.2, 0.2], dtype=DTYPE), points)
    assert torch.max(torch.abs((score * points).sum(dim=1))) < 1e-12


def test_reverse_grw_keeps_all_samples_on_sphere():
    terminal = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE
    )
    noise = torch.zeros(3, 2, 3, dtype=DTYPE)
    output = s2_reverse_grw(
        terminal,
        lambda t, x: torch.zeros_like(x),
        terminal_time=0.3,
        n_steps=3,
        standard_noise=noise,
    )
    torch.testing.assert_close(output, terminal)


def test_s2_score_comparison_reports_heat_varadhan_and_geodesic_bins():
    dataset = generate_s2_teacher_dataset(
        n_paths=12,
        n_steps=2,
        terminal_time=0.2,
        covariance_regularization=1e-8,
        device="cpu",
        dtype=DTYPE,
        seed=0,
        vectorize_jacobian=True,
    )

    metrics = summarize_s2_score_comparison(
        dataset,
        n_heat_terms=40,
        knn_k=3,
        geodesic_bin_edges=[0.0, 0.2, 0.4, 0.8],
    )

    assert math.isfinite(metrics["malliavin_vs_heat_rmse"])
    assert math.isfinite(metrics["malliavin_vs_heat_mean_cosine"])
    assert math.isfinite(metrics["varadhan_vs_heat_rmse"])
    assert math.isfinite(metrics["varadhan_vs_heat_mean_cosine"])
    assert math.isfinite(metrics["malliavin_vs_varadhan_rmse"])
    assert math.isfinite(metrics["malliavin_vs_varadhan_mean_cosine"])
    assert len(metrics["geodesic_bins"]) >= 3
    assert sum(entry["count"] for entry in metrics["geodesic_bins"]) == 12


def test_fixed_start_marginal_dataset_keeps_initial_point_and_varies_time():
    dataset = generate_s2_fixed_start_marginal_teacher_dataset(
        n_paths=6,
        n_steps=2,
        minimum_time=0.05,
        maximum_time=0.2,
        covariance_regularization=1e-8,
        device="cpu",
        dtype=DTYPE,
        seed=1,
        vectorize_jacobian=True,
    )

    initial_points = dataset["initial_point"]
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand_as(initial_points)
    torch.testing.assert_close(initial_points, reference, rtol=1e-12, atol=1e-12)
    assert bool((dataset["time"] >= 0.05).all())
    assert bool((dataset["time"] <= 0.2).all())
    assert float(dataset["time"].std()) > 0.0


def test_s2_reverse_comparison_accepts_reference_and_learned_scores():
    fixed_start_dataset = generate_s2_fixed_start_marginal_teacher_dataset(
        n_paths=8,
        n_steps=2,
        minimum_time=0.05,
        maximum_time=0.2,
        covariance_regularization=1e-8,
        device="cpu",
        dtype=DTYPE,
        seed=2,
        vectorize_jacobian=True,
    )
    trained_score = train_s2_marginal_score(
        fixed_start_dataset,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden=16,
        n_blocks=1,
        num_frequencies=4,
        device="cpu",
    )
    terminal_points = fixed_start_dataset["endpoint"]
    initial_point = fixed_start_dataset["initial_point"][0]
    score_functions = build_s2_reference_score_functions(initial_point, n_heat_terms=30)
    score_functions["trained"] = trained_score

    comparison = compare_s2_reverse_generators(
        terminal_points,
        score_functions,
        initial_point=initial_point,
        terminal_time=0.2,
        n_steps=4,
        seed=0,
    )

    generated = comparison["generated_samples"]
    metrics = comparison["metrics"]
    assert set(generated) == {"heat", "varadhan", "trained"}
    for sample in generated.values():
        norms = torch.linalg.vector_norm(sample, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-10, atol=1e-10)
    for method_metrics in metrics["by_method"].values():
        assert math.isfinite(method_metrics["mean_geodesic_distance"])
        assert math.isfinite(method_metrics["rmse_geodesic_distance"])
        assert math.isfinite(method_metrics["max_geodesic_distance"])
        assert math.isfinite(method_metrics["max_norm_error"])
    assert "heat" in metrics["pairwise_mean_geodesic_distance"]
    assert "varadhan" in metrics["pairwise_mean_geodesic_distance"]["heat"]


def test_s2_mixture_marginal_dataset_samples_from_requested_components():
    components = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE)
    dataset = generate_s2_mixture_marginal_teacher_dataset(
        components,
        weights,
        n_paths=10,
        n_steps=2,
        minimum_time=0.05,
        maximum_time=0.2,
        covariance_regularization=1e-8,
        seed=3,
        vectorize_jacobian=True,
    )

    assert dataset["component_index"].shape == (10,)
    reconstructed = components[dataset["component_index"]]
    torch.testing.assert_close(dataset["initial_point"], reconstructed, rtol=1e-12, atol=1e-12)
    assert dataset["score_target"].shape == (10, 3)


def test_s2_teacher_provider_switches_teachers_with_tangent_targets():
    initial_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    provider = S2TeacherProvider(
        initial_points,
        n_steps=2,
        covariance_regularization=1e-8,
        n_heat_terms=30,
        vectorize_jacobian=True,
    )

    for teacher in ("malliavin", "heat", "varadhan"):
        batch = provider.sample_batch(
            6,
            teacher=teacher,
            minimum_time=0.05,
            maximum_time=0.2,
            seed=4,
        )
        assert batch.score_target.shape == (6, 3)
        tangent_residual = (batch.endpoint * batch.score_target).sum(dim=1).abs()
        assert float(tangent_residual.max()) < 1e-6
        if teacher == "malliavin":
            assert batch.directional_score_target is not None
            assert batch.skorokhod is not None
        else:
            assert batch.directional_score_target is None
            assert batch.skorokhod is None


def test_s2_direct_score_model_trains_on_provider_dataset():
    provider = S2TeacherProvider(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        n_steps=2,
        covariance_regularization=1e-8,
        n_heat_terms=30,
        vectorize_jacobian=True,
    )
    dataset = provider.sample_dataset(
        8,
        teacher="malliavin",
        minimum_time=0.05,
        maximum_time=0.2,
        seed=5,
    )
    model = train_s2_score_model(
        dataset,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden=16,
        n_blocks=1,
        num_frequencies=4,
        device="cpu",
    )
    prediction = model(dataset["time"][:2], dataset["endpoint"][:2])
    assert prediction.shape == (2, 3)
    assert torch.isfinite(prediction).all()


def test_s2_adapter_metrics_are_finite_on_simple_inputs():
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    mmd = s2_rbf_mmd(points, points, sigma=0.5, n_sub=3, seed=0)
    geodesic = nearest_neighbor_geodesic_summary(points, points, n_sub=3, seed=0)

    assert math.isfinite(mmd)
    assert math.isfinite(geodesic["mean"])
    assert geodesic["mean"] < 1e-8
    assert geodesic["median"] < 1e-8
    assert geodesic["max"] < 1e-8


def test_s2_score_model_evaluation_returns_finite_mse():
    provider = S2TeacherProvider(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        n_steps=2,
        covariance_regularization=1e-8,
        n_heat_terms=30,
        vectorize_jacobian=True,
    )
    dataset = provider.sample_dataset(
        6,
        teacher="heat",
        minimum_time=0.05,
        maximum_time=0.2,
        seed=6,
    )
    model = train_s2_score_model(
        dataset,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden=16,
        n_blocks=1,
        num_frequencies=4,
        device="cpu",
    )
    mse = evaluate_s2_score_model(model, dataset)
    assert math.isfinite(mse)
    assert mse >= 0.0


def test_train_validation_split_indices_are_non_empty_and_disjoint():
    rng = np.random.default_rng(0)
    train_index, validation_index = earthquake_script._compute_train_validation_indices(
        10,
        0.2,
        rng=rng,
    )
    assert len(train_index) > 0
    assert len(validation_index) > 0
    assert len(set(train_index.tolist()).intersection(set(validation_index.tolist()))) == 0


def test_train_validation_split_rejects_invalid_fraction_bounds():
    rng = np.random.default_rng(0)
    for fraction in (0.0, 1.0, -0.1, 1.1):
        try:
            earthquake_script._compute_train_validation_indices(10, fraction, rng=rng)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_discrete_teacher_endpoint_jacobian_not_attached_to_graph():
    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    teacher = discrete_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(0.6) * noise,
        z,
        lambda endpoint: torch.eye(2, dtype=DTYPE),
        field_divergence_fn=lambda endpoint: torch.zeros(2, dtype=DTYPE),
        covariance_regularization=1e-12,
    )
    assert teacher.endpoint_jacobian.requires_grad is False


def test_tangent_teacher_endpoint_jacobian_not_attached_to_graph():
    x0 = torch.tensor([0.4, -0.7], dtype=DTYPE)
    z = torch.tensor([0.3, -1.2], dtype=DTYPE)
    teacher = tangent_malliavin_skorokhod_teacher(
        lambda noise: x0 + math.sqrt(0.6) * noise,
        z,
        lambda endpoint: torch.eye(2, dtype=DTYPE),
        lambda endpoint: torch.eye(2, dtype=DTYPE),
        field_divergence_fn=lambda endpoint: torch.zeros(2, dtype=DTYPE),
        covariance_regularization=1e-12,
    )
    assert teacher.endpoint_jacobian.requires_grad is False


def test_malliavin_provider_uses_teacher_endpoint_in_malliavin_branch():
    provider = S2TeacherProvider(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        n_steps=2,
        covariance_regularization=1e-8,
        n_heat_terms=30,
        vectorize_jacobian=True,
    )

    original = earthquake_adapter_module.s2_grw_endpoint
    try:
        earthquake_adapter_module.s2_grw_endpoint = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("s2_grw_endpoint should not be called for malliavin branch"))
        batch = provider.sample_batch(
            2,
            teacher="malliavin",
            minimum_time=0.1,
            maximum_time=0.1,
            seed=7,
        )
    finally:
        earthquake_adapter_module.s2_grw_endpoint = original

    assert batch.endpoint.shape == (2, 3)
