import torch

from scripts.plot_earthquake_teacher_scatter_comparison import (
    load_saved_scatter_artifacts,
)


def test_scatter_postprocess_loads_skip_viz_artifacts(tmp_path):
    run_dirs = {}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin")):
        run_dir = tmp_path / teacher
        run_dir.mkdir()
        run_dirs[teacher] = run_dir
        torch.save(
            torch.full((4, 3), float(index), dtype=torch.float64),
            run_dir / "generated_samples.pt",
        )

    train = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    validation = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    torch.save(
        {
            "train_initial_points": train,
            "validation_initial_points": validation,
        },
        run_dirs["heat"] / "teacher_initial_points.pt",
    )

    observed, generated = load_saved_scatter_artifacts(run_dirs)

    torch.testing.assert_close(observed, torch.cat((train, validation)), rtol=0, atol=0)
    assert generated.keys() == {"heat", "varadhan", "malliavin"}
    for index, teacher in enumerate(("heat", "varadhan", "malliavin")):
        torch.testing.assert_close(
            generated[teacher],
            torch.full((4, 3), float(index), dtype=torch.float64),
            rtol=0,
            atol=0,
        )
