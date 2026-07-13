from eyetracker.heatmap import GazeHeatmap


def test_add_increments_correct_cell():
    heatmap = GazeHeatmap(grid_size=(10, 10), sigma=0.0)
    heatmap.add(0.0, 0.0)
    assert heatmap.raw()[0, 0] == 1.0


def test_add_clips_out_of_range_coordinates():
    heatmap = GazeHeatmap(grid_size=(10, 10), sigma=0.0)
    heatmap.add(-5.0, 5.0)
    assert heatmap.raw()[:, 0].sum() == 1.0


def test_normalized_peak_is_one():
    heatmap = GazeHeatmap(grid_size=(8, 8), sigma=1.0)
    for _ in range(5):
        heatmap.add(0.5, 0.5)
    assert heatmap.normalized().max() == 1.0


def test_normalized_empty_heatmap_has_no_nans():
    heatmap = GazeHeatmap(grid_size=(8, 8), sigma=1.0)
    result = heatmap.normalized()
    assert not (result != result).any()  # no NaNs
