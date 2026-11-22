from types import SimpleNamespace
import numpy as np
from bisindo_app.camera import draw_landmarks


def test_draw_landmarks_preserves_shape_and_does_not_mutate_source():
    src = np.zeros((240, 320, 3), dtype=np.uint8)
    pts = [SimpleNamespace(x=.1 + .035 * (i % 5), y=.2 + .035 * (i // 5)) for i in range(21)]
    result = draw_landmarks(src, [pts])
    assert result.shape == src.shape
    assert np.count_nonzero(result) > 0
    assert np.count_nonzero(src) == 0


def test_draw_landmarks_no_hands_returns_same_pixels():
    src = np.ones((10, 10, 3), dtype=np.uint8)
    assert np.array_equal(draw_landmarks(src, []), src)
