from __future__ import annotations

import numpy as np
import pytest


def test_decode_lstm_model_direct_multi_horizon_when_torch_is_installed(tmp_path) -> None:
    pytest.importorskip("torch")
    from campus_forecast.models.energy import DecodeLSTMRegressor

    features = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    targets = np.ones((4, 3), dtype=np.float32)
    model = DecodeLSTMRegressor(
        epochs=2,
        patience=1,
        batch_size=2,
        checkpoint_path=tmp_path / "decode.pt",
    )
    model.fit(features, targets, validation=(features[:2], targets[:2]))
    prediction = model.predict(features[:1])
    assert prediction.shape == (1, 3)
    assert (tmp_path / "decode.pt").exists()
