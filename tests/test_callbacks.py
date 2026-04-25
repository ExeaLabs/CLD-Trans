from engine.callbacks import EarlyStopping


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(mode="min", patience=1, min_delta=0.0)

    improved, should_stop = stopper.update(1.0)
    assert improved is True
    assert should_stop is False

    improved, should_stop = stopper.update(1.0)
    assert improved is False
    assert should_stop is False

    improved, should_stop = stopper.update(1.0)
    assert improved is False
    assert should_stop is True