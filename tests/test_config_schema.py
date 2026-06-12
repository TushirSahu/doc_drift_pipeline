import copy

import pytest

from src.core.schema import ConfigError, Settings, validate_config
from src.core.settings import get_config


def test_real_config_is_valid():
    settings = validate_config()
    assert isinstance(settings, Settings)
    assert settings.retrieval.top_k > 0
    assert 0.0 <= settings.retrieval.mmr_lambda <= 1.0


def test_bad_top_k_rejected():
    raw = copy.deepcopy(get_config())
    raw["retrieval"]["top_k"] = "five"
    with pytest.raises(ConfigError):
        validate_config(raw)


def test_out_of_range_lambda_rejected():
    raw = copy.deepcopy(get_config())
    raw["retrieval"]["mmr_lambda"] = 2.0
    with pytest.raises(ConfigError):
        validate_config(raw)


def test_bad_chunking_strategy_rejected():
    raw = copy.deepcopy(get_config())
    raw["chunking"]["strategy"] = "banana"
    with pytest.raises(ConfigError):
        validate_config(raw)


def test_missing_section_rejected():
    raw = copy.deepcopy(get_config())
    del raw["models"]
    with pytest.raises(ConfigError):
        validate_config(raw)
