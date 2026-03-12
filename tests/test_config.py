"""Tests for configuration loading and defaults."""

import os
from unittest.mock import patch


class TestConfigDefaults:

    def test_retrieval_k_default(self):
        with patch.dict(os.environ, {}, clear=False):
            import importlib
            import config
            importlib.reload(config)
            assert config.RETRIEVAL_K == int(os.getenv("RETRIEVAL_K", "4"))

    def test_max_loop_steps_default(self):
        import config
        assert config.MAX_LOOP_STEPS >= 1

    def test_tavily_max_results_default(self):
        import config
        assert config.TAVILY_MAX_RESULTS >= 1

    def test_grader_temperature_is_float(self):
        import config
        assert isinstance(config.GRADER_TEMPERATURE, float)

    def test_generator_temperature_is_float(self):
        import config
        assert isinstance(config.GENERATOR_TEMPERATURE, float)

    def test_rewriter_temperature_is_float(self):
        import config
        assert isinstance(config.REWRITER_TEMPERATURE, float)


class TestConfigEnvOverride:

    def test_retrieval_k_overridden_by_env(self):
        with patch.dict(os.environ, {"RETRIEVAL_K": "10"}):
            import importlib
            import config
            importlib.reload(config)
            assert config.RETRIEVAL_K == 10

    def test_max_loop_steps_overridden_by_env(self):
        with patch.dict(os.environ, {"MAX_LOOP_STEPS": "5"}):
            import importlib
            import config
            importlib.reload(config)
            assert config.MAX_LOOP_STEPS == 5

    def test_grader_temperature_overridden_by_env(self):
        with patch.dict(os.environ, {"GRADER_TEMPERATURE": "0.5"}):
            import importlib
            import config
            importlib.reload(config)
            assert config.GRADER_TEMPERATURE == 0.5
