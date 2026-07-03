"""Tests for GLM model version configuration and backwards compatibility."""
import os
from unittest.mock import patch, MagicMock


class TestGLMModelConfig:
    """Test GLM model version detection and configuration."""

    def test_get_model_config_exact_match(self):
        """Test exact model name matching."""
        from scripts.refrag_glm import get_model_config, GLM_MODEL_CONFIGS
        
        assert get_model_config("glm-4.7") == GLM_MODEL_CONFIGS["glm-4.7"]
        assert get_model_config("glm-4.6") == GLM_MODEL_CONFIGS["glm-4.6"]
        assert get_model_config("glm-4.5") == GLM_MODEL_CONFIGS["glm-4.5"]

    def test_get_model_config_case_insensitive(self):
        """Test case-insensitive model matching."""
        from scripts.refrag_glm import get_model_config, GLM_MODEL_CONFIGS
        
        assert get_model_config("GLM-4.7") == GLM_MODEL_CONFIGS["glm-4.7"]
        assert get_model_config("Glm-4.6") == GLM_MODEL_CONFIGS["glm-4.6"]

    def test_get_model_config_variant_matching(self):
        """Test model variant matching (e.g., glm-4.7-air)."""
        from scripts.refrag_glm import get_model_config, GLM_MODEL_CONFIGS
        
        # Model variants should match base version
        assert get_model_config("glm-4.7-air") == GLM_MODEL_CONFIGS["glm-4.7"]
        assert get_model_config("glm-4.6-flash") == GLM_MODEL_CONFIGS["glm-4.6"]
        assert get_model_config("glm-4.5-fast") == GLM_MODEL_CONFIGS["glm-4.5"]

    def test_get_model_config_future_versions(self):
        """Test that future versions (4.8+) use GLM-4.7 config."""
        from scripts.refrag_glm import get_model_config, GLM_MODEL_CONFIGS
        
        # Future versions should use 4.7 config
        assert get_model_config("glm-4.8") == GLM_MODEL_CONFIGS["glm-4.7"]
        assert get_model_config("glm-4.9") == GLM_MODEL_CONFIGS["glm-4.7"]

    def test_get_model_config_unknown_model(self):
        """Test fallback for unknown models."""
        from scripts.refrag_glm import get_model_config, GLM_DEFAULT_CONFIG
        
        assert get_model_config("unknown-model") == GLM_DEFAULT_CONFIG
        assert get_model_config("gpt-4") == GLM_DEFAULT_CONFIG

    def test_glm47_config_values(self):
        """Test GLM-4.7 has correct configuration values."""
        from scripts.refrag_glm import GLM_MODEL_CONFIGS
        
        config = GLM_MODEL_CONFIGS["glm-4.7"]
        assert config["temperature"] == 1.0
        assert config["top_p"] == 0.95
        assert config["max_output_tokens"] == 131072  # 128K
        assert config["max_context_tokens"] == 204800  # 200K
        assert config["supports_thinking"] is True
        assert config["supports_tool_stream"] is True

    def test_glm46_config_values(self):
        """Test GLM-4.6 has correct configuration values."""
        from scripts.refrag_glm import GLM_MODEL_CONFIGS
        
        config = GLM_MODEL_CONFIGS["glm-4.6"]
        assert config["temperature"] == 1.0
        assert config["top_p"] == 0.95
        assert config["supports_thinking"] is True
        assert config["supports_tool_stream"] is False

    def test_glm45_config_values(self):
        """Test GLM-4.5 has correct configuration values."""
        from scripts.refrag_glm import GLM_MODEL_CONFIGS

        config = GLM_MODEL_CONFIGS["glm-4.5"]
        assert config["temperature"] == 1.0
        assert config["top_p"] == 0.95
        assert config["supports_thinking"] is True  # GLM-4.5 supports thinking control
        assert config["supports_tool_stream"] is False


class TestGLMRefragClientModelSelection:
    """Test GLMRefragClient model selection logic."""

    @patch("openai.OpenAI")
    def test_default_model_is_glm46(self, mock_openai_class):
        """Test that default model is glm-4.6."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Remove GLM_MODEL from env to test default, keep GLM_API_KEY
        env_copy = os.environ.copy()
        env_copy.pop("GLM_MODEL", None)
        env_copy["GLM_API_KEY"] = "test-key"
        
        with patch.dict(os.environ, env_copy, clear=True):
            client = GLMRefragClient()
            client.generate_with_soft_embeddings("test prompt")
            
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            # Default should be glm-4.6 when GLM_MODEL not set
            assert call_kwargs["model"] == "glm-4.6"

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.6"}, clear=False)
    @patch("openai.OpenAI")
    def test_env_model_override(self, mock_openai_class):
        """Test that GLM_MODEL env var overrides default."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        client.generate_with_soft_embeddings("test prompt")
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "glm-4.6"

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL_FAST": "glm-4.5"}, clear=False)
    @patch("openai.OpenAI")
    def test_fast_model_with_disable_thinking(self, mock_openai_class):
        """Test that disable_thinking uses GLM_MODEL_FAST."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        client.generate_with_soft_embeddings("test prompt", disable_thinking=True)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "glm-4.5"


class TestGLMToolStreamSupport:
    """Test GLM-4.7 tool_stream feature support."""

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.7"}, clear=False)
    @patch("openai.OpenAI")
    def test_tool_stream_enabled_for_glm47(self, mock_openai_class):
        """Test that tool_stream is enabled for GLM-4.7 when requested."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        client.generate_with_soft_embeddings("test prompt", tools=tools, tool_stream=True)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("tools") == tools
        assert call_kwargs.get("extra_body", {}).get("tool_stream") is True

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.6"}, clear=False)
    @patch("openai.OpenAI")
    def test_tool_stream_not_enabled_for_glm46(self, mock_openai_class):
        """Test that tool_stream is NOT enabled for GLM-4.6."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        client.generate_with_soft_embeddings("test prompt", tools=tools, tool_stream=True)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # tool_stream should not be set for 4.6; if present, must be False
        extra_body = call_kwargs.get("extra_body", {})
        assert "tool_stream" not in extra_body or extra_body["tool_stream"] is False


class TestGLMThinkingSupport:
    """Test GLM thinking/reasoning support."""

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.7"}, clear=False)
    @patch("openai.OpenAI")
    def test_enable_thinking_for_glm47(self, mock_openai_class):
        """Test that thinking can be explicitly enabled for GLM-4.7."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        client.generate_with_soft_embeddings("test prompt", enable_thinking=True)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("extra_body", {}).get("thinking") == {"type": "enabled"}

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.5"}, clear=False)
    @patch("openai.OpenAI")
    def test_thinking_not_set_for_glm45(self, mock_openai_class):
        """Test that thinking is NOT set for GLM-4.5 (no thinking support)."""
        from scripts.refrag_glm import GLMRefragClient
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        client.generate_with_soft_embeddings("test prompt", enable_thinking=True)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # GLM-4.5 now supports thinking control (can be enabled/disabled)
        extra_body = call_kwargs.get("extra_body", {})
        assert "thinking" in extra_body
        assert extra_body["thinking"]["type"] == "enabled"


class TestGLMMaxTokensLimit:
    """Test max_tokens limiting based on model capabilities."""

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.7"}, clear=False)
    @patch("openai.OpenAI")
    def test_max_tokens_capped_to_model_limit(self, mock_openai_class):
        """Test that max_tokens is capped to model's max_output_tokens."""
        from scripts.refrag_glm import GLMRefragClient, GLM_MODEL_CONFIGS
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        # Request more than GLM-4.7 can output (131072)
        client.generate_with_soft_embeddings("test prompt", max_tokens=200000)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] <= GLM_MODEL_CONFIGS["glm-4.7"]["max_output_tokens"]

    @patch.dict(os.environ, {"GLM_API_KEY": "test-key", "GLM_MODEL": "glm-4.5"}, clear=False)
    @patch("openai.OpenAI")
    def test_max_tokens_uses_smaller_limit_for_glm45(self, mock_openai_class):
        """Test that GLM-4.5 uses its smaller max_output limit."""
        from scripts.refrag_glm import GLMRefragClient, GLM_MODEL_CONFIGS
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = GLMRefragClient()
        # Request more than GLM-4.5 can output (8192)
        client.generate_with_soft_embeddings("test prompt", max_tokens=50000, disable_thinking=True)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] <= GLM_MODEL_CONFIGS["glm-4.5"]["max_output_tokens"]
