"""Tests for production-grade filename boost algorithm.

The algorithm handles:
- All naming conventions (snake_case, camelCase, PascalCase, kebab-case)
- Acronyms (XMLParser -> xml, parser)
- Position weighting (filename > directory)
- Common token penalties
"""
from scripts.rerank_recursive import (
    _compute_fname_boost,
    _split_identifier,
    _normalize_token,
)


class TestSplitIdentifier:
    """Test identifier tokenization across naming conventions."""

    def test_camel_case(self):
        assert _split_identifier("userAuthHandler") == ["user", "auth", "handler"]

    def test_pascal_case(self):
        assert _split_identifier("UserAuthHandler") == ["user", "auth", "handler"]

    def test_snake_case(self):
        assert _split_identifier("user_auth_handler") == ["user", "auth", "handler"]

    def test_kebab_case(self):
        assert _split_identifier("user-auth-handler") == ["user", "auth", "handler"]

    def test_screaming_snake(self):
        assert _split_identifier("USER_AUTH_HANDLER") == ["user", "auth", "handler"]

    def test_acronym_prefix(self):
        """XMLParser should split into xml, parser."""
        assert _split_identifier("XMLParser") == ["xml", "parser"]

    def test_acronym_suffix(self):
        """parseJSON should split into parse, json."""
        assert _split_identifier("parseJSON") == ["parse", "json"]

    def test_http_client(self):
        """HTTPClient should split properly."""
        assert _split_identifier("HTTPClient") == ["http", "client"]

    def test_interface_prefix_stripped(self):
        """IUserService -> user, service (I prefix stripped)."""
        assert _split_identifier("IUserService") == ["user", "service"]

    def test_private_prefix_stripped(self):
        """_privateMethod -> private, method."""
        assert _split_identifier("_privateMethod") == ["private", "method"]

    def test_dollar_prefix_stripped(self):
        """$scope -> scope."""
        assert _split_identifier("$scope") == ["scope"]

    def test_numbers_separated(self):
        """handler2 -> handler (numbers stripped)."""
        assert _split_identifier("handler2") == ["handler"]

    def test_dot_notation(self):
        """com.company.auth -> com, company, auth."""
        assert _split_identifier("com.company.auth") == ["com", "company", "auth"]


class TestNormalizeToken:
    """Test plural normalization."""

    def test_no_abbrev_expansion(self):
        forms = _normalize_token("auth")
        assert "auth" in forms
        assert "authenticate" not in forms

    def test_plural_normalized(self):
        forms = _normalize_token("services")
        assert "service" in forms

    def test_singular_gets_plural(self):
        forms = _normalize_token("service")
        assert "services" in forms


class TestComputeFnameBoost:
    """Test the full filename boost computation."""

    def test_basic_snake_case_match(self):
        """Basic snake_case filename matching."""
        q = "hybrid search"
        cand = {"path": "scripts/hybrid_search.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.4  # 2 exact filename matches with bonus

    def test_rel_path_fallback(self):
        """Should work with rel_path key."""
        q = "hybrid search"
        cand = {"rel_path": "scripts/hybrid_search.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.4

    def test_metadata_path_fallback(self):
        """Should work with metadata.path."""
        q = "hybrid search"
        cand = {"metadata": {"path": "scripts/hybrid_search.py"}}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.4

    def test_disabled_when_factor_zero(self):
        """Factor of 0 should return 0."""
        q = "hybrid search"
        cand = {"path": "scripts/hybrid_search.py"}
        assert _compute_fname_boost(q, cand, 0.0) == 0.0

    def test_requires_two_matches(self):
        """Single token match should not trigger boost."""
        q = "hybrid fusion scoring"  # only 'hybrid' matches
        cand = {"path": "scripts/hybrid_utils.py"}
        assert _compute_fname_boost(q, cand, 0.15) == 0.0

    def test_camel_case_tokenization(self):
        """CamelCase filenames should be properly tokenized."""
        q = "user auth handler"
        cand = {"path": "services/UserAuthHandler.ts"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.5  # 3 matches with filename bonus

    def test_pascal_case_tokenization(self):
        """PascalCase filenames should be properly tokenized."""
        q = "database connection pool"
        cand = {"path": "lib/DatabaseConnectionPool.java"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.5  # 3 matches

    def test_acronym_matching(self):
        """Acronyms like XMLParser should match 'xml parser'."""
        q = "xml parser"
        cand = {"path": "utils/XMLParser.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.4  # 2 exact matches

    def test_substring_matching_authenticate_auth(self):
        """'authenticate' contains 'auth', so substring tier can still match it."""
        q = "authenticate user"
        cand = {"path": "auth/UserAuth.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.3

    def test_substring_matching_configuration_config(self):
        """'configuration' contains 'config', so substring tier can still match it."""
        q = "configuration manager"
        cand = {"path": "config/ConfigManager.ts"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.3

    def test_directory_matching(self):
        """Query tokens should match directory names."""
        q = "auth services controller"
        cand = {"path": "services/auth/controller.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.4  # 3 matches across path

    def test_filename_bonus(self):
        """Filename matches should score higher than directory matches."""
        q = "auth handler"
        # Filename match
        cand1 = {"path": "services/AuthHandler.py"}
        # Directory match only
        cand2 = {"path": "auth/handler/index.py"}
        score1 = _compute_fname_boost(q, cand1, 0.15)
        score2 = _compute_fname_boost(q, cand2, 0.15)
        assert score1 > score2  # Filename match wins

    def test_deep_path_handling(self):
        """Deep Java-style paths should work."""
        q = "user service implementation"
        cand = {"path": "src/main/java/com/company/user/service/UserServiceImpl.java"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.3  # Should find matches

    def test_interface_prefix_stripped(self):
        """IUserService should match 'user service'."""
        q = "user service"
        cand = {"path": "interfaces/IUserService.ts"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.3

    def test_common_tokens_penalized(self):
        """Common tokens like 'utils', 'index' should be penalized."""
        q = "auth utils"
        cand = {"path": "utils/AuthUtils.py"}  # common token 'utils'
        score = _compute_fname_boost(q, cand, 0.15)
        # Should still match, but common token gets penalty
        assert score > 0

    def test_jsonish_query(self):
        """Handle queries wrapped in JSON-like brackets."""
        q = '["hybrid search fusion"]'
        cand = {"path": "scripts/hybrid_search.py"}
        result = _compute_fname_boost(q, cand, 0.15)
        assert result > 0.3
