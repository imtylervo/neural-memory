"""Tests for quality scorer engine."""

from __future__ import annotations

import pytest

from neural_memory.engine.quality_scorer import QualityResult, score_memory


class TestScoreMemory:
    """Tests for score_memory()."""

    def test_minimal_content_low_quality(self) -> None:
        """Very short content with no metadata scores low."""
        result = score_memory("hello")
        assert result.quality == "low"
        assert result.score < 4
        assert len(result.hints) > 0

    def test_short_content_has_length_hint(self) -> None:
        """Content under 10 chars gets a length hint."""
        result = score_memory("hi")
        assert any("short" in h.lower() for h in result.hints)

    def test_long_content_gets_length_points(self) -> None:
        """Content >= 50 chars gets 2 length points."""
        content = "a" * 60
        result = score_memory(content)
        # At least 2 points from length
        assert result.score >= 2

    def test_context_dict_adds_3_points(self) -> None:
        """Providing context dict adds 3 points."""
        base = score_memory("some content here")
        with_ctx = score_memory(
            "some content here",
            context={"reason": "testing"},
        )
        assert with_ctx.score >= base.score + 3

    def test_empty_context_no_bonus(self) -> None:
        """Empty context dict does not add points."""
        base = score_memory("some content here")
        with_empty = score_memory("some content here", context={})
        assert with_empty.score == base.score

    def test_tags_add_1_point(self) -> None:
        """Providing at least 1 tag adds 1 point."""
        base = score_memory("some content here")
        with_tags = score_memory("some content here", tags=["test"])
        assert with_tags.score == base.score + 1

    def test_empty_tags_no_bonus(self) -> None:
        """Empty tag list does not add points."""
        base = score_memory("some content here")
        with_empty = score_memory("some content here", tags=[])
        assert with_empty.score == base.score

    def test_non_fact_type_adds_1_point(self) -> None:
        """Non-default type adds 1 point."""
        base = score_memory("some content here", memory_type="fact")
        decision = score_memory("some content here", memory_type="decision")
        assert decision.score == base.score + 1

    def test_no_type_gives_hint(self) -> None:
        """No type provided gives a hint."""
        result = score_memory("some content here")
        assert any("type" in h.lower() for h in result.hints)

    def test_causal_word_adds_point(self) -> None:
        """Causal language adds 1 cognitive richness point."""
        without = score_memory("chose PostgreSQL for payments")
        with_causal = score_memory("chose PostgreSQL because ACID needed for payments")
        assert with_causal.score > without.score

    def test_temporal_word_adds_point(self) -> None:
        """Temporal language adds 1 cognitive richness point."""
        without = score_memory("upgraded the database schema")
        with_temporal = score_memory("after upgrading the database schema, tests broke")
        assert with_temporal.score > without.score

    def test_comparative_word_adds_point(self) -> None:
        """Comparative language adds 1 cognitive richness point."""
        without = score_memory("selected PostgreSQL for the project")
        with_comp = score_memory("selected PostgreSQL over MongoDB for the project")
        assert with_comp.score > without.score

    def test_no_cognitive_words_gives_hint(self) -> None:
        """No causal/temporal/comparative words gives a reasoning hint."""
        result = score_memory("PostgreSQL database")
        assert any("reasoning" in h.lower() or "why" in h.lower() for h in result.hints)

    def test_high_quality_no_hints(self) -> None:
        """High quality memories get no hints."""
        result = score_memory(
            "Chose PostgreSQL over MongoDB because ACID needed for payment processing",
            memory_type="decision",
            tags=["database", "payments"],
            context={"reason": "ACID compliance", "alternatives": ["MongoDB"]},
        )
        assert result.quality == "high"
        assert result.score >= 7
        assert len(result.hints) == 0

    def test_medium_quality(self) -> None:
        """Medium quality has score 4-6."""
        result = score_memory(
            "PostgreSQL is our database because it supports ACID",
            memory_type="fact",
            tags=["db"],
        )
        assert result.quality == "medium"
        assert 4 <= result.score <= 6

    def test_score_capped_at_10(self) -> None:
        """Score never exceeds 10."""
        result = score_memory(
            "After testing, chose PostgreSQL over MongoDB because ACID "
            "needed and it was faster than alternatives for our use case",
            memory_type="decision",
            tags=["database", "payments", "architecture"],
            context={
                "reason": "ACID for payments",
                "alternatives": ["MongoDB", "CockroachDB"],
                "impact": "critical",
            },
        )
        assert result.score <= 10

    def test_immutability(self) -> None:
        """score_memory does not mutate inputs."""
        tags = ["original"]
        context = {"reason": "test"}
        tags_copy = list(tags)
        context_copy = dict(context)
        score_memory("test content", tags=tags, context=context)
        assert tags == tags_copy
        assert context == context_copy


class TestQualityResult:
    """Tests for QualityResult dataclass."""

    def test_to_dict_with_hints(self) -> None:
        result = QualityResult(score=3, quality="low", hints=("Add tags",))
        d = result.to_dict()
        assert d == {"quality": "low", "score": 3, "hints": ["Add tags"]}

    def test_to_dict_no_hints(self) -> None:
        result = QualityResult(score=8, quality="high", hints=())
        d = result.to_dict()
        assert d == {"quality": "high", "score": 8}
        assert "hints" not in d

    def test_frozen(self) -> None:
        result = QualityResult(score=5, quality="medium")
        with pytest.raises(AttributeError):
            result.score = 10  # type: ignore[misc]


class TestTypeSpecificHints:
    """Encoding template hints for error and decision memories."""

    @staticmethod
    def _has(result: QualityResult, word: str) -> bool:
        return any(word in h for h in result.hints)

    def test_error_without_cause_or_fix(self) -> None:
        r = score_memory("Login endpoint returns 500", memory_type="error")
        assert self._has(r, "root cause")
        assert self._has(r, "the fix")

    def test_error_with_cause_only(self) -> None:
        r = score_memory("Login returns 500 because the token was null", memory_type="error")
        assert not self._has(r, "root cause")
        assert self._has(r, "the fix")

    def test_error_with_fix_only(self) -> None:
        r = score_memory("Login returns 500, fixed by adding a null check", memory_type="error")
        assert self._has(r, "root cause")
        assert not self._has(r, "the fix")

    def test_error_with_cause_and_fix(self) -> None:
        r = score_memory(
            "Login failed because token was null, fixed by a null check", memory_type="error"
        )
        assert not self._has(r, "root cause")
        assert not self._has(r, "the fix")

    def test_error_with_root_cause_phrase(self) -> None:
        r = score_memory("Root cause: stale cache entry. Workaround: restart", memory_type="error")
        assert not self._has(r, "root cause")
        assert not self._has(r, "the fix")

    def test_decision_why_hint(self) -> None:
        without = score_memory("Use Postgres over SQLite for sync", memory_type="decision")
        with_why = score_memory(
            "Use Postgres over SQLite for sync because of concurrent writers",
            memory_type="decision",
        )
        assert self._has(without, "state why")
        assert not self._has(with_why, "state why")

    @pytest.mark.parametrize("memory_type", ["fact", "insight", None])
    def test_other_types_get_no_template_hints(self, memory_type: str | None) -> None:
        r = score_memory("Login endpoint returns 500", memory_type=memory_type)
        assert not self._has(r, "Error memory")
        assert not self._has(r, "Decision memory")

    def test_score_unchanged_by_template_hints(self) -> None:
        err = score_memory("Login endpoint returns 500 on empty password", memory_type="error")
        fact = score_memory("Login endpoint returns 500 on empty password", memory_type="insight")
        assert (err.score, err.quality) == (fact.score, fact.quality) == (3, "low")

    def test_high_quality_error_has_no_hints(self) -> None:
        r = score_memory(
            "Auth failed after deploy v2.1 because token was null, fixed by null check in auth.py",
            memory_type="error",
            tags=["auth"],
            context={"cause": "null token"},
        )
        assert r.score >= 7
        assert r.hints == ()
