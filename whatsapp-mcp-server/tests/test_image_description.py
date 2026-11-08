"""Tests for image description helpers."""

from datetime import UTC, datetime, timedelta

from image_description import MIN_MODEL_DIMENSION, within_max_age


def _ago(**kwargs) -> str:
    """An ISO timestamp N units in the past, in the format messages.db stores."""
    return (datetime.now(UTC) - timedelta(**kwargs)).isoformat()


class TestWithinMaxAge:
    """Tests for the CDN refetch age gate."""

    def test_no_cutoff_always_allows(self):
        """No --max-age-days must preserve the previous unconditional behavior."""
        assert within_max_age(_ago(days=3650), None) is True
        assert within_max_age("garbage", None) is True

    def test_recent_message_allowed(self):
        assert within_max_age(_ago(hours=3), 30) is True

    def test_old_message_blocked(self):
        assert within_max_age(_ago(days=400), 30) is False

    def test_boundary_just_inside_and_outside(self):
        assert within_max_age(_ago(days=30, minutes=-5), 30) is True
        assert within_max_age(_ago(days=30, minutes=5), 30) is False

    def test_zero_days_blocks_everything(self):
        assert within_max_age(_ago(seconds=1), 0) is False

    def test_unparseable_timestamp_fails_open(self):
        """Never silently skip media that might still be recoverable."""
        for value in ("", "not-a-date", None):
            assert within_max_age(value, 30) is True

    def test_naive_timestamp_does_not_raise(self):
        """messages.db rows are tz-aware, but a naive value must not TypeError."""
        naive = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)
        assert within_max_age(naive.isoformat(), 30) is True

    def test_stored_timestamp_format(self):
        """The exact shape messages.db yields: space separator plus UTC offset."""
        assert within_max_age("2026-08-04 06:53:39-06:00", 36500) is True
        assert within_max_age("2023-05-01 12:00:00-06:00", 30) is False

    def test_future_timestamp_allowed(self):
        """Clock skew should not block a refetch."""
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        assert within_max_age(future, 30) is True


class TestModelMinimum:
    def test_min_dimension_matches_patch_stride(self):
        """16px patches merged 2x2; below this the Ollama runner crashes."""
        assert MIN_MODEL_DIMENSION == 32


class TestSplitDescriptionAndOcr:
    """The model's "no text" answers are stored as NULL OCR, real text is kept."""

    def test_no_text_phrasings_become_none(self):
        from image_description import _split_description_and_ocr

        for answer in [
            "None visible",
            "None visible.",
            "none",
            "(No visible text)",
            "There is no visible text in the image.",
            "(No text is visible in the image.)",
            "(No discernible text visible)",
            "(none visible)",
            "[No text is visible in the image.]",
            "(No readable text visible in the image)",
            "(not visible)",
        ]:
            assert _split_description_and_ocr(f"A photo.\nText: {answer}") == ("A photo.", None), answer

    def test_real_text_is_kept(self):
        from image_description import _split_description_and_ocr

        for answer in ["TE AMO", "No aplica para comer en restaurante", "None of the above", "Nonetheless"]:
            assert _split_description_and_ocr(f"An ad.\nText: {answer}") == ("An ad.", answer), answer

    def test_multiline_ocr_is_kept(self):
        from image_description import _split_description_and_ocr

        assert _split_description_and_ocr("An ad.\nText:\nLine one\nLine two") == ("An ad.", "Line one\nLine two")
