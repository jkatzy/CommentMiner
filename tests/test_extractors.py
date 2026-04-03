from __future__ import annotations

import unittest

from commentminer.extractors import ML4SEOpeningCommentExtractor
from commentminer.models import InputRecord


class ExtractorTests(unittest.TestCase):
    def test_ml4se_extractor_uses_record_metadata_language(self) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        record = InputRecord(
            dataset="the-stack",
            record_id="r1",
            content="# header line\n# second line\n\nprint('x')\n",
            language="Python",
            metadata={"ext": "py", "lang": "Python", "selected_language": "python"},
        )

        match = extractor.extract_opening_comment(record)

        self.assertEqual(match, "# header line\n# second line")
        self.assertIn("python", extractor._queries)

    def test_ml4se_extractor_returns_none_for_unsupported_language(self) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        record = InputRecord(
            dataset="the-stack",
            record_id="r2",
            content="such wow\n",
            language="Dogescript",
            metadata={"selected_language": "dogescript"},
        )

        self.assertIsNone(extractor.extract_opening_comment(record))

    def test_ml4se_extractor_limits_parser_input_by_character_count(self) -> None:
        extractor = ML4SEOpeningCommentExtractor(max_input_characters=12)
        record = InputRecord(
            dataset="the-stack",
            record_id="r3",
            content="# short\nprint('x')\n",
            language="Python",
            metadata={"selected_language": "python"},
        )

        match = extractor.extract_opening_comment(record)

        self.assertEqual(match, "# short")


if __name__ == "__main__":
    unittest.main()
