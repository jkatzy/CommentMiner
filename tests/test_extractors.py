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

    def test_ml4se_extractor_joins_line_comments_across_up_to_five_blank_lines(
        self,
    ) -> None:
        extractor = ML4SEOpeningCommentExtractor()

        separators = (
            "\n\n",
            "\n \t \n",
            "\r\n\r\n",
            "\r\n \t \r\n",
            "\n" * 6,
            "\n" + (" \t \n" * 5),
            "\r\n" * 6,
            "\r\n" + (" \t \r\n" * 5),
        )
        for separator in separators:
            with self.subTest(separator=repr(separator)):
                record = InputRecord(
                    dataset="the-stack",
                    record_id="r-blank-line",
                    content=(
                        "# first part\n"
                        f"# first continuation{separator}"
                        "# second part\n"
                        "# second continuation\n"
                        "print('x')\n"
                    ),
                    language="Python",
                )

                comments = extractor.extract_opening_comments(record)

                self.assertEqual(len(comments), 1)
                self.assertEqual(
                    comments[0].text,
                    (
                        "# first part\n"
                        f"# first continuation{separator}"
                        "# second part\n"
                        "# second continuation"
                    ),
                )

    def test_ml4se_extractor_keeps_line_comments_separate_across_six_blank_lines(
        self,
    ) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        record = InputRecord(
            dataset="the-stack",
            record_id="r-six-blank-lines",
            content="# first part" + ("\n" * 7) + "# second part\nprint('x')\n",
            language="Python",
        )

        comments = extractor.extract_opening_comments(record)

        self.assertEqual(
            [comment.text for comment in comments],
            ["# first part", "# second part"],
        )

    def test_ml4se_extractor_only_joins_line_comments_across_blank_line(
        self,
    ) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        cases = (
            (
                "block comments",
                "/* first part */\n\n/* second part */\n",
                ["/* first part */", "/* second part */"],
            ),
            (
                "mixed comments",
                "// first part\n\n/* second part */\n",
                ["// first part", "/* second part */"],
            ),
        )

        for name, content, expected in cases:
            with self.subTest(name=name):
                record = InputRecord(
                    dataset="the-stack",
                    record_id="r-non-line-blank",
                    content=content,
                    language="C",
                )

                comments = extractor.extract_opening_comments(record)

                self.assertEqual(
                    [comment.text for comment in comments],
                    expected,
                )

    def test_ml4se_extractor_resolves_common_extension_alias(self) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        record = InputRecord(
            dataset="the-stack",
            record_id="r-ext",
            content="# header line\n\nprint('x')\n",
            metadata={"ext": "py"},
        )

        match = extractor.extract_opening_comment(record)

        self.assertEqual(match, "# header line")
        self.assertIn("python", extractor._queries)

    def test_ml4se_extractor_does_not_cache_unsupported_extension_as_record_failure(self) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        first = InputRecord(
            dataset="the-stack",
            record_id="r-header-1",
            content="/* header */\nint first;\n",
            language="c++",
            metadata={"ext": "h", "path_language": "c++"},
        )
        second = InputRecord(
            dataset="the-stack",
            record_id="r-header-2",
            content="/* header */\nint second;\n",
            language="c++",
            metadata={"ext": "h", "path_language": "c++"},
        )

        self.assertEqual(extractor.extract_opening_comment(first), "/* header */")
        self.assertEqual(extractor.extract_opening_comment(second), "/* header */")
        self.assertIn("c++", extractor._queries)

    def test_ml4se_extractor_returns_comments_starting_within_first_ten_lines(self) -> None:
        extractor = ML4SEOpeningCommentExtractor(max_start_row=10)
        record = InputRecord(
            dataset="the-stack",
            record_id="r-preproc",
            content=(
                "#include <stdio.h>\n"
                "#define VALUE 1\n"
                "\n"
                "/* header */\n"
                "int value = VALUE; // inline detail\n"
                "\n"
                "// setup note\n"
                "int main(void) { return value; }\n"
                "\n"
                "\n"
                "/* too late */\n"
            ),
            language="c",
            metadata={"ext": "c", "path_language": "c"},
        )

        comments = extractor.extract_opening_comments(record)

        self.assertEqual(
            [comment.text for comment in comments],
            ["/* header */", "// inline detail\n\n// setup note"],
        )
        self.assertEqual(
            [comment.start_line for comment in comments],
            [4, 5],
        )

    def test_ml4se_extractor_keeps_long_comment_that_starts_within_first_ten_lines(self) -> None:
        extractor = ML4SEOpeningCommentExtractor(max_start_row=10)
        long_body = "\n".join(f" * detail {index}" for index in range(200))
        record = InputRecord(
            dataset="the-stack-v2",
            record_id="r-long",
            content=f"int prelude;\n/* header\n{long_body}\n*/\nint value;\n",
            language="c++",
            metadata={"selected_language": "C++"},
        )

        comments = extractor.extract_opening_comments(record)

        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].start_line, 2)
        self.assertTrue(comments[0].text.startswith("/* header\n * detail 0"))
        self.assertTrue(comments[0].text.endswith("*/"))
        self.assertIn("detail 199", comments[0].text)

    def test_ml4se_extractor_excludes_long_comment_that_starts_after_first_ten_lines(self) -> None:
        extractor = ML4SEOpeningCommentExtractor(max_start_row=10)
        prefix = "\n".join(f"int value_{index};" for index in range(10))
        record = InputRecord(
            dataset="the-stack-v2",
            record_id="r-late",
            content=f"{prefix}\n/* too late\n * detail\n*/\n",
            language="c++",
            metadata={"selected_language": "C++"},
        )

        self.assertEqual(extractor.extract_opening_comments(record), [])

    def test_ml4se_extractor_ignores_comment_marker_in_opening_string(self) -> None:
        extractor = ML4SEOpeningCommentExtractor(max_start_row=10)
        record = InputRecord(
            dataset="the-stack-v2",
            record_id="r-string",
            content='const char *url = "https://example.test/path";\nint value;\n',
            language="c++",
            metadata={"selected_language": "C++"},
        )

        self.assertEqual(extractor.extract_opening_comments(record), [])

    def test_ml4se_extractor_continues_after_cached_unsupported_candidate(self) -> None:
        extractor = ML4SEOpeningCommentExtractor()
        first = InputRecord(
            dataset="the-stack",
            record_id="r-cache-1",
            content="/* header */\nint first;\n",
            language="c++",
            metadata={"selected_language": "not-a-language"},
        )
        second = InputRecord(
            dataset="the-stack",
            record_id="r-cache-2",
            content="/* header */\nint second;\n",
            language="c++",
            metadata={"selected_language": "not-a-language"},
        )

        self.assertEqual(extractor.extract_opening_comment(first), "/* header */")
        self.assertEqual(extractor.extract_opening_comment(second), "/* header */")
        self.assertIsNone(extractor._query_languages["not-a-language"])
        self.assertIn("c++", extractor._queries)

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


if __name__ == "__main__":
    unittest.main()
