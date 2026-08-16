from __future__ import annotations

import unittest
from unittest.mock import patch

from commentminer.parquet_io import estimated_record_bytes


class ParquetIoTests(unittest.TestCase):
    def test_record_size_estimate_does_not_serialize_json(self) -> None:
        record = {
            "dataset": "the-stack-v3-full",
            "record_id": "row-1",
            "opening_comment": "Copyright © example",
            "language": "Python",
            "path": None,
            "repo": "1",
            "extracted_at": "2026-08-06T00:00:00+00:00",
            "metadata": '{"row_index":1}',
        }

        with patch("commentminer.parquet_io.json.dumps") as dumps:
            estimate = estimated_record_bytes(record)

        dumps.assert_not_called()
        self.assertGreater(estimate, 192)

    def test_record_size_estimate_increases_with_utf8_payload(self) -> None:
        short = estimated_record_bytes({"opening_comment": "x"})
        long = estimated_record_bytes({"opening_comment": "é" * 1_000})

        self.assertGreater(long, short + 2_000)


if __name__ == "__main__":
    unittest.main()
