from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.topic_modelling import run_low_scancode_topic_modelling


class FakeTopicModel:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.topic_labels_ = {
            0: "0_mit_notice",
            1: "1_project_header",
        }

    def fit_transform(self, documents: list[str]):
        self.documents = list(documents)
        topics = [0 if "MIT" in document else 1 for document in documents]
        probabilities = [
            [0.91, 0.09] if topic == 0 else [0.21, 0.79]
            for topic in topics
        ]
        return topics, probabilities

    def get_topic(self, topic_id: int):
        if topic_id == 0:
            return [("mit", 0.6), ("permission", 0.2)]
        if topic_id == 1:
            return [("project", 0.5), ("generated", 0.3)]
        return []

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.txt").write_text("saved", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class TopicModellingTests(unittest.TestCase):
    def test_models_jsonl_comments_below_normalized_scancode_threshold_and_runs_codex_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "license-scan"
            output_directory = root / "topics"
            _write_jsonl(
                input_directory / "part-00000.jsonl",
                [
                    {
                        "dataset": "combined",
                        "record_id": "low-80",
                        "opening_comment": "MIT permission notice",
                        "language": "Python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "metadata": {},
                        "comment_license_score": 80.0,
                    },
                    {
                        "dataset": "combined",
                        "record_id": "low-ratio",
                        "opening_comment": "Generated project header",
                        "language": "Python",
                        "path": "b.py",
                        "repo": "repo-b",
                        "metadata": {},
                        "comment_license_score": 0.94,
                    },
                    {
                        "dataset": "combined",
                        "record_id": "at-threshold",
                        "opening_comment": "exactly threshold",
                        "language": "Python",
                        "path": "c.py",
                        "repo": "repo-c",
                        "metadata": {},
                        "comment_license_score": 95.0,
                    },
                    {
                        "dataset": "combined",
                        "record_id": "missing-score",
                        "opening_comment": "missing score",
                        "language": "Python",
                        "path": "d.py",
                        "repo": "repo-d",
                        "metadata": {},
                    },
                    {
                        "dataset": "combined",
                        "record_id": "empty-comment",
                        "opening_comment": "",
                        "language": "Python",
                        "path": "e.py",
                        "repo": "repo-e",
                        "metadata": {},
                        "comment_license_score": 10.0,
                    },
                ],
            )
            prompts: list[str] = []

            def _fake_codex_runner(prompt: str) -> str:
                prompts.append(prompt)
                return json.dumps(
                    {
                        "overall_assessment": "clusters are coherent",
                        "clusters": [
                            {
                                "topic_id": 0,
                                "valid_cluster": True,
                                "coherence_score": 0.9,
                                "suggested_label": "MIT notice",
                                "rationale": "MIT examples group together",
                                "weak_or_off_topic_ordinals": [],
                            }
                        ],
                    }
                )

            model = FakeTopicModel()
            stats = run_low_scancode_topic_modelling(
                input_directory,
                output_directory=output_directory,
                score_threshold=0.95,
                bertopic_model=model,
                judge_with_codex=True,
                codex_runner=_fake_codex_runner,
                save_model=True,
            )

            self.assertEqual(model.documents, ["MIT permission notice", "Generated project header"])
            self.assertEqual(stats.records_seen, 5)
            self.assertEqual(stats.records_selected, 2)
            self.assertEqual(stats.records_modelled, 2)
            self.assertEqual(stats.records_missing_score, 1)
            self.assertEqual(stats.records_without_comment, 1)
            self.assertEqual(stats.topics_discovered, 2)
            self.assertEqual(stats.outlier_records, 0)
            self.assertEqual(stats.normalized_score_threshold, 95.0)
            self.assertTrue(prompts)
            self.assertIn("Cluster data", prompts[0])

            assignments = [
                json.loads(line)
                for line in (output_directory / "topic-assignments.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["record_id"] for row in assignments], ["low-80", "low-ratio"])
            self.assertEqual([row["comment_license_score_percent"] for row in assignments], [80.0, 94.0])
            self.assertEqual([row["topic_id"] for row in assignments], [0, 1])
            self.assertEqual(assignments[0]["topic_probability"], 0.91)

            topics = json.loads((output_directory / "topics.json").read_text(encoding="utf-8"))
            self.assertEqual(topics["topic_count"], 2)
            self.assertEqual(topics["topics"][0]["keywords"][0]["term"], "mit")

            manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["normalized_score_threshold"], 95.0)
            self.assertEqual(manifest["records_selected"], 2)
            self.assertTrue(manifest["judge_with_codex"])

            report = json.loads((output_directory / "codex-cluster-validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["parsed_response"]["overall_assessment"], "clusters are coherent")
            self.assertTrue((output_directory / "bertopic-model" / "model.txt").exists())

    def test_models_huggingface_parquet_scan_output_with_dataset_language_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "hf-license-scan"
            python_dir = input_directory / "the-stack" / "Python"
            js_dir = input_directory / "the-stack" / "JavaScript"
            python_dir.mkdir(parents=True)
            js_dir.mkdir(parents=True)
            schema = pa.schema(
                [
                    ("dataset", pa.string()),
                    ("record_id", pa.string()),
                    ("opening_comment", pa.string()),
                    ("language", pa.string()),
                    ("path", pa.string()),
                    ("repo", pa.string()),
                    ("metadata", pa.string()),
                    ("comment_license_score", pa.float64()),
                ]
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "dataset": "the-stack",
                            "record_id": "py-low",
                            "opening_comment": "MIT notice",
                            "language": "Python",
                            "path": "a.py",
                            "repo": "repo-a",
                            "metadata": "{}",
                            "comment_license_score": 80.0,
                        },
                        {
                            "dataset": "the-stack",
                            "record_id": "py-high",
                            "opening_comment": "Apache notice",
                            "language": "Python",
                            "path": "b.py",
                            "repo": "repo-b",
                            "metadata": "{}",
                            "comment_license_score": 97.0,
                        },
                    ],
                    schema=schema,
                ),
                python_dir / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "dataset": "the-stack",
                            "record_id": "js-low",
                            "opening_comment": "Generated project header",
                            "language": "JavaScript",
                            "path": "a.js",
                            "repo": "repo-js",
                            "metadata": "{}",
                            "comment_license_score": 10.0,
                        }
                    ],
                    schema=schema,
                ),
                js_dir / "part-00000.parquet",
            )

            stats = run_low_scancode_topic_modelling(
                input_directory,
                output_directory=root / "topics",
                score_threshold=95.0,
                dataset_names=["the-stack"],
                languages=["Python"],
                bertopic_model=FakeTopicModel(),
            )

            self.assertEqual(stats.input_format, "parquet")
            self.assertEqual(stats.shards_read, 1)
            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(stats.records_selected, 1)

            assignments = [
                json.loads(line)
                for line in (root / "topics" / "topic-assignments.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(assignments), 1)
            self.assertEqual(assignments[0]["record_id"], "py-low")
            self.assertEqual(assignments[0]["source_path"], "the-stack/Python/part-00000.parquet")


if __name__ == "__main__":
    unittest.main()
