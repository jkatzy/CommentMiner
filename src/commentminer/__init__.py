"""CommentMiner package."""

from .aggregation import AggregateCommentRunStats, aggregate_comment_runs
from .config import DatasetSpec, PipelineConfig, StorageConfig
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader
from .deduplication import DeduplicateCommentRunStats, deduplicate_comment_run
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import LicenseScanStats, scan_comment_licenses
from .models import CommentRecord, InputRecord
from .pipeline import PipelineRunStats, run_dataset
from .sources import ShardRowCursor, StackV2SWHContentSource, TheStackParquetSource

__all__ = [
    "AggregateCommentRunStats",
    "CommentRecord",
    "DatasetSpec",
    "DeduplicateCommentRunStats",
    "DownloadPlan",
    "DownloadSummary",
    "HuggingFaceDownloader",
    "InputRecord",
    "LicenseScanStats",
    "ML4SEOpeningCommentExtractor",
    "PipelineConfig",
    "PipelineRunStats",
    "ShardRowCursor",
    "StackV2SWHContentSource",
    "StorageConfig",
    "TheStackParquetSource",
    "aggregate_comment_runs",
    "deduplicate_comment_run",
    "run_dataset",
    "scan_comment_licenses",
]
