"""CommentMiner package."""

from .aggregation import AggregateCommentRunStats, aggregate_comment_runs
from .config import DatasetSpec, PipelineConfig, StorageConfig
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import LicenseScanStats, scan_comment_licenses
from .models import CommentRecord, InputRecord
from .pipeline import PipelineRunStats, run_dataset
from .sources import ShardRowCursor, TheStackParquetSource

__all__ = [
    "AggregateCommentRunStats",
    "CommentRecord",
    "DatasetSpec",
    "DownloadPlan",
    "DownloadSummary",
    "HuggingFaceDownloader",
    "InputRecord",
    "LicenseScanStats",
    "ML4SEOpeningCommentExtractor",
    "PipelineConfig",
    "PipelineRunStats",
    "ShardRowCursor",
    "StorageConfig",
    "TheStackParquetSource",
    "aggregate_comment_runs",
    "run_dataset",
    "scan_comment_licenses",
]
