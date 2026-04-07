"""CommentMiner package."""

from .aggregation import AggregateCommentRunStats, aggregate_comment_runs
from .config import DatasetSpec, PipelineConfig, StorageConfig
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader, RedPajamaManifestDownloader
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import LicenseScanStats, scan_comment_licenses
from .models import CommentRecord, InputRecord, ShardedDatasetSource
from .pipeline import PipelineRunStats, run_dataset, run_sharded_dataset
from .sources import RedPajamaGithubSource, ShardRowCursor, TheStackParquetSource

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
    "RedPajamaGithubSource",
    "RedPajamaManifestDownloader",
    "ShardRowCursor",
    "ShardedDatasetSource",
    "StorageConfig",
    "TheStackParquetSource",
    "aggregate_comment_runs",
    "run_dataset",
    "run_sharded_dataset",
    "scan_comment_licenses",
]
