"""CommentMiner package."""

from .config import DatasetSpec, PipelineConfig, StorageConfig
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader, RedPajamaManifestDownloader
from .extractors import ML4SEOpeningCommentExtractor
from .models import CommentRecord, InputRecord, ShardedDatasetSource
from .pipeline import PipelineRunStats, run_dataset, run_sharded_dataset
from .sources import RedPajamaGithubSource, ShardRowCursor, TheStackParquetSource

__all__ = [
    "CommentRecord",
    "DatasetSpec",
    "DownloadPlan",
    "DownloadSummary",
    "HuggingFaceDownloader",
    "InputRecord",
    "ML4SEOpeningCommentExtractor",
    "PipelineConfig",
    "PipelineRunStats",
    "RedPajamaGithubSource",
    "RedPajamaManifestDownloader",
    "ShardRowCursor",
    "ShardedDatasetSource",
    "StorageConfig",
    "TheStackParquetSource",
    "run_dataset",
    "run_sharded_dataset",
]
