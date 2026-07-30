"""CommentMiner package."""

from .config import DatasetSpec, PipelineConfig, StorageConfig
from .downloader import DownloadPlan, DownloadSummary, HuggingFaceDownloader
from .encoding_benchmark import (
    EncodingCapacityStats,
    EncodingModelSpec,
    run_encoding_capacity_benchmark,
)
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import (
    LicenseCachePrewarmStats,
    LicenseScanStats,
    LicenseScoreHistogram,
    build_license_score_histogram,
    prewarm_huggingface_license_detection_cache,
    scan_huggingface_comment_licenses,
)
from .models import CommentRecord, InputRecord
from .pipeline import PipelineRunStats, run_dataset
from .redistribution_candidates import (
    RedistributionBuildStats,
    RedistributionVerificationReport,
    build_redistribution_candidates,
    verify_redistribution_candidates,
)
from .sources import ShardRowCursor, StackV2SWHContentSource, TheStackParquetSource
from .topic_modelling import (
    EXPORT_CONTROL_SEEDS,
    FUNDING_DISSEMINATION_SEEDS,
    GOVERNMENT_RESTRICTION_SEEDS,
    PROPRIETARY_PROVENANCE_SEEDS,
    SEED_TOPICS,
    TopicModellingStats,
    UNPUBLISHED_WORK_SEEDS,
    run_low_scancode_topic_modelling,
)

__all__ = [
    "CommentRecord",
    "DatasetSpec",
    "DownloadPlan",
    "DownloadSummary",
    "EncodingCapacityStats",
    "EncodingModelSpec",
    "EXPORT_CONTROL_SEEDS",
    "FUNDING_DISSEMINATION_SEEDS",
    "GOVERNMENT_RESTRICTION_SEEDS",
    "HuggingFaceDownloader",
    "InputRecord",
    "LicenseCachePrewarmStats",
    "LicenseScanStats",
    "LicenseScoreHistogram",
    "ML4SEOpeningCommentExtractor",
    "PipelineConfig",
    "PipelineRunStats",
    "PROPRIETARY_PROVENANCE_SEEDS",
    "RedistributionBuildStats",
    "RedistributionVerificationReport",
    "SEED_TOPICS",
    "ShardRowCursor",
    "StackV2SWHContentSource",
    "StorageConfig",
    "TheStackParquetSource",
    "TopicModellingStats",
    "UNPUBLISHED_WORK_SEEDS",
    "build_license_score_histogram",
    "build_redistribution_candidates",
    "prewarm_huggingface_license_detection_cache",
    "run_encoding_capacity_benchmark",
    "run_dataset",
    "run_low_scancode_topic_modelling",
    "scan_huggingface_comment_licenses",
    "verify_redistribution_candidates",
]
