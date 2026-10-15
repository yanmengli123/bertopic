from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .common import WorkflowError, read_json

FIELDS = {"title", "abstract", "year", "date", "doi", "journal", "authors", "keywords",
          "institutions", "countries", "category", "research_areas", "addresses", "document_type", "publication_type", "pmid", "wos_id"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputFile(StrictModel):
    path: str
    source: Literal["auto", "VIP", "CNKI", "WanFang", "WoS", "PubMed", "Generic"] = "auto"
    sheet: str | int | None = None
    header_row: int | None = Field(default=None, ge=1)
    encoding: str | None = None
    delimiter: str | None = Field(default=None, min_length=1, max_length=1)
    mapping: dict[str, str] = Field(default_factory=dict, json_schema_extra={'propertyNames': {'enum': sorted(FIELDS)}})

    @field_validator("mapping")
    @classmethod
    def valid_fields(cls, value):
        if set(value) - FIELDS:
            raise ValueError("mapping 含未知标准字段")
        return value


class ImportOptions(StrictModel):
    invalid_rows: Literal["error", "quarantine"] = "quarantine"
    deduplication: Literal["none", "exact_text", "doi"] = "none"
    deduplication_scope: Literal["global", "within_source"] = "global"
    min_records: int = Field(default=5, ge=3)
    max_records: int = Field(default=200000, ge=3)
    max_file_mb: int = Field(default=512, ge=1, le=4096)
    max_uncompressed_mb: int = Field(default=1024, ge=1, le=8192)


class EmbeddingOptions(StrictModel):
    backend: Literal["siliconflow", "openai_compatible", "local", "precomputed"] = "siliconflow"
    model: str = Field(default="BAAI/bge-m3", min_length=1)
    model_revision: str = "main"
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key_env: str = Field(default="SILICONFLOW_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*(?:_API_KEY|_TOKEN)$")
    dimension: int | None = Field(default=None, ge=2, le=65536)
    tokenizer_path: str | None = None
    tokenizer_id: str | None = None
    max_tokens: int | None = Field(default=None, ge=8, le=32768)
    batch_size: int = Field(default=16, ge=1, le=128)
    retries: int = Field(default=4, ge=0, le=10)
    timeout_seconds: int = Field(default=120, ge=5, le=600)
    allow_model_download: bool = True
    allow_loopback_http: bool = False
    offline: bool = False
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    vectors_path: str | None = None
    metadata_path: str | None = None

    @model_validator(mode="after")
    def check_backend(self):
        u = urlsplit(self.base_url)
        if u.username or u.password or u.query or u.fragment or not u.hostname:
            raise ValueError("base_url 不能包含凭据、查询参数或片段")
        loopback = u.hostname in {"127.0.0.1", "localhost", "::1"}
        if u.scheme != "https" and not (u.scheme == "http" and loopback and self.allow_loopback_http):
            raise ValueError("远程 API 必须使用 HTTPS；本机 HTTP 需显式启用")
        if self.backend == "precomputed" and not (self.vectors_path and self.metadata_path):
            raise ValueError("预计算向量必须提供 vectors_path 和 metadata_path")
        if self.backend in {"siliconflow", "openai_compatible"} and self.max_tokens is None:
            if self.model not in {"BAAI/bge-m3", "BAAI/bge-large-zh-v1.5"}:
                raise ValueError("自定义远程模型必须配置 max_tokens 和相应 tokenizer")
        return self


class ModelOptions(StrictModel):
    min_cluster_sizes: list[int] = Field(default_factory=lambda: [15, 30, 50], min_length=1)
    min_samples: int = Field(default=5, ge=1)
    n_neighbors: int = Field(default=30, ge=2)
    n_components: int = Field(default=5, ge=2, le=20)
    seed: int = Field(default=42, ge=0)
    stability_seeds: list[int] = Field(default_factory=lambda: [7, 11, 23])
    top_n_words: int = Field(default=20, ge=5, le=100)
    max_features: int = Field(default=50000, ge=100)
    max_topics: int = Field(default=1000, ge=2, le=10000)
    silhouette_sample: int = Field(default=2000, ge=20, le=10000)
    stopwords: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    representative_count: int = Field(default=5, ge=1, le=30)

    @field_validator("min_cluster_sizes")
    @classmethod
    def sizes(cls, value):
        if any(n < 2 for n in value) or len(set(value)) != len(value):
            raise ValueError("最小簇规模应为不重复的、至少为 2 的整数")
        return value

    @model_validator(mode="after")
    def seeds(self):
        if len(set(self.stability_seeds)) != len(self.stability_seeds) or self.seed in self.stability_seeds:
            raise ValueError("比较种子不能重复或等于主种子")
        if any(s < 0 for s in self.stability_seeds):
            raise ValueError("种子不能为负数")
        return self


class ReportOptions(StrictModel):
    year_start: int | None = Field(default=None, ge=1500, le=2200)
    year_end: int = Field(default_factory=lambda: date.today().year, ge=1500, le=2200)
    incomplete_year: int | None = Field(default_factory=lambda: date.today().year, ge=1500, le=2200)
    top_topics: int = Field(default=16, ge=1, le=100)
    topic_details: int = Field(default=48, ge=0, le=1000)
    max_plot_documents: int = Field(default=60000, ge=100)
    dpi: int = Field(default=300, ge=100, le=600)
    font_path: str | None = None
    formats: list[Literal["png", "svg", "pdf"]] = Field(default_factory=lambda: ["png", "svg", "pdf"], min_length=1)
    network_quantile: float = Field(default=0.95, ge=0, le=1)
    network_min_cosine: float = Field(default=0.5, ge=-1, le=1)
    time_bin_years: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def valid_years(self):
        if self.year_start is not None and self.year_start > self.year_end:
            raise ValueError("year_start 不能晚于 year_end")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("输出格式不能重复")
        return self


class WorkflowConfig(StrictModel):
    schema_version: Literal[1] = 1
    title: str = Field(default="文献主题分析", max_length=256)
    inputs: list[InputFile] = Field(min_length=1)
    output_root: str = "runs"
    cache_root: str = ".cache"
    import_options: ImportOptions = Field(default_factory=ImportOptions)
    embedding: EmbeddingOptions = Field(default_factory=EmbeddingOptions)
    model: ModelOptions = Field(default_factory=ModelOptions)
    report: ReportOptions = Field(default_factory=ReportOptions)


def load_config(path):
    path = Path(path).resolve()
    try:
        config = WorkflowConfig.model_validate(read_json(path))
    except (ValidationError, ValueError, OSError) as error:
        # Never include invalid raw values (they may contain a pasted secret).
        locations = [".".join(map(str, e["loc"])) for e in error.errors()] if isinstance(error, ValidationError) else []
        raise WorkflowError("CONFIG_INVALID", "配置无效，请对照 schema 修改。", details={"fields": locations}) from None
    base = path.parent
    def resolve(value):
        return str((base / value).resolve()) if value else None
    for item in config.inputs:
        item.path = resolve(item.path)
    config.output_root = resolve(config.output_root)
    config.cache_root = resolve(config.cache_root)
    for field in ["tokenizer_path", "vectors_path", "metadata_path"]:
        setattr(config.embedding, field, resolve(getattr(config.embedding, field)))
    if config.embedding.backend == 'local' and (base / config.embedding.model).is_dir():
        config.embedding.model = str((base / config.embedding.model).resolve())
    config.report.font_path = resolve(config.report.font_path)
    if config.output_root == config.cache_root:
        raise WorkflowError("CONFIG_INVALID", "结果目录与缓存目录必须分开。")
    paths = [Path(i.path) for i in config.inputs]
    if len(set(paths)) != len(paths):
        raise WorkflowError("CONFIG_INVALID", "同一个输入文件不能重复导入。")
    for item in config.inputs:
        file = Path(item.path)
        if not file.is_file() or file.is_symlink():
            raise WorkflowError("INPUT_INVALID", "输入不存在或为符号链接。", details={"file": file.name})
        if file.stat().st_size > config.import_options.max_file_mb * 1024**2:
            raise WorkflowError("INPUT_LIMIT", "输入文件超出配置的大小上限。", details={"file": file.name})
    return config
