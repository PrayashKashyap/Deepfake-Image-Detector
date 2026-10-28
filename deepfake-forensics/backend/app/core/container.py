"""Application service container, created once per app instance."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.database.db import Database
from app.database.repositories import AnalysisRepository, JobRepository, SettingsRepository
from app.services.analysis import AnalysisService
from app.services.evaluation_results import EvaluationResults
from app.services.jobs import JobManager
from app.services.model_registry import ModelRegistry


@dataclass
class Container:
    settings: Settings
    db: Database
    analyses_repo: AnalysisRepository
    settings_repo: SettingsRepository
    registry: ModelRegistry
    analyses: AnalysisService
    evaluations: EvaluationResults
    jobs: JobManager

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        settings.ensure_dirs()
        db = Database(settings.database_path)
        analyses_repo = AnalysisRepository(db)
        settings_repo = SettingsRepository(db)
        registry = ModelRegistry(settings, settings_repo)
        return cls(
            settings=settings,
            db=db,
            analyses_repo=analyses_repo,
            settings_repo=settings_repo,
            registry=registry,
            analyses=AnalysisService(settings, registry, analyses_repo),
            evaluations=EvaluationResults(settings),
            jobs=JobManager(JobRepository(db), settings.job_workers),
        )


def get_container(request: Request) -> Container:
    return request.app.state.container
