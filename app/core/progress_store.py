"""Thread-safe progress store for batch OCR jobs.

Lives in the main process. The tqdm patch writes here on every
vLLM prompt completion. The SSE endpoint reads here to stream
updates to Orbst.
"""
import threading
import time
from typing import Optional, Literal
from pydantic import BaseModel, Field

JOB_TTL_SECONDS = 600


class ProgressState(BaseModel):
    job_id: str
    document_id: str = ""
    total_pages: int = 0
    phase: Literal["queued", "decoding", "infer_layout", "infer_text", "done", "failed"] = "queued"
    phase_completed: int = 0
    phase_total: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None


class ProgressStore:
    _instance: Optional["ProgressStore"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ProgressStore":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._jobs: dict[str, ProgressState] = {}
                    inst._jobs_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    def init_job(self, job_id: str, document_id: str, total_pages: int) -> ProgressState:
        state = ProgressState(
            job_id=job_id, document_id=document_id,
            total_pages=total_pages, phase="decoding",
        )
        with self._jobs_lock:
            self._jobs[job_id] = state
        return state

    def set_phase(self, job_id: str, phase: str, phase_total: int = 0) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.phase = phase  # type: ignore[assignment]
                job.phase_total = phase_total
                job.phase_completed = 0

    def update(self, job_id: str, completed: int, total: int) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.phase_completed = completed
                job.phase_total = total

    def complete(self, job_id: str, result: dict) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.phase = "done"  # type: ignore[assignment]
                job.result = result
                job.completed_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.phase = "failed"  # type: ignore[assignment]
                job.error = error
                job.completed_at = time.time()

    def get(self, job_id: str) -> Optional[ProgressState]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def cleanup_expired(self) -> None:
        now = time.time()
        with self._jobs_lock:
            expired = [
                jid for jid, job in self._jobs.items()
                if job.completed_at and (now - job.completed_at) > JOB_TTL_SECONDS
            ]
            for jid in expired:
                del self._jobs[jid]


progress_store = ProgressStore()
