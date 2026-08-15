"""Monkey-patch tqdm.tqdm.update() to intercept vLLM progress bars.

MUST be imported BEFORE any vllm import. vLLM's SyncMPClient creates
tqdm instances in the main process and calls .update() via IPC from
the EngineCore subprocess. By patching .update(), we capture every
prompt completion and write it to progress_store.

ContextVar is used instead of threadlocal because asyncio.to_thread()
propagates contextvars by default (Python 3.9+), and we need the
job_id to flow into the threadpool where batch_two_step_extract runs.
"""
import contextvars
from tqdm import tqdm as _Tqdm

_current_job_id: contextvars.ContextVar[str] = contextvars.ContextVar("procr_job_id", default="")
_original_update = _Tqdm.update


def _patched_update(self: _Tqdm, n: int = 1):  # type: ignore[override]
    result = _original_update(self, n)
    job_id = _current_job_id.get()
    if job_id and self.total and self.total > 0:
        from app.core.progress_store import progress_store
        progress_store.update(job_id, completed=self.n, total=self.total)
    return result


_Tqdm.update = _patched_update  # type: ignore[assignment]


def set_job_id(job_id: str) -> contextvars.Token[str]:
    return _current_job_id.set(job_id)


def reset_job_id(token: contextvars.Token[str]) -> None:
    _current_job_id.reset(token)
