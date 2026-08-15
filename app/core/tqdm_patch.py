"""Monkey-patch tqdm.tqdm.update() to intercept vLLM progress bars.

MUST be imported BEFORE any vllm import. vLLM's SyncMPClient creates
tqdm instances in the main process and calls .update() via IPC from
the EngineCore subprocess. By patching .update(), we capture every
prompt completion and write it to progress_store.
"""
import contextvars
from tqdm import tqdm as _Tqdm

_current_job_id: contextvars.ContextVar[str] = contextvars.ContextVar("procr_job_id", default="")
_original_update = _Tqdm.update

# Track the previous total per job to detect phase transitions.
# Phase 1 (layout): total = page_count (e.g., 4)
# Phase 2 (text): total = block_count (e.g., 126 — much larger)
# When total jumps > 3x, we auto-transition from infer_layout → infer_text.
_last_total: dict[str, int] = {}


def _patched_update(self: _Tqdm, n: int = 1):  # type: ignore[override]
    result = _original_update(self, n)
    job_id = _current_job_id.get()
    if job_id and self.total and self.total > 0:
        from app.core.progress_store import progress_store

        prev = _last_total.get(job_id, 0)
        curr = int(self.total)
        _last_total[job_id] = curr

        if prev > 0 and curr > prev * 3:
            progress_store.set_phase(job_id, "infer_text", phase_total=curr)

        progress_store.update(job_id, completed=int(self.n), total=curr)
    return result


_Tqdm.update = _patched_update  # type: ignore[assignment]


def set_job_id(job_id: str) -> None:
    _current_job_id.set(job_id)
    _last_total.pop(job_id, None)


def clear_job_id() -> None:
    job_id = _current_job_id.get()
    if job_id:
        _last_total.pop(job_id, None)
    _current_job_id.set("")
