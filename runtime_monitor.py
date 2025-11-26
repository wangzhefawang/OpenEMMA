import json
import time
from typing import Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None


class RuntimeMonitor:
    """Collect runtime duration and GPU memory usage for the current process."""

    def __init__(self, run_args: Optional[Dict] = None) -> None:
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None
        self._gpu_samples_mb: list[float] = []
        self._run_args: Optional[Dict] = run_args  # 保存运行参数

    def start(self) -> None:
        if self._start_ts is None:
            self._start_ts = time.perf_counter()

    def record_gpu_usage(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        current_device = torch.cuda.current_device()
        usage_bytes = torch.cuda.memory_allocated(current_device)
        self._gpu_samples_mb.append(usage_bytes / (1024 ** 2))

    def finish(self) -> Dict[str, Optional[float]]:
        if self._end_ts is None:
            self._end_ts = time.perf_counter()
        total_runtime = None
        if self._start_ts is not None:
            total_runtime = self._end_ts - self._start_ts
        avg_gpu = (
            sum(self._gpu_samples_mb) / len(self._gpu_samples_mb)
            if self._gpu_samples_mb
            else None
        )
        result = {
            "total_runtime_sec": total_runtime,
            "avg_gpu_memory_mb": avg_gpu,
            "gpu_samples_count": len(self._gpu_samples_mb),
        }
        # 如果有运行参数，也包含进去
        if self._run_args is not None:
            result["run_args"] = self._run_args
        return result

    def dump(self, file_path: str) -> None:
        summary = self.finish()
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)


