"""Per-rank, durable wall-clock timing for the profiled FLUX entry point."""
import contextlib
import csv
import json
import os
from pathlib import Path
import time


class PhaseTimer:
    def __init__(self):
        import torch
        self.torch = torch
        self.sync = os.environ.get("FLOW_TIMING_SYNC", "1") == "1"
        self.rank = int(os.environ.get("RANK", "0"))
        self.root = Path(os.environ.get("FLOW_TIMING_DIR", "logs/flux_timing"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.epoch = -1
        self.totals = {}
        self.file = (self.root / f"phases_rank{self.rank}.csv").open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow(["epoch", "phase", "seconds", "elapsed_seconds", "timestep", "completed"])
        self.file.flush()

    def synchronize(self):
        if self.sync and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    @contextlib.contextmanager
    def phase(self, name, timestep=None):
        self.synchronize()
        start = time.perf_counter()
        completed = False
        try:
            yield
            completed = True
        finally:
            self.synchronize()
            duration = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + duration
            self.writer.writerow([self.epoch, name, duration, time.perf_counter() - self.started, timestep, completed])
            self.file.flush()

    @contextlib.contextmanager
    def iteration(self, epoch):
        self.epoch = epoch
        self.totals = {}
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            total = time.perf_counter() - start
            accounted = sum(self.totals.values())
            phases = dict(self.totals)
            phases["other"] = max(0.0, total - accounted)
            record = {"epoch": epoch, "rank": self.rank, "epoch_seconds": total,
                      "elapsed_seconds": time.perf_counter() - self.started,
                      "synchronized": self.sync,
                      "phases": {key: {"seconds": value, "percent": 100 * value / total if total else 0}
                                 for key, value in phases.items()}}
            with (self.root / f"epochs_rank{self.rank}.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")

    def log(self, values, *args, **kwargs):
        import wandb
        rewards = {}
        for key, value in values.items():
            if "reward" in key.lower():
                try:
                    rewards[key] = float(value)
                except (TypeError, ValueError):
                    pass
        if rewards:
            with (self.root / f"rewards_rank{self.rank}.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"epoch": self.epoch, "elapsed_seconds": time.perf_counter() - self.started,
                                      "step": kwargs.get("step"), **rewards}) + "\n")
        return wandb.log(values, *args, **kwargs)
