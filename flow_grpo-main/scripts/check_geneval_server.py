#!/usr/bin/env python3
"""Fail fast when the local GenEval reward service is unavailable."""

import socket
import sys


def main() -> int:
    # Keep this aligned with flow_grpo.rewards.geneval_score, whose endpoint is
    # intentionally left untouched so existing FlowGRPO behavior does not change.
    host = "127.0.0.1"
    port = 18085

    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError as error:
        print(
            f"ERROR: GenEval reward server is unavailable at "
            f"http://{host}:{port} ({error}).\n"
            "Start it in a separate terminal from the reward-server repository:\n"
            "  gunicorn \"app_geneval:create_app()\"\n"
            "Then rerun this launcher.",
            file=sys.stderr,
        )
        return 1

    print(f"GenEval reward server is reachable at http://{host}:{port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
