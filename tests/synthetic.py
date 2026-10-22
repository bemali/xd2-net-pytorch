"""Synthetic event-log generator for tests and multi-dataset smoke runs.

The log uses deliberately non-standard column names and mixes case- and
event-level variables, so it exercises the dataset-config mapping:

    case_id, ts, task, team               core columns
    channel (cat), amount (num)           case-level
    priority (cat), cost (num)            event-level
    result                                outcome column ("won" / "lost")

The outcome depends on whether the case contains "Escalate", on ``priority``
and on ``amount``, so a model can learn it. Every case ends with "Close_Won"
or "Close_Lost", so the same log also works with ``outcome.mode: activity``.

Run as a script to write a CSV and a matching dataset YAML:

    python tests/synthetic.py --out <dir> [--cases 600]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

ACTIVITIES = ["Register", "Check", "Review", "Escalate", "Approve", "Notify"]
TEAMS = ["front", "back", "audit"]


def make_log(n_cases: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-01-01 08:00")
    for c in range(n_cases):
        n_events = int(rng.integers(4, 13))
        channel = rng.choice(["web", "branch", "phone"])
        amount = float(rng.lognormal(9, 0.6))
        t = start + pd.Timedelta(hours=float(rng.uniform(0, 24 * 90)))
        escalated = False
        high_prio = 0
        for e in range(n_events):
            act = "Register" if e == 0 else rng.choice(ACTIVITIES[1:])
            escalated |= act == "Escalate"
            prio = rng.choice(["low", "high"], p=[0.7, 0.3])
            high_prio += prio == "high"
            rows.append({
                "case_id": f"C{c:05d}",
                "ts": t,
                "task": act,
                "team": rng.choice(TEAMS),
                "channel": channel,
                "amount": amount,
                "priority": prio,
                "cost": float(rng.gamma(2.0, 10.0)),
            })
            t += pd.Timedelta(minutes=float(rng.exponential(240)))
        score = 1.5 * escalated + 0.6 * high_prio - 0.8 * (amount > 12000) + rng.normal(0, 0.7)
        won = score > 1.0
        rows.append({
            "case_id": f"C{c:05d}", "ts": t, "task": "Close_Won" if won else "Close_Lost",
            "team": "back", "channel": channel, "amount": amount, "priority": "low", "cost": 0.0,
        })
        for r in rows[-(n_events + 1):]:
            r["result"] = "won" if won else "lost"
    df = pd.DataFrame(rows)
    # Shuffle rows so the loader must sort by case + timestamp.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def dataset_dict(path: str, mode: str = "column", name: str = "synthetic") -> dict:
    outcome = (
        {"mode": "column", "column": "result", "positive": "won", "labels": ["lost", "won"]}
        if mode == "column"
        else {"mode": "activity", "activities": ["Close_Won"], "labels": ["lost", "won"]}
    )
    return {
        "name": name,
        "path": path,
        "columns": {"caseid": "case_id", "timestamp": "ts", "activity": "task", "role": "team"},
        # Terminal events reveal the outcome; drop them (activity mode cuts Close_Won itself).
        "drop_activities": ["Close_Won", "Close_Lost"] if mode == "column" else ["Close_Lost"],
        "event_cat": ["priority"],
        "event_num": ["cost"],
        "case_cat": ["channel"],
        "case_num": ["amount"],
        "outcome": outcome,
        "time_features": {"since_prev": True, "since_start": True, "hour": True, "weekday": False},
        "norm": {"default": "max", "overrides": {"TIME_SINCE_PREV": "lognorm", "amount": "lognorm"}},
        "prefix": {"lengths": "auto", "min_len": 2, "max_len": 8, "min_cases": 10},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["column", "activity"], default="column")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "synthetic_log.csv")
    make_log(args.cases, args.seed).to_csv(csv_path, index=False)
    yml = os.path.join(args.out, f"synthetic_{args.mode}.yaml")
    with open(yml, "w") as f:
        yaml.safe_dump(dataset_dict(csv_path, args.mode, name=f"synthetic_{args.mode}"), f, sort_keys=False)
    print(csv_path)
    print(yml)


if __name__ == "__main__":
    main()
