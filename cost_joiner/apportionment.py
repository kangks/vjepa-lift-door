"""Apportionment of a workload's daily cost onto runs by busy-seconds.

Adapted from <LOCAL_PATH>/Documents/workplace/attributes/app/cost_joiner/apportionment.py
(methodology doc section 4.2). Both views are computed, never just one:

  fully absorbed:  share(r) = C_day * d(r) / sum_r' d(r')
  marginal:        share(r) = C_day * d(r) / T_billed
  idle residual:   C_day - sum_r marginal(r)

One deliberate change for lda. The original divides by 86,400 s because the
luminara services ran desired_count=1 all day. lda services sit at desired_count=0
and are scaled up per pass, so the capacity a run competes for is the billed
TASK time of that workload on that day (T_billed, from ECS task start/stop times),
not the wall-clock day. With T_billed = 86,400 s the formula reduces to the original.

Overlap: the lda coordinator processes one SQS message at a time and the embed
service serialises on an asyncio lock, so per-run busy intervals on one task do not
overlap and sum(d) <= T_billed holds. apportion() raises if it does not, rather
than silently creating negative idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Apportionment:
    workload: str
    daily_cost_usd: float
    billed_task_seconds: float
    busy_seconds_total: float
    absorbed_by_run: dict[str, float] = field(default_factory=dict)
    marginal_by_run: dict[str, float] = field(default_factory=dict)
    idle_residual_usd: float = 0.0
    unallocated: bool = False

    @property
    def utilisation(self) -> float:
        return self.busy_seconds_total / self.billed_task_seconds if self.billed_task_seconds else 0.0


def apportion(workload: str, daily_cost_usd: float, billed_task_seconds: float,
              busy_seconds_by_run: dict[str, float], *, tolerance_s: float = 1.0) -> Apportionment:
    total = sum(busy_seconds_by_run.values())
    result = Apportionment(workload, daily_cost_usd, billed_task_seconds, total)
    if any(v < 0 for v in busy_seconds_by_run.values()):
        raise ValueError(f"{workload}: negative busy seconds")
    if not busy_seconds_by_run or total == 0:
        # No runs that day: the whole cost is an explicit unallocated line.
        result.unallocated = True
        result.idle_residual_usd = daily_cost_usd
        return result
    if billed_task_seconds <= 0:
        raise ValueError(f"{workload}: runs have busy time but billed task time is {billed_task_seconds}")
    if total > billed_task_seconds + tolerance_s:
        raise ValueError(f"{workload}: busy {total:.1f}s exceeds billed task time {billed_task_seconds:.1f}s; "
                         "overlapping runs or missing task records")
    marginal_sum = 0.0
    for run_id, d in busy_seconds_by_run.items():
        result.absorbed_by_run[run_id] = daily_cost_usd * d / total
        m = daily_cost_usd * d / billed_task_seconds
        result.marginal_by_run[run_id] = m
        marginal_sum += m
    result.idle_residual_usd = daily_cost_usd - marginal_sum
    return result


def even_split(cost_usd: float, run_ids: list[str]) -> dict[str, float]:
    """Shared overhead with no per-run handle (the batch driver's Step Functions
    transitions, step/watchdog Lambdas, scheduler fire): split evenly over the runs
    it served, rather than absorbing it silently into some other line."""
    if not run_ids:
        return {}
    share = cost_usd / len(run_ids)
    return {r: share for r in run_ids}
