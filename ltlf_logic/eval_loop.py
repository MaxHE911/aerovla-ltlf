"""M3+M5 integration: evaluation loop with the LTLf task-logic layer.

This module imports the *unmodified* baseline modules (it must be imported after
``sys.path`` contains the ``AeroVLA`` directory and after the baseline argument parser has
seen the CLI).  It only reads the baseline classes; no baseline file is edited.

Loop shape is the baseline's (``eval_aerovla.eval``) with three additions:

1. ``prepare_inputs`` receives the logic layer's **current guide point** and **current stage
   description** instead of the fixed final target / original description;
2. after every control step the monitor progresses the residual formula (events from the
   simulator's pose ground truth, 12 m radius);
3. the model's LAND is treated as a *phase* termination request: it is forwarded to the
   baseline success logic only when ``accept(residual)``; near-zero actions that would trip the
   baseline stuck detector are replaced by an explicit fallback push toward the current
   sub-goal (M5 L2), always logged.

In ``original`` / ``single_stage`` mode nothing is injected and nothing is gated, so the loop is
behaviour-compatible with the baseline: identical prompts, identical actions, identical
termination rules (verified by comparing per-step positions against a baseline run).

Baseline facts this file relies on (measured, see work/stage0_audit.md):
* ``eval_env.get_obs()`` returns one 4-tuple per environment
  ``(observations, dones, collisions, oracle_success)``; ``observations`` is a list of 5
  identical post-step states (the simulator caps the same state 5 times).
* ``EvalBatchState.update_from_env_output`` already unpacks and book-keeps that output.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.param import args  # noqa: E402  (baseline argument parser, already parsed)
from src.vlnce_src.assist import Assist  # noqa: E402
from src.vlnce_src.closeloop_util import (  # noqa: E402
    BatchIterator,
    EvalBatchState,
    initialize_env_eval,
    save_to_dataset_eval,
)
import src.vlnce_src.closeloop_util as closeloop_util  # noqa: E402

from ltlf_logic.formula import accept, normalize, parse, to_str
from ltlf_logic.monitor import Landmark, Monitor
from ltlf_logic.runner import (
    EpisodeLogger,
    LTLfConfig,
    TaskPlan,
    build_stage_instruction,
    is_near_zero_action,
    next_target_atom,
    should_gate_stop,
    takeover_action,
)
from ltlf_logic.tasks import split_instruction

__all__ = ["run_eval_ltlf", "TaskSource"]


# --------------------------------------------------------------------------------------
# task source: task records for a batch item
# --------------------------------------------------------------------------------------
class TaskSource:
    """Resolves a batch item to its task record (M0 JSON, or a minimal single-goal task)."""

    def __init__(self, task_json: Optional[str], *, require_long: bool):
        self.by_key: Dict[Tuple[str, str], dict] = {}
        self.require_long = require_long
        if task_json:
            with open(task_json, "r") as f:
                tasks = json.load(f)
            for t in tasks:
                if t.get("skipped"):
                    continue
                self.by_key[(t["map"], t["seq"])] = t

    def get(self, batch_item: dict) -> dict:
        key = (batch_item["map_name"], batch_item["seq_name"])
        if key in self.by_key:
            return self.by_key[key]
        if self.require_long:
            raise KeyError("no long-horizon task for %s" % (key,))
        return minimal_single_goal_task(batch_item)


def minimal_single_goal_task(batch_item: dict) -> dict:
    """The original task as a task record: ``F(g)`` with the baseline's own final target."""
    desc = split_instruction(batch_item["instruction"])
    obj = batch_item["object"]
    return {
        "task_id": "%s/%s" % (batch_item["map_name"], batch_item["seq_name"]),
        "map": batch_item["map_name"],
        "seq": batch_item["seq_name"],
        "formula": "F(g)",
        "formula_single_goal": "F(g)",
        "event_order": ["g"],
        "landmarks": [],
        "n_stages": 1,
        "st_rt": "",
        "final_target": {
            "ap": "g",
            "area_index": -1,
            "center": [float(x) for x in batch_item["object_position"]],
            "asset": str(obj["asset_name"]),
            "description": desc,
            "description_source": "own_episode_instruction",
        },
        "long_instruction_zh": "",
        "long_instruction_en": "",
        "generator_version": "minimal",
    }


# --------------------------------------------------------------------------------------
# batch state
# --------------------------------------------------------------------------------------
class LTLfBatchState(EvalBatchState):
    def __init__(
        self,
        batch_size: int,
        env_batchs: List[dict],
        env,
        assist: Assist,
        *,
        config: LTLfConfig,
        task_source: TaskSource,
        logger: EpisodeLogger,
    ):
        super().__init__(batch_size, env_batchs, env, assist)
        self.config = config
        self.task_source = task_source
        self.logger = logger
        self.env_batchs = env_batchs
        self.seq_names = [b["seq_name"] for b in env_batchs]
        self.map_names = [b["map_name"] for b in env_batchs]
        self.task_dicts = [task_source.get(b) for b in env_batchs]

        single = config.mode in ("original", "single_stage", "long_instruction")
        self.plans: List[TaskPlan] = [
            TaskPlan.from_task_dict(t, mode=config.mode, single_goal=single) for t in self.task_dicts
        ]
        self.monitors: List[Monitor] = [
            Monitor(p.formula, p.landmarks, dedupe_repeated_positions=True) for p in self.plans
        ]
        self.task_atoms = [sorted({L.ap for L in p.landmarks}) for p in self.plans]

        # values injected into the baseline wrapper
        self.guide_positions: List[List[float]] = [list(p) for p in self.target_positions]
        self.stage_instructions: List[str] = list(self.instructions)
        if config.mode == "long_instruction":
            for i, t in enumerate(self.task_dicts):
                long_en = t.get("long_instruction_en") or ""
                final_desc = t["final_target"]["description"]
                obj_desc = (long_en + " " + final_desc).strip() if long_en else final_desc
                self.stage_instructions[i] = build_stage_instruction(obj_desc)

        self.current_atom: List[Optional[str]] = [None] * batch_size
        self.stage_idx: List[int] = [0] * batch_size
        self.stage_log: List[List[dict]] = [[] for _ in range(batch_size)]
        self.counters: List[dict] = [
            {
                "model_stops": 0,
                "suppressed_stops": 0,
                "suppressed_stop_zero_action": 0,
                "takeovers": 0,
                "stage_switches": 0,
                "plan_switches": 0,
                "accepted_stops": 0,
                "violations": 0,
                "event_points": 0,
            }
            for _ in range(batch_size)
        ]
        self.finalized = [False] * batch_size
        self.switched_this_step = [False] * batch_size
        self.pending_action: List[Optional[dict]] = [None] * batch_size
        self.step_records: List[List[dict]] = [[] for _ in range(batch_size)]

        # sampling point 0 = the initial observation (baseline appends it in _initialize_batch_data)
        for i in range(batch_size):
            self.monitors[i].step(self._pose(i))
            self._refresh_injection(i, step=0, initial=True)

    # -- helpers ---------------------------------------------------------------
    def _pose(self, i: int) -> Tuple[float, float, float]:
        p = self.episodes[i][-1]["sensors"]["state"]["position"]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _quat(self, i: int) -> Tuple[float, float, float, float]:
        q = self.episodes[i][-1]["sensors"]["state"]["orientation"]
        if isinstance(q, dict):
            return (float(q.get("x", 0)), float(q.get("y", 0)), float(q.get("z", 0)), float(q.get("w", 1)))
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _refresh_injection(self, i: int, *, step: int, initial: bool = False) -> None:
        cfg = self.config
        plan = self.plans[i]
        mon = self.monitors[i]
        if cfg.mode in ("original", "single_stage"):
            self.guide_positions[i] = list(self.target_positions[i])
            self.stage_instructions[i] = self.instructions[i]
            return
        if cfg.mode == "long_instruction":
            self.guide_positions[i] = list(self.target_positions[i])
            return

        # fixed_order: next phase derived from the residual formula
        atom = next_target_atom(mon.residual, plan, preset_atoms=self.task_atoms[i])
        prev_atom = self.current_atom[i]
        if atom is None:
            self.guide_positions[i] = list(plan.final_landmark().center)
            return
        if atom == prev_atom:
            return
        lm = plan.landmark(atom)
        self.guide_positions[i] = [float(c) for c in lm.center]
        desc = plan.descriptions.get(atom) or lm.description
        self.stage_instructions[i] = build_stage_instruction(desc)
        if prev_atom is not None:
            j = plan.stage_index(prev_atom)
            plan.stages[j].end_step = step
            plan.stages[j].entered = True
            self.counters[i]["stage_switches"] += 1
            plan.stage_switches += 1
        self.current_atom[i] = atom
        j = plan.stage_index(atom)
        plan.stages[j].start_step = 0 if initial else step
        self.stage_idx[i] = j
        self.switched_this_step[i] = True
        self.stage_log[i].append(
            {"stage_idx": j, "ap": atom, "start_step": step, "asset": lm.asset, "description": desc}
        )

    def _mark_stage_entry(self, i: int, new_events: Sequence[str]) -> None:
        for ap in new_events:
            try:
                j = self.plans[i].stage_index(ap)
            except KeyError:
                continue
            self.plans[i].stages[j].entered = True

    # -- monitor update --------------------------------------------------------
    def update_ltlf(self, observations: List[list], step: int) -> None:
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            pose = observations[i][-1]["sensors"]["state"]["position"]
            rec = self.monitors[i].step(pose)
            if rec.get("duplicate"):
                continue
            if rec["new_events"]:
                self.counters[i]["event_points"] += 1
                self._mark_stage_entry(i, rec["new_events"])
            if self.monitors[i].violated and self.counters[i]["violations"] == 0:
                self.counters[i]["violations"] += 1
                print("[ltlf] %s: spec violated at step %d (residual=BOT)" % (self.seq_names[i], step))
            self._refresh_injection(i, step=step)

    # -- gating + takeover -----------------------------------------------------
    def gate_and_takeover(
        self, batch_actions: List[dict], model_stops: Sequence[bool], step: int
    ) -> Tuple[List[dict], List[bool]]:
        cfg = self.config
        executed: List[dict] = []
        gated_stops: List[bool] = []
        for i in range(self.batch_size):
            action = dict(batch_actions[i])
            mon = self.monitors[i]
            model_stop = bool(model_stops[i])
            zero = is_near_zero_action(action)
            cause = "zero_action" if zero else ("text_land" if model_stop else "none")
            if model_stop:
                self.counters[i]["model_stops"] += 1

            allow, reason = should_gate_stop(model_stop, mon.residual, cfg.gate_enabled)
            if model_stop and allow:
                self.counters[i]["accepted_stops"] += 1
            if model_stop and not allow:
                self.counters[i]["suppressed_stops"] += 1
                if zero:
                    self.counters[i]["suppressed_stop_zero_action"] += 1

            takeover = False
            takeover_reason = ""
            if cfg.takeover_enabled and not accept(mon.residual) and zero:
                # M5 L2: a hovering near-zero action would trip the baseline stuck detector
                action = takeover_action(self._pose(i), self._quat(i), self.guide_positions[i], cfg)
                takeover = True
                takeover_reason = "near_zero_action_unfinished_residual"
                self.counters[i]["takeovers"] += 1

            executed.append(action)
            gated_stops.append(bool(allow))
            self.pending_action[i] = {
                "step": step,
                "model_stop": model_stop,
                "model_stop_cause": cause,
                "gated_stop": bool(allow),
                "gate_reason": reason,
                "takeover": takeover,
                "takeover_reason": takeover_reason,
                "action_model": {k: round(float(v), 4) for k, v in batch_actions[i].items()},
                "action_executed": {k: round(float(v), 4) for k, v in action.items()},
            }
        return executed, gated_stops

    # -- per-step record (written after the observation arrived) ---------------
    def record_step(self, step: int) -> None:
        cfg = self.config
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            pending = self.pending_action[i] or {}
            mon = self.monitors[i]
            last = mon.event_log[-1] if mon.event_log else None
            pose = self._pose(i)
            guide = [float(v) for v in self.guide_positions[i]]
            rec = {
                "seq": self.seq_names[i],
                "task_id": self.plans[i].task_id,
                "map": self.map_names[i],
                "mode": cfg.mode,
                "step": step,
                "stage_idx": self.stage_idx[i],
                "stage_ap": self.current_atom[i],
                "position": [round(v, 4) for v in pose],
                "guide_point": [round(v, 4) for v in guide],
                "stage_description": _current_stage_desc(self.stage_instructions[i]),
                "events": sorted(last["events"]) if last else [],
                "new_events": sorted(last["new_events"]) if last else [],
                "residual": to_str(mon.residual),
                "accepted": accept(mon.residual),
                "violated": mon.violated,
                "model_stop": pending.get("model_stop"),
                "model_stop_cause": pending.get("model_stop_cause"),
                "gated_stop": pending.get("gated_stop"),
                "gate_reason": pending.get("gate_reason"),
                "takeover": pending.get("takeover"),
                "takeover_reason": pending.get("takeover_reason"),
                "action_model": pending.get("action_model"),
                "action_executed": pending.get("action_executed"),
                "distance_to_final": round(float(self.distance_to_ends[i][-1]), 3),
                "distance_to_guide": round(math.dist(pose, guide), 3),
                "stage_switch": self.switched_this_step[i],
                "ltlf_switches": self.counters[i]["stage_switches"],
            }
            self.step_records[i].append(rec)
            self.logger.write_step(self.seq_names[i], rec)
            self.switched_this_step[i] = False
            self.pending_action[i] = None

    # -- finalisation ----------------------------------------------------------
    def finalize_done(self, t: int) -> None:
        for i in range(self.batch_size):
            if not self.skips[i] or self.finalized[i]:
                continue
            self.finalized[i] = True
            plan = self.plans[i]
            mon = self.monitors[i]
            if self.current_atom[i] is not None:
                j = plan.stage_index(self.current_atom[i])
                plan.stages[j].end_step = t
            summary = {
                "seq": self.seq_names[i],
                "task_id": plan.task_id,
                "map": self.map_names[i],
                "mode": self.config.mode,
                "formula": to_str(plan.formula),
                "st_rt": plan.st_rt,
                "n_stages": plan.n_stages,
                "stages": [s.to_dict() for s in plan.stages],
                "stage_switches": self.counters[i]["stage_switches"],
                "steps": t,
                "success": bool(self.success[i]),
                "oracle_success": bool(self.oracle_success[i]),
                "collision": bool(self.collisions[i]),
                "early_end": bool(self.early_end[i]),
                "esr_accepted": accept(mon.residual),
                "esr_residual": to_str(mon.residual),
                "esr_violated": mon.violated,
                "final_distance_to_target": round(float(self.distance_to_ends[i][-1]), 3),
                "events_order": [e for r in mon.event_log for e in r["new_events"]],
                "sampling_points": mon.step_index,
                "model_stops": self.counters[i]["model_stops"],
                "suppressed_stops": self.counters[i]["suppressed_stops"],
                "suppressed_stop_zero_action": self.counters[i]["suppressed_stop_zero_action"],
                "takeovers": self.counters[i]["takeovers"],
                "accepted_stops": self.counters[i]["accepted_stops"],
                "violations": self.counters[i]["violations"],
                "final_target": {
                    "center": [float(x) for x in self.task_dicts[i]["final_target"]["center"]],
                    "asset": self.task_dicts[i]["final_target"]["asset"],
                    "description": self.task_dicts[i]["final_target"]["description"],
                },
                "landmarks": [L.to_dict() for L in plan.landmarks],
                "config": self.config.to_dict(),
            }
            summary["sr_and_esr"] = bool(summary["success"]) and bool(summary["esr_accepted"])
            self.logger.write_summary(summary)


def _current_stage_desc(instruction: str) -> str:
    try:
        return split_instruction(instruction)
    except Exception:
        return ""


# --------------------------------------------------------------------------------------
# evaluation driver
# --------------------------------------------------------------------------------------
def run_eval_ltlf(
    model_wrapper,
    assist: Assist,
    eval_env,
    eval_save_dir: str,
    *,
    config: LTLfConfig,
    log_dir: str,
) -> dict:
    logger = EpisodeLogger(log_dir)
    task_source = TaskSource(config.task_json, require_long=config.mode == "fixed_order")

    if config.disable_distance_increasing:
        closeloop_util.target_distance_increasing_for_10frames = lambda lst: False
        print("[ltlf] protocol: 'distance increasing for 10 frames' heuristic DISABLED")
    else:
        print("[ltlf] protocol: 'distance increasing for 10 frames' heuristic ENABLED (baseline)")

    steps_done = 0
    n_batches = 0
    import torch

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            n_batches += 1
            for batch_item in env_batchs:
                print(" [New Mission] ---------------------------------------------")
                print(' Instruction: "%s"' % batch_item["instruction"][:160])
                print(" Target Object: %s" % batch_item["object"]["asset_name"])

            state = LTLfBatchState(
                batch_size=eval_env.batch_size,
                env_batchs=env_batchs,
                env=eval_env,
                assist=assist,
                config=config,
                task_source=task_source,
                logger=logger,
            )
            inputs, _ = model_wrapper.prepare_inputs(
                state.episodes, state.guide_positions, instructions=state.stage_instructions
            )
            for t in range(int(args.maxWaypoints) + 1):
                terminated = state.check_batch_termination(t)
                state.finalize_done(t)
                if terminated:
                    break

                batch_actions, model_stops = model_wrapper.run(
                    inputs=inputs, episodes=state.episodes, rot_to_targets=None
                )
                actions, gated_stops = state.gate_and_takeover(batch_actions, model_stops, t)
                eval_env.makeActions(actions)
                time.sleep(0.01)
                outputs = eval_env.get_obs()
                state.update_from_env_output(outputs)
                observations = [list(x)[0] for x in zip(*outputs)]
                state.update_ltlf(observations, t)
                state.predict_dones = gated_stops
                state.update_metric()
                state.record_step(t)
                steps_done += eval_env.batch_size
                inputs, _ = model_wrapper.prepare_inputs(
                    state.episodes, state.guide_positions, instructions=state.stage_instructions
                )

    index_path = logger.flush_index()
    return {
        "mode": config.mode,
        "config": config.to_dict(),
        "batches": n_batches,
        "episodes": len(logger.summaries),
        "steps": steps_done,
        "log_dir": os.path.abspath(log_dir),
        "eval_save_dir": os.path.abspath(eval_save_dir),
        "index": index_path,
        "max_waypoints": int(args.maxWaypoints),
        "batch_size": int(args.batchSize),
    }
