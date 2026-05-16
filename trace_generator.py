"""
Synthetic trace generator for bootstrapping the AMN Markov model.
Generates realistic real estate workflow traces before any real user runs.

Usage: python amn.py --bootstrap
"""

import random
import json
from amn import (
    BuyerContext, StepLog, TOOL_COSTS, STATES,
    update_markov, save_rank_file, seed_initial_buckets,
)

BUYER_PROFILES = [
    {
        "profile": "first_time_buyer",
        "budget": "$350,000",
        "location": "Pittsburgh, PA",
        "urgency": "high",
        "buyer_intent": "First home purchase, moving for new job",
        "ideal_path": [
            ("UNDERSTAND_BUYER",  "ask_clarifying_question"),
            ("QUALIFY_LEAD",      "qualify_lead"),
            ("SEARCH_OPTIONS",    "search_listings"),
            ("FILTER_OPTIONS",    "filter_listings"),
            ("RANK_OPTIONS",      "rank_listings"),
            ("CONTACT_BUYER",     "send_message"),
            ("SCHEDULE_TOUR",     "schedule_tour"),
            ("UPDATE_CRM",        "update_crm"),
        ],
    },
    {
        "profile": "investor",
        "budget": "$800,000",
        "location": "Suburbs of Pittsburgh",
        "urgency": "low",
        "buyer_intent": "Multi-unit investment property, cash buyer",
        "ideal_path": [
            ("QUALIFY_LEAD",   "qualify_lead"),
            ("SEARCH_OPTIONS", "search_listings"),
            ("FILTER_OPTIONS", "filter_listings"),
            ("RANK_OPTIONS",   "rank_listings"),
            ("CONTACT_BUYER",  "send_message"),
            ("UPDATE_CRM",     "update_crm"),
        ],
    },
    {
        "profile": "relocation",
        "budget": "$1,200,000",
        "location": "Boston, MA",
        "urgency": "high",
        "buyer_intent": "Corporate relocation, 4BR, good school district",
        "ideal_path": [
            ("UNDERSTAND_BUYER", "ask_clarifying_question"),
            ("QUALIFY_LEAD",     "qualify_lead"),
            ("SEARCH_OPTIONS",   "search_listings"),
            ("RANK_OPTIONS",     "rank_listings"),
            ("CONTACT_BUYER",    "send_message"),
            ("SCHEDULE_TOUR",    "schedule_tour"),
            ("UPDATE_CRM",       "update_crm"),
        ],
    },
    {
        "profile": "upgrade",
        "budget": "$600,000",
        "location": "Squirrel Hill, Pittsburgh",
        "urgency": "medium",
        "buyer_intent": "Upgrading from current home, family growing",
        "ideal_path": [
            ("UNDERSTAND_BUYER", "ask_clarifying_question"),
            ("QUALIFY_LEAD",     "qualify_lead"),
            ("SEARCH_OPTIONS",   "search_listings"),
            ("FILTER_OPTIONS",   "filter_listings"),
            ("RANK_OPTIONS",     "rank_listings"),
            ("SCHEDULE_TOUR",    "schedule_tour"),
            ("UPDATE_CRM",       "update_crm"),
        ],
    },
]

# State → tool actions that could plausibly be chosen at each state
PLAUSIBLE_ALTERNATES = {
    "UNDERSTAND_BUYER":  ["ask_clarifying_question", "extract_requirements"],
    "QUALIFY_LEAD":      ["qualify_lead", "ask_clarifying_question"],
    "SEARCH_OPTIONS":    ["search_listings", "ask_clarifying_question"],
    "FILTER_OPTIONS":    ["filter_listings", "search_listings"],
    "RANK_OPTIONS":      ["rank_listings", "filter_listings"],
    "CONTACT_BUYER":     ["send_message", "check_calendar"],
    "SCHEDULE_TOUR":     ["schedule_tour", "check_calendar", "send_message"],
    "UPDATE_CRM":        ["update_crm", "send_message"],
    "RECOVER":           ["ask_clarifying_question", "extract_requirements"],
    "FOLLOW_UP":         ["send_message", "check_calendar"],
}


def _make_step_log(step_num: int, state: str, action: str, next_state: str,
                   success: bool, llm_action: str = None) -> StepLog:
    base_cost = TOOL_COSTS.get(action, 0.02)
    cost = base_cost * random.uniform(0.8, 1.2)
    return StepLog(
        step=step_num,
        current_state=state,
        llm_action=llm_action or action,
        markov_action=None,
        chosen_action=action,
        confidence=0.0,
        next_state=next_state,
        tool_cost=cost,
        success=success,
    )


def generate_trace(profile: dict, noise_level: float = 0.15, seed: int = None) -> list:
    """
    Generate a synthetic StepLog trace following the ideal path with noise.
    noise_level: probability of inserting a wrong step (causing RECOVER) at each step.
    """
    if seed is not None:
        random.seed(seed)

    ideal_path = profile["ideal_path"]
    logs = []
    step_num = 1
    i = 0

    while i < len(ideal_path) and step_num <= 20:
        state, action = ideal_path[i]
        next_state = ideal_path[i + 1][0] if i + 1 < len(ideal_path) else "DONE"

        # Random noise: insert a wrong action that causes recovery
        if random.random() < noise_level and step_num > 1:
            alternates = PLAUSIBLE_ALTERNATES.get(state, [action])
            wrong_action = random.choice([a for a in alternates if a != action] or [action])
            # Wrong action fails with 60% chance
            if random.random() < 0.6:
                logs.append(_make_step_log(step_num, state, wrong_action, "RECOVER", False, action))
                step_num += 1
                # Recover step
                logs.append(_make_step_log(step_num, "RECOVER", "ask_clarifying_question", state, True, action))
                step_num += 1
                continue  # retry same state

        # Execute the correct action
        # Slight failure chance even for correct actions
        success = random.random() > 0.05
        actual_next = next_state if success else "RECOVER"

        logs.append(_make_step_log(step_num, state, action, actual_next, success, action))
        step_num += 1

        if not success:
            # Recover and retry
            logs.append(_make_step_log(step_num, "RECOVER", "ask_clarifying_question", state, True, action))
            step_num += 1
        else:
            i += 1

    # Final DONE step
    if i >= len(ideal_path):
        logs.append(_make_step_log(step_num, "UPDATE_CRM", "update_crm", "DONE", True))

    return logs


def generate_all_traces(n_per_profile: int = 10) -> list:
    all_traces = []
    for profile in BUYER_PROFILES:
        for i in range(n_per_profile):
            trace = generate_trace(profile, noise_level=0.15, seed=hash(profile["profile"]) + i)
            all_traces.append(trace)
    return all_traces


def bootstrap_rank_file(embed_model):
    """Generate synthetic traces, build Markov policy, load into in-memory store."""
    print("[bootstrap] Generating synthetic training traces...")
    traces = generate_all_traces(n_per_profile=10)
    print(f"[bootstrap] Generated {len(traces)} traces ({len(BUYER_PROFILES)} profiles × 10 each)")

    policy = {}
    for trace in traces:
        policy = update_markov(policy, trace, task_success=True)

    total_pairs = sum(len(v) for v in policy.items())
    print(f"[bootstrap] Learned policy: {len(policy)} states, {sum(len(v) for v in policy.values())} state-action pairs")

    buckets = seed_initial_buckets(embed_model)
    save_rank_file(buckets, policy, {"source": "synthetic_bootstrap"})
    print(f"[bootstrap] In-memory store loaded with {len(traces)} training runs worth of data")
    print("[bootstrap] Warm run will benefit immediately from this policy")

    # Show a sample of the learned policy
    print("\n[bootstrap] Sample learned policy:")
    for state in list(policy.keys())[:4]:
        for action, entry in list(policy[state].items())[:2]:
            print(f"  {state} → {action}: success_rate={entry['success_rate']:.2f}, count={entry['count']}")
