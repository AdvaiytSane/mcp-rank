"""
OpenHome Agent Memory Network (AMN)
Real estate lead conversion agent with Markov policy learning.

Usage:
    python amn.py               # Run cold + warm demo
    python amn.py --bootstrap   # Generate synthetic training traces first
    python amn.py --lifelogger  # Connect to live LifeLogger instance
"""

import json
import random
import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES = [
    "START", "UNDERSTAND_BUYER", "QUALIFY_LEAD", "SEARCH_OPTIONS",
    "FILTER_OPTIONS", "RANK_OPTIONS", "CONTACT_BUYER", "SCHEDULE_TOUR",
    "UPDATE_CRM", "FOLLOW_UP", "DONE", "RECOVER",
]

CONFIDENCE_THRESHOLD = 0.70
MAX_STEPS = 20

# In-memory store — persists for the lifetime of the process (across multiple calls)
_MEMORY_STORE: dict = {"version": 1, "total_runs": 0, "buckets": [], "global_policy": {}}

# OpenRouter config — key read inside llm_call() to avoid module-level os import
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.0-flash-001"

# ANSI colors
RED   = "\033[91m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"

# Tool costs (mock)
TOOL_COSTS = {
    "extract_requirements":    0.02,
    "qualify_lead":            0.03,
    "search_listings":         0.05,
    "filter_listings":         0.03,
    "rank_listings":           0.04,
    "send_message":            0.02,
    "check_calendar":          0.01,
    "schedule_tour":           0.03,
    "update_crm":              0.02,
    "ask_clarifying_question": 0.02,
}

# State → canonical action mapping (ideal workflow)
IDEAL_ACTIONS = {
    "UNDERSTAND_BUYER":  "ask_clarifying_question",
    "QUALIFY_LEAD":      "qualify_lead",
    "SEARCH_OPTIONS":    "search_listings",
    "FILTER_OPTIONS":    "filter_listings",
    "RANK_OPTIONS":      "rank_listings",
    "CONTACT_BUYER":     "send_message",
    "SCHEDULE_TOUR":     "schedule_tour",
    "UPDATE_CRM":        "update_crm",
    "START":             "extract_requirements",
    "RECOVER":           "ask_clarifying_question",
    "FOLLOW_UP":         "send_message",
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LifeLoggerContext:
    conversation_summary: str
    recent_speech: list = field(default_factory=list)
    history_chunks: list = field(default_factory=list)
    voice_profiles: dict = field(default_factory=dict)
    room_state: dict = field(default_factory=dict)


@dataclass
class BuyerContext:
    raw_conversation: str
    buyer_intent: str = ""
    budget: str = ""
    location: str = ""
    urgency: str = ""
    listings: list = field(default_factory=list)
    qualified: bool = False
    ranked: bool = False
    contacted: bool = False
    tour_scheduled: bool = False
    crm_updated: bool = False


@dataclass
class StepLog:
    step: int
    current_state: str
    llm_action: str
    markov_action: Optional[str]
    chosen_action: str
    confidence: float
    next_state: str
    tool_cost: float
    success: bool

# ---------------------------------------------------------------------------
# LifeLogger Adapters
# ---------------------------------------------------------------------------

def from_lifelogger(ll) -> LifeLoggerContext:
    return LifeLoggerContext(
        conversation_summary=ll.conversation_summary,
        recent_speech=ll.recent_speech[-20:],
        history_chunks=ll.history_chunks,
        voice_profiles=dict(ll.voice_profiles),
        room_state=ll.room_state,
    )


def from_text(raw: str) -> LifeLoggerContext:
    return LifeLoggerContext(
        conversation_summary=raw,
        recent_speech=[f"[00:00:00] Speaker 0: {raw}"],
        history_chunks=[{"summary": raw, "topics": [], "raw_transcript": raw}],
        voice_profiles={},
        room_state={},
    )

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_text(text: str, model: SentenceTransformer) -> list:
    return model.encode(text).tolist()


def build_embedding_text(ctx: BuyerContext, state: str) -> str:
    return f"{ctx.buyer_intent} {ctx.location} {ctx.budget} {ctx.urgency} {state}"


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------------------------------------------------------------------------
# Bucket / Rank File
# ---------------------------------------------------------------------------

def seed_initial_buckets(embed_model: SentenceTransformer) -> list:
    profiles = [
        {
            "bucket_id": "first_time_buyer",
            "description": "First-time home buyer, moderate budget, needs guidance and hand-holding",
            "markov_policy": {},
        },
        {
            "bucket_id": "investor",
            "description": "Real estate investor, cash buyer, flexible timeline, multiple units",
            "markov_policy": {},
        },
        {
            "bucket_id": "relocation",
            "description": "Corporate relocation buyer, high budget, specific school district, urgent timeline",
            "markov_policy": {},
        },
        {
            "bucket_id": "upgrade",
            "description": "Existing homeowner upgrading, equity available, family growing, local area",
            "markov_policy": {},
        },
    ]
    for p in profiles:
        p["centroid"] = embed_text(p["description"], embed_model)
        p["run_count"] = 0
    return profiles


def retrieve_bucket(embedding: list, buckets: list) -> Optional[dict]:
    if not buckets:
        return None
    best, best_sim = None, -1.0
    for b in buckets:
        sim = cosine_similarity(embedding, b["centroid"])
        if sim > best_sim:
            best_sim = sim
            best = b
    return best


def load_rank_file() -> tuple:
    return _MEMORY_STORE.get("buckets", []), _MEMORY_STORE.get("global_policy", {})


def _merge_policy_entries(base: dict, update: dict) -> dict:
    merged = dict(base)
    for state, actions in update.items():
        if state not in merged:
            merged[state] = {}
        for action, entry in actions.items():
            if action not in merged[state]:
                merged[state][action] = dict(entry)
            else:
                e = merged[state][action]
                n_old, n_new = e["count"], entry["count"]
                n_total = n_old + n_new
                e["success_rate"] = (e["success_rate"] * n_old + entry["success_rate"] * n_new) / n_total
                e["avg_cost"] = (e["avg_cost"] * n_old + entry["avg_cost"] * n_new) / n_total
                e["count"] = n_total
                e["next_state"] = entry["next_state"]
    return merged


def save_rank_file(buckets: list, global_policy: dict, run_meta: dict = None):
    existing = _MEMORY_STORE

    existing["global_policy"] = _merge_policy_entries(existing["global_policy"], global_policy)

    existing_bucket_map = {b["bucket_id"]: b for b in existing["buckets"]}
    for b in buckets:
        bid = b["bucket_id"]
        if bid not in existing_bucket_map:
            existing_bucket_map[bid] = b
        else:
            eb = existing_bucket_map[bid]
            eb["markov_policy"] = _merge_policy_entries(eb["markov_policy"], b["markov_policy"])
            eb["run_count"] = eb.get("run_count", 0) + b.get("run_count", 0)

    existing["buckets"] = list(existing_bucket_map.values())
    existing["total_runs"] = existing.get("total_runs", 0) + 1


def merge_for_lookup(bucket_policy: dict, global_policy: dict) -> dict:
    merged = dict(global_policy)
    for state, actions in bucket_policy.items():
        if state not in merged:
            merged[state] = {}
        for action, entry in actions.items():
            if action not in merged[state] or entry["success_rate"] > merged[state][action]["success_rate"]:
                merged[state][action] = entry
    return merged

# ---------------------------------------------------------------------------
# LLM Helper (OpenRouter)
# ---------------------------------------------------------------------------

def llm_call(system_prompt: str, user_message: str, max_tokens: int = 100) -> str:
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# State Classifier
# ---------------------------------------------------------------------------

def classify_state_rules(ctx: BuyerContext) -> Optional[str]:
    if not ctx.budget or not ctx.location:
        return "UNDERSTAND_BUYER"
    if not ctx.qualified:
        return "QUALIFY_LEAD"
    if not ctx.listings:
        return "SEARCH_OPTIONS"
    if len(ctx.listings) > 5:
        return "FILTER_OPTIONS"
    if not ctx.ranked:
        return "RANK_OPTIONS"
    if not ctx.contacted:
        return "CONTACT_BUYER"
    if not ctx.tour_scheduled:
        return "SCHEDULE_TOUR"
    if not ctx.crm_updated:
        return "UPDATE_CRM"
    return "DONE"


_CLASSIFY_SYSTEM = f"""You are a real estate workflow state classifier.

Given context about a buyer interaction, classify the current workflow state as exactly one of:
{", ".join(STATES)}

Rules:
- UNDERSTAND_BUYER: missing budget or location info
- QUALIFY_LEAD: have basic info but not yet qualified
- SEARCH_OPTIONS: qualified but no listings yet
- FILTER_OPTIONS: too many listings (>5), need to narrow down
- RANK_OPTIONS: listings found but not ranked
- CONTACT_BUYER: listings ranked, ready to contact buyer
- SCHEDULE_TOUR: contacted buyer, need to schedule tour
- UPDATE_CRM: tour scheduled, update CRM records
- DONE: all steps complete
- RECOVER: error or stuck state

Respond with ONLY the state name, nothing else."""


def classify_state_llm(ctx: BuyerContext) -> str:
    user_msg = json.dumps({
        "budget": ctx.budget,
        "location": ctx.location,
        "qualified": ctx.qualified,
        "listings_count": len(ctx.listings),
        "ranked": ctx.ranked,
        "contacted": ctx.contacted,
        "tour_scheduled": ctx.tour_scheduled,
        "crm_updated": ctx.crm_updated,
    })
    state = llm_call(_CLASSIFY_SYSTEM, user_msg, max_tokens=20)
    return state if state in STATES else "RECOVER"


def classify_state(ctx: BuyerContext, _client=None) -> str:
    state = classify_state_rules(ctx)
    if state:
        return state
    return classify_state_llm(ctx)

# ---------------------------------------------------------------------------
# Task Extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You are a real estate task extractor. Extract structured buyer information from conversation text.

Return ONLY valid JSON with these fields:
{
  "buyer_intent": "brief description of what they want",
  "budget": "budget amount as string, empty string if unknown",
  "location": "city/neighborhood, empty string if unknown",
  "urgency": "high/medium/low based on timeline mentions"
}

If a field is not mentioned, use empty string for text fields."""


def extract_task(ll_ctx: LifeLoggerContext, _client=None) -> BuyerContext:
    # Combine LifeLogger context into a single text block
    parts = [ll_ctx.conversation_summary]
    if ll_ctx.recent_speech:
        parts.append("Recent speech:\n" + "\n".join(ll_ctx.recent_speech[-10:]))
    topics = []
    for chunk in ll_ctx.history_chunks:
        if "topics" in chunk and chunk["topics"]:
            topics.extend(chunk["topics"])
    if topics:
        parts.append("Topics: " + ", ".join(topics[:10]))
    user_text = "\n\n".join(parts)

    raw = llm_call(_EXTRACT_SYSTEM, user_text, max_tokens=200)
    try:
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return BuyerContext(
        raw_conversation=ll_ctx.conversation_summary,
        buyer_intent=data.get("buyer_intent", ""),
        budget=data.get("budget", ""),
        location=data.get("location", ""),
        urgency=data.get("urgency", "medium"),
    )

# ---------------------------------------------------------------------------
# Mock Tools
# ---------------------------------------------------------------------------

def _tool_extract_requirements(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["extract_requirements"] * random.uniform(0.8, 1.2)
    if not ctx.buyer_intent:
        ctx.buyer_intent = "Buy a home"
    return ctx, cost, True


def _tool_qualify_lead(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["qualify_lead"] * random.uniform(0.8, 1.2)
    if mode == "cold" and random.random() < 0.20:
        return ctx, cost, False  # cold run failure
    ctx.qualified = True
    return ctx, cost, True


def _tool_search_listings(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["search_listings"] * random.uniform(0.8, 1.2)
    if mode == "cold" and random.random() < 0.20:
        return ctx, cost, False
    ctx.listings = [f"Home {i+1} at {ctx.location}" for i in range(random.randint(4, 8))]
    return ctx, cost, True


def _tool_filter_listings(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["filter_listings"] * random.uniform(0.8, 1.2)
    ctx.listings = ctx.listings[:3]
    return ctx, cost, True


def _tool_rank_listings(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["rank_listings"] * random.uniform(0.8, 1.2)
    ctx.ranked = True
    return ctx, cost, True


def _tool_send_message(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["send_message"] * random.uniform(0.8, 1.2)
    ctx.contacted = True
    return ctx, cost, True


def _tool_check_calendar(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["check_calendar"] * random.uniform(0.8, 1.2)
    return ctx, cost, True


def _tool_schedule_tour(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["schedule_tour"] * random.uniform(0.8, 1.2)
    if not ctx.contacted:
        return ctx, cost, False
    ctx.tour_scheduled = True
    return ctx, cost, True


def _tool_update_crm(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["update_crm"] * random.uniform(0.8, 1.2)
    ctx.crm_updated = True
    return ctx, cost, True


def _tool_ask_clarifying_question(ctx: BuyerContext, mode: str) -> tuple:
    cost = TOOL_COSTS["ask_clarifying_question"] * random.uniform(0.8, 1.2)
    if not ctx.budget:
        ctx.budget = "$400,000"
    if not ctx.location:
        ctx.location = "Pittsburgh, PA"
    return ctx, cost, True


TOOLS = {
    "extract_requirements":    _tool_extract_requirements,
    "qualify_lead":            _tool_qualify_lead,
    "search_listings":         _tool_search_listings,
    "filter_listings":         _tool_filter_listings,
    "rank_listings":           _tool_rank_listings,
    "send_message":            _tool_send_message,
    "check_calendar":          _tool_check_calendar,
    "schedule_tour":           _tool_schedule_tour,
    "update_crm":              _tool_update_crm,
    "ask_clarifying_question": _tool_ask_clarifying_question,
}

# ---------------------------------------------------------------------------
# LLM Action Selection
# ---------------------------------------------------------------------------

_ACTION_SYSTEM = """You are a real estate AI agent choosing the next tool to call.

Available tools:
- extract_requirements: Extract buyer requirements from conversation
- qualify_lead: Qualify if buyer is a serious prospect
- search_listings: Search MLS for matching properties
- filter_listings: Filter listings that don't match criteria
- rank_listings: Rank listings by match score
- send_message: Send message to buyer
- check_calendar: Check agent calendar availability
- schedule_tour: Schedule a property tour
- update_crm: Update CRM with buyer/lead info
- ask_clarifying_question: Ask buyer for missing information

Given the current workflow state and buyer context, choose the single most appropriate tool.
Respond with ONLY the tool name, nothing else."""


def get_llm_action(state: str, ctx: BuyerContext, _client=None) -> str:
    user_msg = f"State: {state}\nBuyer intent: {ctx.buyer_intent}\nBudget: {ctx.budget}\nLocation: {ctx.location}\nQualified: {ctx.qualified}\nListings: {len(ctx.listings)}\nRanked: {ctx.ranked}"
    action = llm_call(_ACTION_SYSTEM, user_msg, max_tokens=30).lower()
    # Find first matching tool name
    for tool in TOOLS:
        if tool in action:
            return tool
    return "ask_clarifying_question"


def action_is_valid(action: str, ctx: BuyerContext) -> bool:
    if action not in TOOLS:
        return False
    if action == "schedule_tour" and not ctx.contacted:
        return False
    if action == "rank_listings" and not ctx.listings:
        return False
    if action == "filter_listings" and not ctx.listings:
        return False
    return True

# ---------------------------------------------------------------------------
# Markov Model
# ---------------------------------------------------------------------------

def get_markov_action(state: str, policy: dict) -> tuple:
    if state not in policy or not policy[state]:
        return None, 0.0
    best_action, best_score = None, -1.0
    for action, entry in policy[state].items():
        score = entry["success_rate"] - 0.1 * entry["avg_cost"]
        if score > best_score:
            best_score = score
            best_action = action
    return best_action, best_score


def update_markov(policy: dict, logs: list, task_success: bool) -> dict:
    for log in logs:
        s, a = log.current_state, log.chosen_action
        if s not in policy:
            policy[s] = {}
        if a not in policy[s]:
            policy[s][a] = {"count": 0, "success_rate": 0.0, "avg_cost": 0.0, "next_state": log.next_state}
        entry = policy[s][a]
        outcome = 1.0 if (log.success and task_success) else (0.5 if log.success else 0.0)
        n = entry["count"] + 1
        entry["success_rate"] = (entry["success_rate"] * entry["count"] + outcome) / n
        entry["avg_cost"] = (entry["avg_cost"] * entry["count"] + log.tool_cost) / n
        entry["count"] = n
        entry["next_state"] = log.next_state
    return policy

# ---------------------------------------------------------------------------
# Policy Prior Logic
# ---------------------------------------------------------------------------

def choose_action(state: str, ctx: BuyerContext, policy: dict, mode: str) -> tuple:
    markov_action, confidence = get_markov_action(state, policy)
    llm_action = get_llm_action(state, ctx)

    if mode == "cold":
        return llm_action, markov_action, llm_action, 0.0

    if (confidence > CONFIDENCE_THRESHOLD
            and markov_action
            and action_is_valid(markov_action, ctx)):
        return llm_action, markov_action, markov_action, confidence

    return llm_action, markov_action, llm_action, confidence

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_step(log: StepLog, mode: str):
    color = RED if mode == "cold" else GREEN
    markov_str = log.markov_action if log.markov_action else "N/A"
    chosen_marker = ""
    if log.markov_action and log.chosen_action == log.markov_action and mode == "warm":
        chosen_marker = f" {BOLD}[MARKOV]{RESET}"
    print(f"\n{color}Step {log.step}{RESET}")
    print(f"  Current State:       {CYAN}{log.current_state}{RESET}")
    print(f"  LLM Action:          {log.llm_action}")
    print(f"  Markov Suggested:    {YELLOW}{markov_str}{RESET}")
    print(f"  Chosen Action:       {BOLD}{log.chosen_action}{RESET}{chosen_marker}")
    print(f"  Confidence:          {log.confidence:.2f}")
    print(f"  Next State:          {CYAN}{log.next_state}{RESET}")
    if not log.success:
        print(f"  {RED}[TOOL FAILED — entering RECOVER]{RESET}")


def print_summary(mode: str, logs: list, cost: float, success: bool):
    color = RED if mode == "cold" else GREEN
    print(f"\n{BOLD}{color}=== {mode.upper()} RUN ==={RESET}")
    print(f"  steps:   {len(logs)}")
    print(f"  cost:    ${cost:.4f}")
    print(f"  success: {success}")

# ---------------------------------------------------------------------------
# Execution Loop
# ---------------------------------------------------------------------------

def run_agent(
    ctx: BuyerContext,
    policy: dict,
    mode: str,
    buckets: list,
    embed_model: SentenceTransformer,
) -> tuple:
    state = classify_state(ctx)
    logs, total_cost = [], 0.0

    while state != "DONE" and len(logs) < MAX_STEPS:
        embedding = embed_text(build_embedding_text(ctx, state), embed_model)
        bucket = retrieve_bucket(embedding, buckets)
        effective_policy = merge_for_lookup(
            bucket["markov_policy"] if bucket else {},
            policy
        )

        llm_act, markov_act, chosen, conf = choose_action(state, ctx, effective_policy, mode)

        # Execute tool
        if chosen in TOOLS:
            ctx, cost, success = TOOLS[chosen](ctx, mode)
        else:
            cost, success = 0.01, False

        next_state = classify_state(ctx)
        if not success:
            next_state = "RECOVER"

        log = StepLog(
            step=len(logs) + 1,
            current_state=state,
            llm_action=llm_act,
            markov_action=markov_act,
            chosen_action=chosen,
            confidence=conf,
            next_state=next_state,
            tool_cost=cost,
            success=success,
        )
        logs.append(log)
        print_step(log, mode)

        total_cost += cost
        state = next_state

        # Small pause to avoid rate limits
        time.sleep(0.2)

    return logs, total_cost, (state == "DONE")

# ---------------------------------------------------------------------------
# Demo Conversations
# ---------------------------------------------------------------------------

COLD_TASK_TEXT = (
    "Hi there, I saw your ad online. I'm thinking about maybe buying a house sometime soon. "
    "I'm not totally sure where I want to live yet — maybe Pittsburgh or somewhere nearby? "
    "I don't have a firm budget in mind, somewhere between $300k and $600k I guess. "
    "My wife and I have two young kids so schools matter. No huge rush though."
)

WARM_TASK_TEXT = (
    "We need a 3-bedroom home in Squirrel Hill, Pittsburgh. Our budget is $450,000 and we're "
    "pre-approved. Hoping to move by August — my new job starts then. Good schools are a must. "
    "We've done this before so we know what we want."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenHome Agent Memory Network")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic training traces and exit")
    parser.add_argument("--lifelogger", action="store_true", help="Connect to live LifeLogger instance")
    args = parser.parse_args()

    print(f"\n{BOLD}OpenHome Agent Memory Network (AMN){RESET}")
    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.bootstrap:
        from trace_generator import bootstrap_rank_file
        bootstrap_rank_file(embed_model)
        return

    # Load shared Markov state (may include contributions from other users)
    buckets, global_policy = load_rank_file()
    if not buckets:
        print("No rank_file.json found. Seeding initial buckets...")
        buckets = seed_initial_buckets(embed_model)

    print(f"Loaded policy: {sum(len(v) for v in global_policy.values())} state-action pairs from shared memory\n")

    if args.lifelogger:
        print("LifeLogger mode: import lifelogger and pass an EnhancedListenerCapability instance to from_lifelogger()")
        print("Falling back to demo text for now.")

    random.seed(42)

    # -------- COLD RUN --------
    print("\n" + "=" * 60)
    print(f"{BOLD}{RED}COLD RUN (no Markov prior — LLM-only decisions){RESET}")
    print("=" * 60)
    cold_ll = from_text(COLD_TASK_TEXT)
    print("Extracting buyer context from conversation...")
    cold_ctx = extract_task(cold_ll)
    print(f"Buyer: intent={cold_ctx.buyer_intent!r}, budget={cold_ctx.budget!r}, location={cold_ctx.location!r}")

    cold_logs, cold_cost, cold_ok = run_agent(cold_ctx, {}, "cold", buckets, embed_model)

    # Learn from cold run
    updated_policy = update_markov(dict(global_policy), cold_logs, cold_ok)
    save_rank_file(buckets, updated_policy, {"source": "live_run"})
    print(f"\n[AMN] Cold run complete. Policy learned and saved to {RANK_FILE}")

    # -------- WARM RUN --------
    print("\n" + "=" * 60)
    print(f"{BOLD}{GREEN}WARM RUN (Markov prior loaded — policy-guided decisions){RESET}")
    print("=" * 60)

    # Re-read: picks up this run's learning + any other users' contributions
    buckets, global_policy = load_rank_file()
    print(f"Policy now has {sum(len(v) for v in global_policy.values())} state-action pairs")

    warm_ll = from_text(WARM_TASK_TEXT)
    print("Extracting buyer context from conversation...")
    warm_ctx = extract_task(warm_ll)
    print(f"Buyer: intent={warm_ctx.buyer_intent!r}, budget={warm_ctx.budget!r}, location={warm_ctx.location!r}")

    warm_logs, warm_cost, warm_ok = run_agent(warm_ctx, global_policy, "warm", buckets, embed_model)

    # Also update policy with warm run learnings
    final_policy = update_markov(dict(global_policy), warm_logs, warm_ok)
    save_rank_file(buckets, final_policy, {"source": "warm_run"})

    # -------- FINAL COMPARISON --------
    print("\n" + "=" * 60)
    print(f"{BOLD}FINAL COMPARISON{RESET}")
    print("=" * 60)
    print_summary("cold", cold_logs, cold_cost, cold_ok)
    print_summary("warm", warm_logs, warm_cost, warm_ok)

    step_diff = len(cold_logs) - len(warm_logs)
    cost_diff = cold_cost - warm_cost
    print(f"\n{BOLD}Improvement:{RESET}")
    print(f"  {GREEN}{step_diff} fewer steps{RESET} ({len(cold_logs)} → {len(warm_logs)})")
    print(f"  {GREEN}${cost_diff:.4f} cost reduction{RESET} (${cold_cost:.4f} → ${warm_cost:.4f})")

    print(f"\n[AMN Network] In-memory store total_runs: {_MEMORY_STORE.get('total_runs', 0)} (shared across calls this session)")


if __name__ == "__main__":
    main()
