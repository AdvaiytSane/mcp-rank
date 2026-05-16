"""
OpenHome Agent Memory Network (AMN)
Real estate lead conversion agent with Markov policy learning.

pip install requests sentence-transformers numpy

Usage:
    python amn.py            # Bootstrap synthetic traces + run cold/warm demo
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# CONFIG — paste your OpenRouter key here
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.0-flash-001"

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

# In-memory Markov store — persists across calls within the same session
_STORE: dict = {"total_runs": 0, "buckets": [], "global_policy": {}}

# ANSI colors
RED    = "\033[91m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

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
    """Wrap a live EnhancedListenerCapability instance."""
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
    return float(np.dot(a, b) / denom) if denom else 0.0

# ---------------------------------------------------------------------------
# In-Memory Store & Bucket Management
# ---------------------------------------------------------------------------

def seed_initial_buckets(embed_model: SentenceTransformer) -> list:
    profiles = [
        {"bucket_id": "first_time_buyer",
         "description": "First-time home buyer, moderate budget, needs guidance and hand-holding",
         "markov_policy": {}},
        {"bucket_id": "investor",
         "description": "Real estate investor, cash buyer, flexible timeline, multiple units",
         "markov_policy": {}},
        {"bucket_id": "relocation",
         "description": "Corporate relocation buyer, high budget, specific school district, urgent timeline",
         "markov_policy": {}},
        {"bucket_id": "upgrade",
         "description": "Existing homeowner upgrading, equity available, family growing, local area",
         "markov_policy": {}},
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
            best_sim, best = sim, b
    return best


def _merge_entries(base: dict, update: dict) -> dict:
    merged = dict(base)
    for state, actions in update.items():
        merged.setdefault(state, {})
        for action, entry in actions.items():
            if action not in merged[state]:
                merged[state][action] = dict(entry)
            else:
                e = merged[state][action]
                n_old, n_new = e["count"], entry["count"]
                n = n_old + n_new
                e["success_rate"] = (e["success_rate"] * n_old + entry["success_rate"] * n_new) / n
                e["avg_cost"] = (e["avg_cost"] * n_old + entry["avg_cost"] * n_new) / n
                e["count"] = n
                e["next_state"] = entry["next_state"]
    return merged


def store_save(buckets: list, global_policy: dict):
    _STORE["global_policy"] = _merge_entries(_STORE["global_policy"], global_policy)
    bucket_map = {b["bucket_id"]: b for b in _STORE["buckets"]}
    for b in buckets:
        bid = b["bucket_id"]
        if bid not in bucket_map:
            bucket_map[bid] = b
        else:
            bucket_map[bid]["markov_policy"] = _merge_entries(
                bucket_map[bid]["markov_policy"], b["markov_policy"]
            )
            bucket_map[bid]["run_count"] = bucket_map[bid].get("run_count", 0) + b.get("run_count", 0)
    _STORE["buckets"] = list(bucket_map.values())
    _STORE["total_runs"] += 1


def store_load() -> tuple:
    return _STORE["buckets"], _STORE["global_policy"]


def merge_for_lookup(bucket_policy: dict, global_policy: dict) -> dict:
    merged = dict(global_policy)
    for state, actions in bucket_policy.items():
        merged.setdefault(state, {})
        for action, entry in actions.items():
            if action not in merged[state] or entry["success_rate"] > merged[state][action]["success_rate"]:
                merged[state][action] = entry
    return merged

# ---------------------------------------------------------------------------
# LLM Helper
# ---------------------------------------------------------------------------

def llm_call(system_prompt: str, user_message: str, max_tokens: int = 100) -> str:
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
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
    if not ctx.budget or not ctx.location: return "UNDERSTAND_BUYER"
    if not ctx.qualified:                  return "QUALIFY_LEAD"
    if not ctx.listings:                   return "SEARCH_OPTIONS"
    if len(ctx.listings) > 5:             return "FILTER_OPTIONS"
    if not ctx.ranked:                     return "RANK_OPTIONS"
    if not ctx.contacted:                  return "CONTACT_BUYER"
    if not ctx.tour_scheduled:            return "SCHEDULE_TOUR"
    if not ctx.crm_updated:              return "UPDATE_CRM"
    return "DONE"


_CLASSIFY_SYSTEM = (
    "You are a real estate workflow state classifier.\n"
    "Given context, return ONLY one of these state names:\n"
    + ", ".join(STATES) + "\n\n"
    "Rules:\n"
    "- UNDERSTAND_BUYER: missing budget or location\n"
    "- QUALIFY_LEAD: info present but not yet qualified\n"
    "- SEARCH_OPTIONS: qualified but no listings\n"
    "- FILTER_OPTIONS: too many listings (>5)\n"
    "- RANK_OPTIONS: listings found but not ranked\n"
    "- CONTACT_BUYER: listings ranked, ready to contact\n"
    "- SCHEDULE_TOUR: contacted, need to schedule tour\n"
    "- UPDATE_CRM: tour scheduled, update records\n"
    "- DONE: all steps complete\n"
    "- RECOVER: error or stuck"
)


def classify_state(ctx: BuyerContext) -> str:
    state = classify_state_rules(ctx)
    if state:
        return state
    result = llm_call(_CLASSIFY_SYSTEM, json.dumps({
        "budget": ctx.budget, "location": ctx.location,
        "qualified": ctx.qualified, "listings": len(ctx.listings),
        "ranked": ctx.ranked, "contacted": ctx.contacted,
        "tour_scheduled": ctx.tour_scheduled, "crm_updated": ctx.crm_updated,
    }), max_tokens=20)
    return result if result in STATES else "RECOVER"

# ---------------------------------------------------------------------------
# Task Extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are a real estate task extractor. Return ONLY valid JSON:\n"
    '{"buyer_intent": "...", "budget": "...", "location": "...", "urgency": "high|medium|low"}\n'
    "Use empty string for unknown fields."
)


def extract_task(ll_ctx: LifeLoggerContext) -> BuyerContext:
    parts = [ll_ctx.conversation_summary]
    if ll_ctx.recent_speech:
        parts.append("Recent speech:\n" + "\n".join(ll_ctx.recent_speech[-10:]))
    topics = [t for chunk in ll_ctx.history_chunks for t in chunk.get("topics", [])]
    if topics:
        parts.append("Topics: " + ", ".join(topics[:10]))

    raw = llm_call(_EXTRACT_SYSTEM, "\n\n".join(parts), max_tokens=200)
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
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

def _tool_extract_requirements(ctx, mode):
    cost = TOOL_COSTS["extract_requirements"] * random.uniform(0.8, 1.2)
    if not ctx.buyer_intent:
        ctx.buyer_intent = "Buy a home"
    return ctx, cost, True

def _tool_qualify_lead(ctx, mode):
    cost = TOOL_COSTS["qualify_lead"] * random.uniform(0.8, 1.2)
    if mode == "cold" and random.random() < 0.20:
        return ctx, cost, False
    ctx.qualified = True
    return ctx, cost, True

def _tool_search_listings(ctx, mode):
    cost = TOOL_COSTS["search_listings"] * random.uniform(0.8, 1.2)
    if mode == "cold" and random.random() < 0.20:
        return ctx, cost, False
    ctx.listings = [f"Home {i+1} at {ctx.location}" for i in range(random.randint(4, 8))]
    return ctx, cost, True

def _tool_filter_listings(ctx, mode):
    cost = TOOL_COSTS["filter_listings"] * random.uniform(0.8, 1.2)
    ctx.listings = ctx.listings[:3]
    return ctx, cost, True

def _tool_rank_listings(ctx, mode):
    cost = TOOL_COSTS["rank_listings"] * random.uniform(0.8, 1.2)
    ctx.ranked = True
    return ctx, cost, True

def _tool_send_message(ctx, mode):
    cost = TOOL_COSTS["send_message"] * random.uniform(0.8, 1.2)
    ctx.contacted = True
    return ctx, cost, True

def _tool_check_calendar(ctx, mode):
    return ctx, TOOL_COSTS["check_calendar"] * random.uniform(0.8, 1.2), True

def _tool_schedule_tour(ctx, mode):
    cost = TOOL_COSTS["schedule_tour"] * random.uniform(0.8, 1.2)
    if not ctx.contacted:
        return ctx, cost, False
    ctx.tour_scheduled = True
    return ctx, cost, True

def _tool_update_crm(ctx, mode):
    cost = TOOL_COSTS["update_crm"] * random.uniform(0.8, 1.2)
    ctx.crm_updated = True
    return ctx, cost, True

def _tool_ask_clarifying_question(ctx, mode):
    cost = TOOL_COSTS["ask_clarifying_question"] * random.uniform(0.8, 1.2)
    if not ctx.budget:   ctx.budget = "$400,000"
    if not ctx.location: ctx.location = "Pittsburgh, PA"
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

_ACTION_SYSTEM = (
    "You are a real estate AI agent. Choose the single best tool to call next.\n"
    "Tools: extract_requirements, qualify_lead, search_listings, filter_listings,\n"
    "rank_listings, send_message, check_calendar, schedule_tour, update_crm, ask_clarifying_question\n"
    "Respond with ONLY the tool name."
)


def get_llm_action(state: str, ctx: BuyerContext) -> str:
    user_msg = (f"State: {state} | intent: {ctx.buyer_intent} | budget: {ctx.budget} | "
                f"location: {ctx.location} | qualified: {ctx.qualified} | "
                f"listings: {len(ctx.listings)} | ranked: {ctx.ranked}")
    action = llm_call(_ACTION_SYSTEM, user_msg, max_tokens=30).lower()
    for tool in TOOLS:
        if tool in action:
            return tool
    return "ask_clarifying_question"


def action_is_valid(action: str, ctx: BuyerContext) -> bool:
    if action not in TOOLS:                                  return False
    if action == "schedule_tour" and not ctx.contacted:      return False
    if action in ("rank_listings", "filter_listings") and not ctx.listings: return False
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
            best_score, best_action = score, action
    return best_action, best_score


def update_markov(policy: dict, logs: list, task_success: bool) -> dict:
    for log in logs:
        s, a = log.current_state, log.chosen_action
        policy.setdefault(s, {})
        policy[s].setdefault(a, {"count": 0, "success_rate": 0.0, "avg_cost": 0.0, "next_state": log.next_state})
        entry = policy[s][a]
        outcome = 1.0 if (log.success and task_success) else (0.5 if log.success else 0.0)
        n = entry["count"] + 1
        entry["success_rate"] = (entry["success_rate"] * entry["count"] + outcome) / n
        entry["avg_cost"]     = (entry["avg_cost"]     * entry["count"] + log.tool_cost) / n
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

    if confidence > CONFIDENCE_THRESHOLD and markov_action and action_is_valid(markov_action, ctx):
        return llm_action, markov_action, markov_action, confidence

    return llm_action, markov_action, llm_action, confidence

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_step(log: StepLog, mode: str):
    color = RED if mode == "cold" else GREEN
    markov_str = log.markov_action or "N/A"
    marker = f" {BOLD}[MARKOV]{RESET}" if (log.markov_action and log.chosen_action == log.markov_action and mode == "warm") else ""
    print(f"\n{color}Step {log.step}{RESET}")
    print(f"  Current State:    {CYAN}{log.current_state}{RESET}")
    print(f"  LLM Action:       {log.llm_action}")
    print(f"  Markov Suggested: {YELLOW}{markov_str}{RESET}")
    print(f"  Chosen Action:    {BOLD}{log.chosen_action}{RESET}{marker}")
    print(f"  Confidence:       {log.confidence:.2f}")
    print(f"  Next State:       {CYAN}{log.next_state}{RESET}")
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

def run_agent(ctx: BuyerContext, policy: dict, mode: str, buckets: list, embed_model) -> tuple:
    state = classify_state(ctx)
    logs, total_cost = [], 0.0

    while state != "DONE" and len(logs) < MAX_STEPS:
        embedding = embed_text(build_embedding_text(ctx, state), embed_model)
        bucket = retrieve_bucket(embedding, buckets)
        effective_policy = merge_for_lookup(bucket["markov_policy"] if bucket else {}, policy)

        llm_act, markov_act, chosen, conf = choose_action(state, ctx, effective_policy, mode)

        ctx, cost, success = TOOLS[chosen](ctx, mode) if chosen in TOOLS else (ctx, 0.01, False)
        next_state = "RECOVER" if not success else classify_state(ctx)

        log = StepLog(len(logs) + 1, state, llm_act, markov_act, chosen, conf, next_state, cost, success)
        logs.append(log)
        print_step(log, mode)
        total_cost += cost
        state = next_state
        time.sleep(0.2)

    return logs, total_cost, (state == "DONE")

# ---------------------------------------------------------------------------
# Synthetic Trace Bootstrap
# ---------------------------------------------------------------------------

_BUYER_PROFILES = [
    {
        "profile": "first_time_buyer",
        "ideal_path": [
            ("UNDERSTAND_BUYER", "ask_clarifying_question"),
            ("QUALIFY_LEAD",     "qualify_lead"),
            ("SEARCH_OPTIONS",   "search_listings"),
            ("FILTER_OPTIONS",   "filter_listings"),
            ("RANK_OPTIONS",     "rank_listings"),
            ("CONTACT_BUYER",    "send_message"),
            ("SCHEDULE_TOUR",    "schedule_tour"),
            ("UPDATE_CRM",       "update_crm"),
        ],
    },
    {
        "profile": "investor",
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

_PLAUSIBLE_ALTERNATES = {
    "UNDERSTAND_BUYER": ["ask_clarifying_question", "extract_requirements"],
    "QUALIFY_LEAD":     ["qualify_lead", "ask_clarifying_question"],
    "SEARCH_OPTIONS":   ["search_listings", "ask_clarifying_question"],
    "FILTER_OPTIONS":   ["filter_listings", "search_listings"],
    "RANK_OPTIONS":     ["rank_listings", "filter_listings"],
    "CONTACT_BUYER":    ["send_message", "check_calendar"],
    "SCHEDULE_TOUR":    ["schedule_tour", "check_calendar", "send_message"],
    "UPDATE_CRM":       ["update_crm", "send_message"],
    "RECOVER":          ["ask_clarifying_question", "extract_requirements"],
}


def _make_step(num, state, action, next_state, success):
    return StepLog(
        step=num, current_state=state, llm_action=action, markov_action=None,
        chosen_action=action, confidence=0.0, next_state=next_state,
        tool_cost=TOOL_COSTS.get(action, 0.02) * random.uniform(0.8, 1.2),
        success=success,
    )


def _generate_trace(profile: dict, noise: float = 0.15, seed: int = 0) -> list:
    random.seed(seed)
    path, logs, step, i = profile["ideal_path"], [], 1, 0
    while i < len(path) and step <= 20:
        state, action = path[i]
        next_state = path[i + 1][0] if i + 1 < len(path) else "DONE"
        if random.random() < noise and step > 1:
            alts = _PLAUSIBLE_ALTERNATES.get(state, [action])
            wrong = random.choice([a for a in alts if a != action] or [action])
            if random.random() < 0.6:
                logs.append(_make_step(step, state, wrong, "RECOVER", False)); step += 1
                logs.append(_make_step(step, "RECOVER", "ask_clarifying_question", state, True)); step += 1
                continue
        success = random.random() > 0.05
        logs.append(_make_step(step, state, action, next_state if success else "RECOVER", success)); step += 1
        if not success:
            logs.append(_make_step(step, "RECOVER", "ask_clarifying_question", state, True)); step += 1
        else:
            i += 1
    if i >= len(path):
        logs.append(_make_step(step, "UPDATE_CRM", "update_crm", "DONE", True))
    return logs


def bootstrap(embed_model) -> tuple:
    """Generate 40 synthetic traces and load into the in-memory store."""
    print("[bootstrap] Generating synthetic training traces...")
    traces = [
        _generate_trace(p, seed=hash(p["profile"]) + i)
        for p in _BUYER_PROFILES for i in range(10)
    ]
    policy = {}
    for trace in traces:
        policy = update_markov(policy, trace, task_success=True)
    buckets = seed_initial_buckets(embed_model)
    store_save(buckets, policy)
    print(f"[bootstrap] {len(traces)} traces loaded — "
          f"{len(policy)} states, "
          f"{sum(len(v) for v in policy.values())} state-action pairs")
    return buckets, policy

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

COLD_TASK = (
    "Hi there, I saw your ad online. I'm thinking about maybe buying a house sometime soon. "
    "I'm not totally sure where I want to live yet — maybe Pittsburgh or somewhere nearby? "
    "I don't have a firm budget in mind, somewhere between $300k and $600k I guess. "
    "My wife and I have two young kids so schools matter. No huge rush though."
)

WARM_TASK = (
    "We need a 3-bedroom home in Squirrel Hill, Pittsburgh. Our budget is $450,000 and we're "
    "pre-approved. Hoping to move by August — my new job starts then. Good schools are a must. "
    "We've done this before so we know what we want."
)


def run_demo(ll_ctx: LifeLoggerContext = None):
    """
    Entry point. Pass a LifeLoggerContext to use live data,
    or leave None to run the built-in cold/warm demo.
    """
    print(f"\n{BOLD}OpenHome Agent Memory Network (AMN){RESET}")
    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Bootstrap synthetic prior then load
    buckets, global_policy = bootstrap(embed_model)

    random.seed(42)

    # ---- COLD RUN ----
    print("\n" + "=" * 60)
    print(f"{BOLD}{RED}COLD RUN (LLM-only, no Markov prior){RESET}")
    print("=" * 60)
    cold_ll = ll_ctx or from_text(COLD_TASK)
    cold_ctx = extract_task(cold_ll)
    print(f"Buyer: {cold_ctx.buyer_intent!r} | {cold_ctx.budget!r} | {cold_ctx.location!r}")
    cold_logs, cold_cost, cold_ok = run_agent(cold_ctx, {}, "cold", buckets, embed_model)

    global_policy = update_markov(dict(global_policy), cold_logs, cold_ok)
    store_save(buckets, global_policy)

    # ---- WARM RUN ----
    print("\n" + "=" * 60)
    print(f"{BOLD}{GREEN}WARM RUN (Markov policy-guided){RESET}")
    print("=" * 60)
    buckets, global_policy = store_load()
    print(f"Policy: {sum(len(v) for v in global_policy.values())} state-action pairs loaded")
    warm_ll = from_text(WARM_TASK)
    warm_ctx = extract_task(warm_ll)
    print(f"Buyer: {warm_ctx.buyer_intent!r} | {warm_ctx.budget!r} | {warm_ctx.location!r}")
    warm_logs, warm_cost, warm_ok = run_agent(warm_ctx, global_policy, "warm", buckets, embed_model)

    store_save(buckets, update_markov(dict(global_policy), warm_logs, warm_ok))

    # ---- COMPARISON ----
    print("\n" + "=" * 60)
    print(f"{BOLD}FINAL COMPARISON{RESET}")
    print("=" * 60)
    print_summary("cold", cold_logs, cold_cost, cold_ok)
    print_summary("warm", warm_logs, warm_cost, warm_ok)
    step_diff, cost_diff = len(cold_logs) - len(warm_logs), cold_cost - warm_cost
    print(f"\n{BOLD}Improvement:{RESET}")
    print(f"  {GREEN}{step_diff} fewer steps{RESET} ({len(cold_logs)} → {len(warm_logs)})")
    print(f"  {GREEN}${cost_diff:.4f} saved{RESET} (${cold_cost:.4f} → ${warm_cost:.4f})")
    print(f"\n[AMN] Session total_runs: {_STORE['total_runs']}")


if __name__ == "__main__":
    run_demo()
