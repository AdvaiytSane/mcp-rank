You are an expert full-stack + AI systems engineer.

Build a hackathon prototype called:

OpenHome Agent Memory Network (AMN)

Goal:
Create an AI agent system that improves repeated real-estate workflows by learning reusable action policies using bucketed Markov models over abstract workflow states.

---

## CORE IDEA

Agents do NOT operate on raw tool/page states.

Instead:
- Convert all context into ABSTRACT WORKFLOW STATES
- Learn transitions between these states using a Markov model
- Use this learned policy as a PRIOR to guide future agent decisions

---

## DOMAIN

Real estate lead conversion.

Example task:
“Given a buyer conversation, qualify the lead, find matching homes, recommend next action, schedule a tour, and update CRM.”

---

## SYSTEM ARCHITECTURE

### 1. Task Extraction

Input:
- raw conversation text

Output JSON:
{
  "buyer_intent": "...",
  "budget": "...",
  "location": "...",
  "urgency": "...",
  "task_type": "lead_conversion"
}

---

### 2. MCP Router

Choose tool namespace:

- listings_mcp
- crm_mcp
- calendar_mcp
- messaging_mcp

---

### 3. ABSTRACT WORKFLOW STATES (CRITICAL)

Define fixed state set:

STATES = [
  "START",
  "UNDERSTAND_BUYER",
  "QUALIFY_LEAD",
  "SEARCH_OPTIONS",
  "FILTER_OPTIONS",
  "RANK_OPTIONS",
  "CONTACT_BUYER",
  "SCHEDULE_TOUR",
  "UPDATE_CRM",
  "FOLLOW_UP",
  "DONE",
  "RECOVER"
]

---

### 4. STATE CLASSIFIER

After every step, map context → abstract state.

Implement:

#### (A) Rule-based first

Example:

- missing budget/location → UNDERSTAND_BUYER
- not qualified → QUALIFY_LEAD
- no listings yet → SEARCH_OPTIONS
- listings found but not ranked → RANK_OPTIONS
- ready to contact → CONTACT_BUYER
- tour not scheduled → SCHEDULE_TOUR
- finished → DONE

#### (B) LLM fallback

Prompt:

“Given this context, classify into one of the predefined workflow states. Return only the state name.”

---

### 5. EMBEDDINGS + BUCKETS

Embed:

task + buyer_intent + constraints + current_state

Use embedding model.

Buckets:

Stored in rank_file.json:

{
  "buckets": [
    {
      "bucket_id": "...",
      "centroid": [...],
      "description": "...",
      "markov_policy": {...}
    }
  ]
}

---

### 6. BUCKET RETRIEVAL

At each step:

- embed current state
- compute cosine similarity with bucket centroids
- retrieve top-1 or top-3 buckets

---

### 7. MARKOV MODEL (CORE)

Define transitions over ABSTRACT STATES:

markov_policy = {
  "QUALIFY_LEAD": {
    "search_listings": {
      "count": 10,
      "success_rate": 0.9,
      "avg_cost": 0.05,
      "next_state": "SEARCH_OPTIONS"
    }
  }
}

IMPORTANT:
- States are abstract (not tool outputs)
- Actions are tool calls

---

### 8. POLICY PRIOR LOGIC

At each step:

1. Get current abstract state
2. Retrieve bucket(s)
3. Get candidate actions from Markov model

Score each action:

score = success_rate - 0.1 * avg_cost

---

### 9. HARNESS DECISION LOGIC

Implement:

if markov_confidence > threshold AND action_valid:
    use markov_action
else:
    use LLM action

Where:
- action_valid = tool exists + inputs available
- markov_confidence = success_rate or weighted score

---

### 10. TOOL SET (MOCK)

Implement:

- extract_requirements
- qualify_lead
- search_listings
- rank_listings
- send_message
- check_calendar
- schedule_tour
- update_crm
- ask_clarifying_question

---

### 11. EXECUTION LOOP

Loop:

state = START

while state != DONE:

    classify_state(context)
    retrieve_bucket
    get_markov_action
    get_llm_action

    choose action (policy prior logic)

    execute tool
    observe result

    update context
    transition to next state

---

### 12. LEARNING / UPDATE

After run:

For each transition:

- increment count
- update success_rate
- update avg_cost

If task succeeded:
- reward transitions

If failed:
- penalize transitions

---

### 13. DEMO REQUIREMENTS

Run SAME or SIMILAR task twice.

Cold run:
- no Markov guidance
- messy ordering
- more steps

Warm run:
- uses Markov policy
- clean workflow:
  UNDERSTAND → QUALIFY → SEARCH → RANK → CONTACT → SCHEDULE → UPDATE

---

### 14. OUTPUT LOGGING (VERY IMPORTANT)

Print at each step:

Current State:
LLM Action:
Markov Suggested Action:
Chosen Action:
Confidence:
Next State:

---

### 15. FINAL METRICS

Print:

- total steps
- total cost
- failures
- success

Compare:

Cold vs Warm

---

## CONSTRAINTS

- Python only
- No heavy infra
- Use in-memory structures or JSON
- Focus on clarity and demo impact

---

## KEY DESIGN PRINCIPLE

“We compress noisy agent context into stable workflow states, and learn reusable decision policies over those states.”

---

## DELIVERABLE

A runnable script that:

- runs a cold agent
- learns transitions
- runs a warm agent
- clearly shows improvement