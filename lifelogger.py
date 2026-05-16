

import json
import re
import asyncio
import time
import struct
import base64
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from src.agent.capability import MatchingCapability
from src.main import AgentWorker
from src.agent.capability_worker import CapabilityWorker

# =============================================================================
# ARCHITECTURE — Why it works this way
# =============================================================================
#
# This is an ambient intelligence smart speaker. It sits in a room and builds
# a MENTAL MODEL of what's happening — like a person on the couch, not a
# security camera replaying footage.
#
# A person doesn't maintain a timestamped log. They maintain a model:
#   "Jesse and Chris are here. They're testing software. The printer is running."
# When they're wrong, they update the model. When corrected, they fix it.
#
# THREE CONCURRENT LOOPS:
#
#   1. FAST CYCLE (every 15s) — "What just changed?"
#      - Sends 60s audio to both Deepgram (transcript) and Gemini (scene analysis)
#      - FUSES the results: Deepgram's speaker count is ground truth for WHO spoke,
#        Gemini fills in demographics/mood/background
#      - REWRITES the canonical room_state (not appends — rewrites)
#      - Detects corrections ("that's wrong", "be quiet", "I wasn't talking to you")
#
#   2. DEEP CYCLE (every 3 min) — "What's the full picture?"
#      - Sends full 3-minute audio buffer to Deepgram for high-quality diarization
#      - Sends that transcript to a fast LLM for topic extraction
#      - Produces a CONVERSATION SUMMARY that replaces raw transcript fragments
#      - Corrects speaker assignments from fast cycles (long audio = better diarization)
#
#   3. CONVERSATION HANDLER — "Am I being spoken to?"
#      - IDLE: only wake word activates
#      - ENGAGED: WHITELIST approach — only respond when there's positive evidence
#        of being addressed (wake word, reply window, direct address phrases, question+you)
#        Everything else is assumed to be cross-talk. Default is SILENCE.
#      - COOLDOWN: "Was that for me?" clarification
#
# CONTEXT FOR THE LLM:
#   Instead of 100+ timestamped entries with compounding hallucinations, the
#   conversation LLM gets:
#     1. ROOM STATE — single paragraph, rewritten each cycle, cross-validated
#     2. CONVERSATION SUMMARY — 3-4 sentences from the deep cycle
#     3. RECENT SPEECH — last 60 seconds of raw transcript for immediacy
#     4. CORRECTIONS — anything the users corrected so the LLM doesn't repeat lies
#
# HALLUCINATION PREVENTION:
#   - Room state is REWRITTEN, not appended. Errors don't compound.
#   - Gemini's claims are cross-validated against Deepgram's speaker count.
#   - When they disagree, Deepgram wins on speaker count (it has word-level evidence).
#   - User corrections are detected and applied immediately.
#   - The deep cycle's long-form transcript corrects fast-cycle diarization errors.
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

ANALYSIS_INTERVAL_SECONDS = 15
DEEP_ANALYSIS_INTERVAL_SECONDS = 180   # 3 minutes
AUDIO_WINDOW_SECONDS = 60
ENGAGEMENT_TIMEOUT_SECONDS = 30
COOLDOWN_SECONDS = 8
COOLDOWN_CONFIRM_TIMEOUT = 5
REPLY_WINDOW_SECONDS = 6
MAX_HISTORY_MESSAGES = 16
MAX_RECENT_SPEECH = 20                 # Rolling buffer of recent speech lines

DASHBOARD_URL = "https://file-sender.replit.app/api"
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# --- Models ---
AUDIO_ANALYSIS_MODEL = "google/gemini-3-flash-preview"   # Must support audio input
ROUTER_MODEL = "google/gemini-2.0-flash-001"           # Fast text model for intent gating
CONVERSATION_MODEL = "google/gemini-2.0-flash-001"     # Model for conversation responses
CONVERSATION_MAX_TOKENS = 300             

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPGRAM_MODEL = "nova-3"
ABILITY_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

WAKE_WORD_REGEX = re.compile(r'(?:hey\s+|oh\s+)?open\s*home[\s,\.!]*(.*)$', re.IGNORECASE)

# Phrases that ALWAYS trigger a response in ENGAGED mode, regardless of reply window
DIRECT_ADDRESS_PHRASES = [
    "what do you think", "any thoughts", "your opinion", "your take",
    "do you hear", "what do you hear", "what's going on", "what's happening",
    "what about you", "you listening", "are you there", "you awake",
    "recap", "summarize", "what did you hear", "who's talking",
    "how many people", "what music", "what song",
]

# Expanded disengage — anything that means "stop talking to me"
DISENGAGE_SIGNALS = [
    "thank you", "thanks", "that's all", "never mind",
    "go back to sleep", "stop talking", "okay thanks",
    "we're done", "go to sleep", "be quiet", "shut up",
    "wasn't talking to you", "not talking to you",
    "i wasn't talking to you", "i'm not talking to you",
    "quiet", "stop listening", "go away", "enough",
    "stop it", "can you stop", "please stop",
]

EXIT_KEYWORDS = [
    "exit listener", "stop listener", "quit listener",
    "deactivate listener", "shut down listener",
]

# Names for cross-talk detection — starts EMPTY, populated only from what's heard
# Common address terms are included as they signal cross-talk even without a proper name
KNOWN_NAMES = set()
CROSS_TALK_TERMS = {"mom", "mama", "dad", "daddy", "babe", "honey", "sweetie",
                    "dude", "bro", "sis", "baby"}

# Correction patterns — when users tell us we're wrong
CORRECTION_PATTERNS = [
    r"that'?s\s+(wrong|not right|incorrect|not true)",
    r"you'?re\s+wrong", r"no[,.]?\s+(there|that|it|i)\s+(aren|isn|wasn|didn|don)",
    r"(wasn'?t|not)\s+talking\s+to\s+you",
    r"(stop|quit|be quiet|shut up|enough)",
    r"i\s+(didn'?t|never)\s+say\s+that",
    r"there\s+(aren'?t|isn'?t|are\s+only|is\s+only)\s+\d+",
]

SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE // 8
WAV_HEADER_SIZE = 44


# =============================================================================
# PROMPTS
# =============================================================================

PERSONALITY_PROMPT = (
    "You are the OpenHome ambient intelligence speaker — sharp, witty, aware.\n\n"
    "Personality: Concise (1-3 sentences). Dry humor. Genuinely helpful. "
    "Aware of the room — reference what you've observed when relevant.\n\n"
    "WHAT YOU CURRENTLY KNOW ABOUT THE ROOM:\n{room_state}\n\n"
    "PEOPLE IN THE ROOM (voice profiles):\n{voice_profiles}\n\n"
    "WHAT'S BEEN HAPPENING (3-minute history blocks):\n{history_chain}\n\n"
    "{conversation_context}"
    "LAST 15 SECONDS OF SPEECH:\n{recent_speech}\n\n"
    "NOTABLE EVENTS (timestamped):\n{event_log}\n\n"
    "CORRECTIONS (things you got wrong — do NOT repeat these):\n{corrections}\n\n"
    "Rules:\n"
    "- Speak naturally. Read aloud. No markdown.\n"
    "- 1-3 sentences unless asked for detail.\n"
    "- Own your observations: 'You sound like...' not 'The analysis shows...'\n"
    "- If you don't know, say so. Don't hedge.\n"
    "- NEVER repeat something from YOUR previous responses.\n"
    "- If a correction is listed above, your old answer was WRONG. Use the corrected info.\n"
    "- Use the VOICE PROFILES to address people by name when known.\n"
    "- Use the HISTORY BLOCKS to understand what's been discussed over time.\n"
    "- ONLY reference names that appear in voice profiles or known names. Do NOT guess.\n"
    "- Your conversation history with the user follows in the messages below. "
    "Reference it naturally — if they asked about something 2 turns ago, you remember."
)

# Gemini gets structured prompt with confidence levels and DG transcript
GEMINI_STRUCTURED_PROMPT = (
    "You are analyzing live audio from a room with a smart speaker.\n\n"
    "CURRENT ROOM STATE (what we already believe — correct it if wrong):\n"
    "{room_state}\n\n"
    "{corrections}\n\n"
    "DEEPGRAM TRANSCRIPT OF THIS SAME AUDIO (machine-generated, may have errors):\n"
    "{dg_transcript}\n\n"
    "Analyze this audio snapshot. Use the transcript above as a reference for "
    "who spoke, but trust your OWN ears for background sounds, mood, and speaker "
    "demographics. The transcript may miss speakers or mis-attribute.\n\n"
    "For each observation, note WHEN in the clip you hear it using relative "
    "timestamps (e.g. '~0:15', '~0:30', '~0:45'). These are relative to the "
    "START of this audio clip, not wall clock time.\n\n"
    "Rate confidence: HIGH (clearly heard), MEDIUM (probably right), LOW (uncertain).\n\n"
    "Reply in this format:\n"
    "SPEAKER_COUNT: [number] | [HIGH/MEDIUM/LOW]\n"
    "SPEAKERS: [describe each - gender, estimated age range, tone, emotional state]\n"
    "BACKGROUND: [non-speech sounds with relative timestamps] | [HIGH/MEDIUM/LOW]\n"
    "BACKGROUND_GONE: [sounds from room state that you NO LONGER hear]\n"
    "NAMES_HEARD: [any proper names spoken, with speaker and ~timestamp]\n"
    "CHANGES: [what's new or different from room state above]\n"
    "ACTIVITY: [what's happening right now]\n"
    "MOOD: [overall emotional atmosphere]\n\n"
    "Rules:\n"
    "- Only report what you ACTUALLY HEAR. Do not invent speakers.\n"
    "- If you hear fewer speakers than the room state says, report the LOWER number.\n"
    "- If a sound from BACKGROUND in the room state is no longer audible, list it under BACKGROUND_GONE.\n"
    "- If nothing changed, say 'No significant changes.'\n"
    "- Plain text, no markdown."
)

SUMMARY_PROMPT = (
    "Summarize this 3-minute conversation transcript in 3-4 sentences.\n"
    "Focus on: who is speaking (use Speaker IDs, and assign real names if you "
    "see them mentioned in the transcript), what topics they discussed, "
    "any decisions or plans, the overall tone.\n"
    "Write as a brief narrative. No timestamps. No bullet points.\n\n"
    "Known names in the room so far: {known_names}\n\n"
    "Transcript:\n{transcript}"
)

# --- DEEP CYCLE: Single LLM call extracts everything from the 3-min transcript ---
DEEP_EXTRACTION_PROMPT = (
    "You are analyzing a 3-minute transcript from a smart speaker in a room.\n"
    "The transcript uses Speaker IDs (Speaker 0, Speaker 1, etc.) from Deepgram diarization.\n\n"
    "CURRENT VOICE PROFILES (what we already know — update or correct):\n"
    "{voice_profiles}\n\n"
    "PREVIOUS 3-MINUTE SUMMARY (what happened in the LAST chunk):\n"
    "{previous_summary}\n\n"
    "GEMINI AUDIO ANALYSIS OF THE SAME 3-MINUTE CLIP:\n"
    "{gemini_analysis}\n\n"
    "NEW TRANSCRIPT:\n"
    "{transcript}\n\n"
    "Respond ONLY with valid JSON (no markdown, no backticks, no preamble):\n"
    '{{\n'
    '  "voice_profiles": {{\n'
    '    "<speaker_id>": {{\n'
    '      "gender": "male|female|child|unknown",\n'
    '      "estimated_age": "infant|child|teen|20s|30s|40s|50s+|unknown",\n'
    '      "probable_name": "<name or null — ONLY if you hear them addressed by name>",\n'
    '      "voice_description": "<brief description: pitch, accent, speaking style>",\n'
    '      "emotional_state": "<current emotional state: calm, excited, frustrated, etc>",\n'
    '      "role": "<relationship role if evident: parent, child, friend, spouse, etc>"\n'
    '    }}\n'
    '  }},\n'
    '  "names_mentioned": ["<list of all proper names spoken in this transcript>"],\n'
    '  "topics": ["<list of 2-4 main topics discussed>"],\n'
    '  "summary": "<3-4 sentence narrative summary of JUST this 3-minute chunk>",\n'
    '  "running_summary": "<3-4 sentence summary combining the previous summary with this new chunk — what is the OVERALL story so far?>",\n'
    '  "notable_events": ["<timestamped notable things: someone arrived, left, name mentioned, topic changed, emotion shifted>"],\n'
    '  "speaker_count": <number of distinct speakers you can identify>,\n'
    '  "corrections": ["<anything that contradicts the previous summary or voice profiles>"]\n'
    '}}'
)

# --- DEEP CYCLE: Gemini gets the full 3-min audio ---
DEEP_GEMINI_PROMPT = (
    "You are analyzing a 3-minute audio recording from a room with a smart speaker.\n"
    "This is a LONG clip — listen carefully to the full duration.\n\n"
    "CURRENT UNDERSTANDING OF THE ROOM:\n"
    "{room_state}\n\n"
    "Focus on:\n"
    "1. How many DISTINCT voices do you hear? Describe each: gender, approximate age, pitch, accent.\n"
    "2. Background sounds: what non-speech sounds are present? Do they change during the clip?\n"
    "3. Emotional arc: how does the mood change over the 3 minutes?\n"
    "4. Any names spoken aloud? Who says them and roughly when?\n"
    "5. What is happening physically? (movement, doors, objects, crying, cooking, etc.)\n\n"
    "Use relative timestamps (~0:30, ~1:15, ~2:45) to indicate WHEN things happen.\n\n"
    "Reply in this format:\n"
    "VOICES: [number] — [description of each voice]\n"
    "NAMES_HEARD: [name — who said it — ~timestamp]\n"
    "BACKGROUND: [sounds with ~timestamps]\n"
    "EMOTIONAL_ARC: [how mood changes across the 3 minutes]\n"
    "PHYSICAL_ACTIVITY: [what's happening in the room]\n"
    "SCENE_SUMMARY: [2-3 sentence overview of the full 3 minutes]"
)


# =============================================================================
# CAPABILITY CLASS
# =============================================================================

class EnhancedListenerCapability(MatchingCapability):
    worker: AgentWorker = None
    capability_worker: CapabilityWorker = None

    # --- Core ---
    is_running: bool = False
    session_id: str = ""
    session_start_time: float = 0.0

    # --- The Mental Model (replaces flat context list) ---
    room_state: dict = {}               # Canonical: rewritten each cycle
    conversation_summary: str = ""       # Running summary: chains across 3-min blocks
    recent_speech: list = []             # Rolling buffer of last N speech lines
    corrections: list = []               # Recent corrections detected
    enhanced_context: list = []          # Full log for dashboard only
    event_log: list = []                 # Rolling timestamped notable events
    voice_profiles: dict = {}            # {speaker_id: {gender, age, name, voice, emotion, role}}
    history_chunks: list = []            # List of 3-min summaries: [{cycle, ts, summary, topics, ...}]
    _field_last_seen: dict = {}          # Staleness: {field: cycle_id}

    # --- Conversation ---
    conversation_history: list = []
    recent_agent_speech: list = []

    # --- Transcript Dedup ---
    previous_transcript_lines: list = []

    # --- Engagement ---
    engagement_state: str = "IDLE"
    state_entered_time: float = 0.0
    last_engagement_time: float = 0.0
    last_speak_time: float = 0.0

    # --- Stats ---
    deepgram_calls: int = 0
    gemini_calls: int = 0
    analysis_cycle_id: int = 0
    deep_cycle_id: int = 0
    _cycle_data: dict = {}

    #{{register capability}}

    def call(self, worker: AgentWorker):
        self.worker = worker
        self.capability_worker = CapabilityWorker(self.worker)
        now = time.time()
        self.is_running = True
        self.session_id = f"el_{int(now)}"
        self.session_start_time = now
        self.room_state = {
            "speaker_count": 0, "speaker_count_confidence": "low",
            "speakers": "No one detected yet.",
            "background": "Unknown — just started listening.",
            "activity": "Unknown", "mood": "Unknown",
            "known_names": [],
            "last_updated": "00:00",
        }
        self.conversation_summary = "No conversation yet — just started listening."
        self.recent_speech = []
        self.corrections = []
        self.enhanced_context = []
        self.event_log = []
        self.voice_profiles = {}
        self.history_chunks = []
        self._field_last_seen = {"background": 0, "speakers": 0, "activity": 0, "mood": 0}
        self.conversation_history = []
        self.recent_agent_speech = []
        self.previous_transcript_lines = []
        self.engagement_state = "IDLE"
        self.state_entered_time = now
        self.last_engagement_time = 0.0
        self.last_speak_time = 0.0
        self.deepgram_calls = 0
        self.gemini_calls = 0
        self.analysis_cycle_id = 0
        self.deep_cycle_id = 0
        self._cycle_data = {}
        self.worker.session_tasks.create(self.run_main())

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def speak(self, text: str):
        self.log(f"[SPEAK] {text}")
        self.recent_agent_speech.append((time.time(), text))
        cutoff = time.time() - 90
        self.recent_agent_speech = [(t, s) for t, s in self.recent_agent_speech if t > cutoff]
        self._add_to_log("YOUR_SPEECH", text)
        await self.capability_worker.text_to_speech(text, ABILITY_VOICE_ID)
        self.last_speak_time = time.time()

    def log(self, msg: str):
        self.worker.editor_logging_handler.info(f"[EL] {msg}")

    def log_error(self, msg: str):
        self.worker.editor_logging_handler.error(f"[EL] {msg}")

    def elapsed(self) -> float:
        return time.time() - self.session_start_time

    def timestamp(self) -> str:
        m, s = divmod(int(self.elapsed()), 60)
        return f"{m:02d}:{s:02d}"

    def wall_clock(self) -> str:
        t = time.localtime()
        h = t.tm_hour % 12 or 12
        return f"{h}:{t.tm_min:02d} {'AM' if t.tm_hour < 12 else 'PM'}"

    def fire_confirm(self, fn: str):
        async def _p():
            try:
                await self.capability_worker.play_from_audio_file(fn)
                self.log(f"[AUDIO] Played {fn}")
            except Exception as e:
                self.log_error(f"[AUDIO] {fn}: {e}")
        self.worker.session_tasks.create(_p())

    def _add_to_log(self, entry_type: str, data: str):
        """Append to the full log (for dashboard). NOT used for LLM context."""
        if not data or not data.strip():
            return
        self.enhanced_context.append({
            "elapsed": self.elapsed(), "ts": self.timestamp(),
            "type": entry_type, "data": data.strip()
        })

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def _post_dashboard(self, endpoint: str, payload: dict):
        async def _send():
            try:
                await asyncio.to_thread(requests.post,
                    f"{DASHBOARD_URL}/listener/{endpoint}", json=payload, timeout=3)
            except Exception:
                pass
        self.worker.session_tasks.create(_send())

    def _post_state_change(self, from_st: str, to_st: str, trigger: str, text: str = ""):
        now = time.time()
        secs = now - self.state_entered_time if self.state_entered_time else 0
        self.state_entered_time = now
        self.engagement_state = to_st
        self._post_dashboard("state_change", {
            "session_id": self.session_id, "timestamp": now,
            "wall_clock_time": self.wall_clock(),
            "from_state": from_st, "to_state": to_st,
            "trigger": trigger, "trigger_text": text[:200],
            "seconds_in_previous_state": round(secs, 1),
        })

    def _post_utterance(self, text: str, action: str, **kw):
        now = time.time()
        since = now - self.last_speak_time if self.last_speak_time else 999
        self._post_dashboard("utterance", {
            "session_id": self.session_id, "timestamp": now,
            "wall_clock_time": self.wall_clock(),
            "engagement_state": self.engagement_state,
            "utterance_text": text[:500], "action_taken": action,
            "wake_word_detected": kw.get("wake", False),
            "wake_word_request": kw.get("wake_req"),
            "is_cross_talk": kw.get("cross", False),
            "cross_talk_name": kw.get("cross_name"),
            "is_disengage": kw.get("diseng", False),
            "is_exit": kw.get("exit", False),
            "in_reply_window": since < REPLY_WINDOW_SECONDS,
            "seconds_since_agent_spoke": round(since, 1),
            "looks_directed": kw.get("directed", False),
            "directed_reason": kw.get("reason"),
            "correction_detected": kw.get("correction", False),
            "correction_text": kw.get("correction_text"),
        })

    def _post_response(self, ui: str, resp: str, ctx_c: int, hist_n: int, llm_s: float, tot_s: float):
        self._post_dashboard("response", {
            "session_id": self.session_id, "timestamp": time.time(),
            "wall_clock_time": self.wall_clock(),
            "user_input": ui[:500], "agent_response": resp[:500],
            "context_chars": ctx_c, "history_messages": hist_n,
            "llm_elapsed_seconds": round(llm_s, 2),
            "total_elapsed_seconds": round(tot_s, 2),
            "engagement_state": self.engagement_state,
        })

    def _post_heartbeat(self, buf_s: float):
        now = time.time()
        self._post_dashboard("heartbeat", {
            "session_id": self.session_id, "timestamp": now,
            "session_uptime_seconds": round(now - self.session_start_time, 1),
            "engagement_state": self.engagement_state,
            "seconds_since_last_speak": round(now - self.last_speak_time, 1) if self.last_speak_time else 999,
            "seconds_since_last_engagement": round(now - self.last_engagement_time, 1) if self.last_engagement_time else 999,
            "analysis_cycle_id": self.analysis_cycle_id,
            "deepgram_calls": self.deepgram_calls, "gemini_calls": self.gemini_calls,
            "context_entries": len(self.enhanced_context),
            "conversation_turns": len(self.conversation_history) // 2,
            "audio_buffer_seconds": round(buf_s, 1),
            "known_names": list(self.room_state.get("known_names", [])),
            "room_scene_summary": self.format_room_state()[:500],
            "event_count": len(self.event_log),
            "recent_events": self.event_log[-5:],
            "voice_profiles": dict(self.voice_profiles),
        })

    def _post_transcript_line(self, speaker_id: str, text: str, cycle_id: int, source: str = "fast_cycle"):
        """Fire for each individual transcript line that survives dedup + echo filter."""
        self._post_dashboard("transcript_line", {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "wall_clock_time": self.wall_clock(),
            "cycle_id": cycle_id,
            "speaker_id": speaker_id,
            "text": text[:500],
            "is_new": True,
            "source": source,
        })

    def _post_room_state(self, cycle_id: int, dg_speakers: set, gemini_count: int | None, fusion_note: str):
        """Fire after every fusion step rewrites the canonical room state."""
        self._post_dashboard("room_state", {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "wall_clock_time": self.wall_clock(),
            "cycle_id": cycle_id,
            "speaker_count": self.room_state.get("speaker_count", 0),
            "speaker_count_confidence": self.room_state.get("speaker_count_confidence", "low"),
            "speaker_count_source": "fused" if dg_speakers and gemini_count else
                                    "deepgram_confirmed" if dg_speakers else "gemini_only",
            "speakers_description": self.room_state.get("speakers", "")[:300],
            "known_names": self.room_state.get("known_names", []),
            "background": self.room_state.get("background", "")[:200],
            "activity": self.room_state.get("activity", "")[:200],
            "mood": self.room_state.get("mood", "")[:200],
            "corrections": self.corrections[-5:],
            "dg_speaker_ids": sorted(dg_speakers) if dg_speakers else [],
            "gemini_speaker_count": gemini_count,
            "fusion_notes": fusion_note,
            "voice_profiles": dict(self.voice_profiles),
            "recent_events": self.event_log[-5:],
        })

    def _post_deep_cycle(self, dcid: int, audio_s: float, dg_s: float, sum_s: float,
                          total_s: float, speaker_count: int, speaker_ids: list,
                          names: list, raw_transcript: str, summary: str,
                          correction: str | None):
        """Fire after every deep analysis cycle completes."""
        self._post_dashboard("deep_cycle", {
            "session_id": self.session_id,
            "deep_cycle_id": dcid,
            "timestamp": time.time(),
            "wall_clock_time": self.wall_clock(),
            "audio_seconds": round(audio_s, 1),
            "deepgram_elapsed_seconds": round(dg_s, 2),
            "summary_elapsed_seconds": round(sum_s, 2),
            "total_elapsed_seconds": round(total_s, 2),
            "speaker_count": speaker_count,
            "speaker_ids": speaker_ids,
            "names_discovered": names,
            "raw_transcript": raw_transcript[:5000],
            "conversation_summary": summary[:1000],
            "speaker_count_correction": correction,
            "voice_profiles": dict(self.voice_profiles),
            "history_chunk_count": len(self.history_chunks),
            "running_summary": self.conversation_summary[:1000],
        })

    # =========================================================================
    # ECHO FILTERING + TRANSCRIPT DEDUP (unchanged)
    # =========================================================================

    def filter_echo(self, transcript: str) -> str:
        if not self.recent_agent_speech:
            return transcript
        agent_words = set()
        for _, s in self.recent_agent_speech:
            for w in s.lower().split():
                agent_words.add(w.strip(".,!?;:'\""))
        out = []
        for line in transcript.split("\n"):
            if not line.strip():
                continue
            tp = line.split(": ", 1)[-1] if ": " in line else line
            words = [w.strip(".,!?;:'\"") for w in tp.lower().split()]
            if not words:
                out.append(line)
                continue
            ratio = sum(1 for w in words if w in agent_words) / len(words)
            if ratio > 0.5:
                self.log(f"[ECHO] Removed ({ratio:.0%}): {line[:60]}")
            else:
                out.append(line)
        return "\n".join(out)

    def dedup_transcript(self, transcript: str) -> str:
        cur = [l for l in transcript.split("\n") if l.strip()]
        if not self.previous_transcript_lines:
            self.previous_transcript_lines = cur
            return transcript
        prev_sets = []
        for pl in self.previous_transcript_lines:
            tp = pl.split(": ", 1)[-1] if ": " in pl else pl
            prev_sets.append(set(w.strip(".,!?;:'\"").lower() for w in tp.split() if w.strip()))
        new = []
        for line in cur:
            tp = line.split(": ", 1)[-1] if ": " in line else line
            lw = set(w.strip(".,!?;:'\"").lower() for w in tp.split() if w.strip())
            if not lw:
                continue
            dup = any(ps and len(lw & ps) / max(len(lw), len(ps)) > 0.7 for ps in prev_sets)
            if not dup:
                new.append(line)
        self.previous_transcript_lines = cur
        if new:
            self.log(f"[DEDUP] {len(new)} new from {len(cur)}")
            return "\n".join(new)
        self.log(f"[DEDUP] All {len(cur)} duplicates")
        return ""

    # =========================================================================
    # AUDIO BUFFER
    # =========================================================================

    def build_wav(self, pcm: bytes) -> bytes:
        n = len(pcm)
        return struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36+n, b'WAVE', b'fmt ',
            16, 1, CHANNELS, SAMPLE_RATE, BYTES_PER_SECOND,
            CHANNELS*BITS_PER_SAMPLE//8, BITS_PER_SAMPLE, b'data', n) + pcm

    def strip_wav(self, ab: bytes) -> bytes:
        return ab[WAV_HEADER_SIZE:] if len(ab) > 4 and ab[:4] == b"RIFF" else ab

    def slice_window(self, ab: bytes, secs: int) -> bytes:
        pcm = self.strip_wav(ab)
        target = secs * BYTES_PER_SECOND
        sliced = pcm[-target:] if len(pcm) > target else pcm
        self.log(f"[SLICE] {len(sliced)/BYTES_PER_SECOND:.0f}s (target:{secs}s)")
        return self.build_wav(sliced)

    # =========================================================================
    # API CALLS
    # =========================================================================

    def _call_deepgram(self, wav: bytes) -> dict | None:
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"}
        params = {"model": DEEPGRAM_MODEL, "diarize": "true", "smart_format": "true",
                  "punctuate": "true", "utterances": "true"}
        try:
            r = requests.post(DEEPGRAM_URL, headers=headers, params=params, data=wav, timeout=30)
            if r.status_code == 200:
                self.deepgram_calls += 1
                return r.json()
            self.log_error(f"DG {r.status_code}: {r.text[:200]}")
        except Exception as e:
            self.log_error(f"DG error: {e}")
        return None

    def _extract_transcript(self, dg: dict) -> tuple[str, set]:
        """Returns (transcript_text, set_of_speaker_ids)."""
        speakers = set()
        try:
            utts = dg.get("results", {}).get("utterances", [])
            if utts:
                lines = []
                for u in utts:
                    sid = u.get("speaker", "?")
                    txt = u.get("transcript", "").strip()
                    if txt:
                        lines.append(f"Speaker {sid}: {txt}")
                        speakers.add(str(sid))
                return "\n".join(lines), speakers
            chs = dg.get("results", {}).get("channels", [])
            if chs:
                alts = chs[0].get("alternatives", [])
                if alts:
                    return alts[0].get("transcript", ""), set()
        except Exception as e:
            self.log_error(f"Transcript extract: {e}")
        return "", set()

    def _call_gemini_audio(self, wav: bytes, prompt: str) -> str:
        b64 = base64.b64encode(wav).decode("utf-8")
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": AUDIO_ANALYSIS_MODEL, "max_tokens": 600,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 200:
                self.gemini_calls += 1
                return r.json()["choices"][0]["message"]["content"]
            self.log_error(f"Gemini {r.status_code}: {r.text[:200]}")
        except Exception as e:
            self.log_error(f"Gemini error: {e}")
        return ""

    def _call_text_llm(self, prompt: str, max_tokens: int = 200) -> str:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, timeout=10,
                json={"model": SUMMARY_MODEL, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            self.log_error(f"TextLLM {r.status_code}: {r.text[:200]}")
        except Exception as e:
            self.log_error(f"TextLLM error: {e}")
        return ""

    def _call_conversation(self, user_input: str, history: list, sys_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_input}]
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, timeout=8,
                json={"model": CONVERSATION_MODEL, "messages": msgs, "max_tokens": CONVERSATION_MAX_TOKENS})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            self.log_error(f"Conv {r.status_code}: {r.text[:200]}")
        except Exception as e:
            self.log_error(f"Conv error: {e}")
        return ""

    # =========================================================================
    # ROOM STATE — The Mental Model
    # =========================================================================

    def format_room_state(self) -> str:
        """Format room_state as a readable paragraph for LLM injection."""
        rs = self.room_state
        parts = []
        parts.append(f"Speakers: {rs.get('speaker_count', '?')} "
                     f"(confidence: {rs.get('speaker_count_confidence', '?')})")
        parts.append(f"Who: {rs.get('speakers', 'Unknown')}")
        if rs.get("known_names"):
            parts.append(f"Names heard: {', '.join(rs['known_names'])}")
        if self.voice_profiles:
            parts.append("Voice profiles:")
            for sid, vp in self.voice_profiles.items():
                if isinstance(vp, dict):
                    name = vp.get("probable_name") or "unnamed"
                    gender = vp.get("gender", "?")
                    age = vp.get("estimated_age", "?")
                    emotion = vp.get("emotional_state", "?")
                    role = vp.get("role", "")
                    parts.append(f"  Speaker {sid}: {name} ({gender}, {age}) — {emotion}"
                                 f"{f', {role}' if role else ''}")
                else:
                    parts.append(f"  Speaker {sid}: {vp}")
        parts.append(f"Background: {rs.get('background', 'Unknown')}")
        parts.append(f"Activity: {rs.get('activity', 'Unknown')}")
        parts.append(f"Mood: {rs.get('mood', 'Unknown')}")
        parts.append(f"(Updated: {rs.get('last_updated', '?')})")
        return "\n".join(parts)

    def parse_gemini_structured(self, raw: str) -> dict:
        """
        Parse Gemini's structured output. If it follows format, extract fields.
        If not, treat the whole thing as an unstructured scene description.
        """
        result = {"raw": raw}
        raw_clean = raw.replace("**", "").replace("*", "").replace("##", "").strip()

        # Try to extract structured fields
        for line in raw_clean.split("\n"):
            line = line.strip()
            if not line:
                continue
            ul = line.upper()
            if ul.startswith("SPEAKER_COUNT:"):
                rest = line.split(":", 1)[1].strip()
                # Extract number and confidence
                m = re.search(r'(\d+)', rest)
                if m:
                    result["speaker_count"] = int(m.group(1))
                for conf in ["HIGH", "MEDIUM", "LOW"]:
                    if conf in rest.upper():
                        result["speaker_count_confidence"] = conf.lower()
                        break
            elif ul.startswith("SPEAKERS:"):
                result["speakers"] = line.split(":", 1)[1].strip()
            elif ul.startswith("BACKGROUND:"):
                rest = line.split(":", 1)[1].strip()
                # Strip trailing confidence if present
                for conf in ["| HIGH", "| MEDIUM", "| LOW"]:
                    if conf in rest.upper():
                        result["background_confidence"] = conf.split()[-1].lower()
                        rest = rest[:rest.upper().index(conf)].strip()
                result["background"] = rest
            elif ul.startswith("CHANGES:"):
                result["changes"] = line.split(":", 1)[1].strip()
            elif ul.startswith("ACTIVITY:"):
                result["activity"] = line.split(":", 1)[1].strip()
            elif ul.startswith("MOOD:"):
                result["mood"] = line.split(":", 1)[1].strip()
            elif ul.startswith("BACKGROUND_GONE:"):
                result["background_gone"] = line.split(":", 1)[1].strip()
            elif ul.startswith("NAMES_HEARD:"):
                result["names_heard"] = line.split(":", 1)[1].strip()

        # Fallback: if no structured fields found, use raw as description
        if "speakers" not in result and "activity" not in result:
            result["speakers"] = raw_clean[:300]
            result["activity"] = raw_clean[:300]

        self.log(f"[GEMINI-PARSE] Fields: {[k for k in result if k != 'raw']}")
        return result

    def fuse_and_update_room_state(self, dg_transcript: str, dg_speakers: set,
                                     gemini_parsed: dict):
        """
        FUSION: Cross-validate Deepgram + Gemini and REWRITE room_state.

        Deepgram is ground truth for: speaker count (it has word-level diarization).
        Gemini fills in: demographics, mood, background, activity.
        Names: extracted from Deepgram transcript with strict patterns.

        STALENESS: If Gemini doesn't mention a field for 3 cycles, clear it.
        EVENT LOG: Notable changes get timestamped entries.
        """
        ts = self.timestamp()
        cid = self.analysis_cycle_id

        # --- Speaker count: Deepgram wins when they disagree ---
        dg_count = len(dg_speakers) if dg_speakers else 0
        gem_count = gemini_parsed.get("speaker_count", 0)
        old_count = self.room_state.get("speaker_count", 0)

        if dg_count > 0 and gem_count > 0:
            if dg_count == gem_count:
                count = dg_count
                conf = "high"
            elif abs(dg_count - gem_count) == 1:
                count = max(dg_count, gem_count)
                conf = "medium"
            else:
                count = dg_count  # Trust Deepgram
                conf = "low"
                self.log(f"[FUSION] Speaker count disagree: DG={dg_count} Gem={gem_count} → using DG")
        elif dg_count > 0:
            count = dg_count
            conf = "medium"
        elif gem_count > 0:
            count = gem_count
            conf = "low"
        else:
            count = self.room_state.get("speaker_count", 0)
            conf = "low"

        # --- EVENT: speaker count changed ---
        if count != old_count and count > 0:
            if count > old_count:
                self._add_event(f"Speaker count increased to {count} (was {old_count})")
            elif count < old_count:
                self._add_event(f"Speaker count decreased to {count} (was {old_count})")

        # --- Names: extract from transcript ---
        new_names = self._extract_names(dg_transcript)
        known = set(self.room_state.get("known_names", []))
        for name in new_names:
            if name not in known:
                self._add_event(f"Name '{name}' mentioned for the first time")
        known.update(new_names)

        # Also update KNOWN_NAMES set for cross-talk detection (lowercase)
        KNOWN_NAMES.update(n.lower() for n in new_names)

        # --- STALENESS: track when each field was last mentioned by Gemini ---
        STALE_CYCLES = 3  # Clear field after 3 cycles of Gemini not mentioning it

        new_bg = gemini_parsed.get("background")
        new_speakers = gemini_parsed.get("speakers")
        new_activity = gemini_parsed.get("activity")
        new_mood = gemini_parsed.get("mood")

        if new_bg:
            self._field_last_seen["background"] = cid
        if new_speakers:
            self._field_last_seen["speakers"] = cid
        if new_activity:
            self._field_last_seen["activity"] = cid
        if new_mood:
            self._field_last_seen["mood"] = cid

        # Decay stale fields
        old_bg = self.room_state.get("background", "")
        if not new_bg and (cid - self._field_last_seen.get("background", 0)) >= STALE_CYCLES:
            if old_bg and old_bg != "Quiet" and old_bg != "Unknown — just started listening.":
                self._add_event(f"Background sound no longer detected: {old_bg[:60]}")
                self.log(f"[STALE] Background cleared after {STALE_CYCLES} cycles: {old_bg[:40]}")
            new_bg = "Quiet"

        # --- EVENT: background changed ---
        if new_bg and old_bg and new_bg != old_bg:
            old_lower = old_bg.lower()
            new_lower = new_bg.lower()
            if old_lower != new_lower:
                self._add_event(f"Background changed: '{old_bg[:40]}' → '{new_bg[:40]}'")

        # --- REWRITE room_state ---
        self.room_state = {
            "speaker_count": count,
            "speaker_count_confidence": conf,
            "speakers": new_speakers or self.room_state.get("speakers", "Unknown"),
            "background": new_bg or self.room_state.get("background", ""),
            "activity": new_activity or self.room_state.get("activity", ""),
            "mood": new_mood or self.room_state.get("mood", ""),
            "known_names": sorted(known),
            "voice_profiles": dict(self.voice_profiles),
            "last_updated": ts,
        }

        self.log(f"[ROOM-STATE] Rewritten at {ts}: {count} speakers ({conf}), "
                 f"names={sorted(known)}")

    # =========================================================================
    # EVENT LOG — Timestamped notable things
    # =========================================================================

    MAX_EVENTS = 30

    def _add_event(self, description: str):
        """Add a timestamped event to the rolling event log."""
        entry = f"[{self.timestamp()}] {description}"
        self.event_log.append(entry)
        self.event_log = self.event_log[-self.MAX_EVENTS:]
        self.log(f"[EVENT] {entry}")
        self._add_to_log("EVENT", description)

    def format_event_log(self) -> str:
        """Format the last 15 events for LLM context."""
        if not self.event_log:
            return "(No notable events yet.)"
        return "\n".join(self.event_log[-15:])

    def _extract_names(self, transcript: str) -> set:
        """
        Pull proper names from transcript. VERY strict to avoid garbage.
        Only accept names in natural address patterns:
          - "Hey Shannon", "Hi PJ", "Hello Chris"
          - "The name is Grayson"
          - "someone named Grayson"
          - "[Name], can you..." / "[Name], come here"
          - "this is [Name]" / "I'm [Name]"
        """
        names = set()

        # Massive exclusion list — common words that appear capitalized in transcripts
        EXCLUDE = {
            # Common words at sentence starts
            "the", "and", "but", "yeah", "okay", "what", "how", "open", "home",
            "speaker", "mhmm", "like", "well", "right", "hey", "alright",
            "basically", "really", "so", "no", "yes", "sure", "oh", "um",
            "just", "get", "got", "into", "she", "her", "him", "his", "he",
            "they", "them", "this", "that", "here", "there", "where", "when",
            "why", "who", "can", "could", "would", "should", "will", "going",
            "gonna", "want", "need", "know", "think", "feel", "look", "let",
            "put", "take", "come", "make", "give", "tell", "say", "said",
            "mean", "start", "stop", "okay", "fine", "good", "bad", "all",
            "not", "don", "didn", "doesn", "isn", "wasn", "aren", "won",
            "also", "maybe", "probably", "actually", "apparently",
            "because", "then", "than", "with", "from", "about", "over",
            "after", "before", "between", "through", "under", "again",
            "for", "are", "was", "were", "been", "being", "have", "has",
            "had", "having", "you", "your", "my", "our", "their", "its",
            "some", "any", "every", "each", "both", "few", "more", "most",
            "other", "only", "very", "still", "already", "even",
            "now", "out", "back", "down", "off", "away", "around",
            "kind", "sort", "pretty", "quite", "rather", "too", "enough",
            "morning", "night", "today", "yesterday", "tomorrow",
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "june", "january", "february",
            "feeding", "starting", "leaving", "change", "doing",
            "wait", "hang", "chain", "fun", "new",
        }

        # Pattern 1: Explicit naming — "The name is [Name]", "I'm [Name]", "called [Name]"
        for m in re.finditer(r'(?:the\s+name\s+is|my\s+name\s+is|i\'?m|called|named)\s+([A-Z][a-z]{2,})', transcript, re.IGNORECASE):
            name = m.group(1).strip()
            if name.lower() not in EXCLUDE:
                names.add(name)
                self.log(f"[NAMES] Explicit: '{name}'")

        # Pattern 2: Direct address — "Hey [Name]", "Hi [Name]", "[Name], can you"
        for m in re.finditer(r'(?:hey|hi|hello|yo|oh)\s+([A-Z][a-z]{2,})', transcript):
            name = m.group(1).strip()
            if name.lower() not in EXCLUDE:
                names.add(name)
                self.log(f"[NAMES] Address: '{name}'")

        # Pattern 3: Vocative comma — "[Name], ..." at start of utterance
        for m in re.finditer(r'(?:^|\n)\s*(?:Speaker\s+\d+:\s*)?([A-Z][a-z]{2,}),\s', transcript):
            name = m.group(1).strip()
            if name.lower() not in EXCLUDE:
                names.add(name)
                self.log(f"[NAMES] Vocative: '{name}'")

        # Pattern 4: Third-person reference — "thing is [Name]", "[Name] can only come"
        for m in re.finditer(r'(?:is|and|with)\s+([A-Z][A-Z]|[A-Z][a-z]{1,})\s+(?:can|will|is|was|has|had|does|could|would|should)', transcript):
            name = m.group(1).strip()
            if name.lower() not in EXCLUDE and len(name) >= 2:
                names.add(name)
                self.log(f"[NAMES] Reference: '{name}'")

        return names

    # =========================================================================
    # CORRECTION DETECTION
    # =========================================================================

    def detect_corrections(self, text: str) -> bool:
        """
        Check if the user is correcting something the agent said.
        If so, add to corrections list so the LLM knows not to repeat the error.
        """
        lower = text.lower()
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, lower):
                correction = f"[{self.timestamp()}] User said: \"{text[:150]}\""
                self.corrections.append(correction)
                # Keep last 5 corrections
                self.corrections = self.corrections[-5:]
                self.log(f"[CORRECTION] Detected: '{text[:60]}'")
                self._add_to_log("CORRECTION", text)
                return True
        return False

    # =========================================================================
    # CONTEXT BUILDING — What the conversation LLM sees
    # =========================================================================

    def build_context(self) -> str:
        """
        Build the full system prompt for the conversation LLM.
        When user says "open home", this is what the LLM sees:
        - Room state (current snapshot)
        - Voice profiles (who's here, gender, name, emotion)
        - History chain (3-min summary blocks — the full story)
        - Active conversation framing (what you've been discussing)
        - Recent speech (last 15 seconds for immediacy)
        - Event log (timestamped notable things)
        - Corrections
        """
        room = self.format_room_state()
        vp = self._format_voice_profiles_for_prompt()
        chain = self._format_history_chain()
        convo = self._format_conversation_context()
        # Use fewer recent speech lines — the history chain covers the past
        speech = "\n".join(self.recent_speech[-8:]) if self.recent_speech else "(Quiet.)"
        events = self.format_event_log()
        corr = "\n".join(self.corrections) if self.corrections else "(None.)"

        return PERSONALITY_PROMPT.format(
            room_state=room,
            voice_profiles=vp,
            history_chain=chain,
            conversation_context=convo,
            recent_speech=speech,
            event_log=events,
            corrections=corr,
        )

    def _format_history_chain(self) -> str:
        """Format the chain of 3-minute summary blocks for LLM context."""
        if not self.history_chunks:
            return "(No history yet — listening just started.)"
        lines = []
        # Show last 5 chunks (15 min of history)
        for chunk in self.history_chunks[-5:]:
            ts = chunk.get("timestamp", "?")
            summary = chunk.get("running_summary") or chunk.get("summary", "?")
            topics = ", ".join(chunk.get("topics", [])) or "general"
            names = ", ".join(chunk.get("names_mentioned", [])) or "none"
            lines.append(f"[{ts}] Topics: {topics} | Names: {names}\n  {summary}")
        return "\n\n".join(lines)

    def _format_conversation_context(self) -> str:
        """
        Build a framing section for the active conversation.
        Tells the LLM: how many turns, what was discussed, when engagement started.
        Returns empty string if no conversation history (first turn).
        """
        hist = self.conversation_history
        if not hist:
            return ""  # First turn — no framing needed

        turns = len(hist) // 2
        if turns == 0:
            return ""

        # Build a quick digest of what's been discussed
        lines = []
        lines.append(f"YOUR ACTIVE CONVERSATION ({turns} turn{'s' if turns != 1 else ''} so far):")

        # How long ago did this conversation start?
        if self.last_engagement_time:
            ago = time.time() - self.last_engagement_time
            if ago < 60:
                lines.append(f"Engaged {int(ago)}s ago.")
            else:
                lines.append(f"Engaged {int(ago/60)}m {int(ago%60)}s ago.")

        # Show a compressed digest: first question, and last 2 exchanges
        first_q = None
        for msg in hist:
            if msg.get("role") == "user":
                first_q = msg["content"]
                break

        if first_q and turns > 2:
            lines.append(f"They first asked: \"{first_q[:100]}\"")

        # Last 2 exchanges as recap
        recent = hist[-4:]  # Last 2 user+assistant pairs
        if recent:
            lines.append("Recent exchanges:")
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "You"
                text = msg.get("content", "")[:120]
                lines.append(f"  {role}: {text}")

        lines.append("")  # Blank line before next section
        return "\n".join(lines) + "\n"

    def _handle_engagement_gap(self):
        """
        Called on IDLE → ENGAGED. If there's a significant gap since the last
        conversation, trim old history and insert a context marker so the LLM
        knows time has passed.
        """
        if not self.conversation_history:
            return  # First engagement — nothing to handle

        if not self.last_engagement_time:
            return  # Never engaged before

        gap = time.time() - self.last_engagement_time

        if gap > 120:
            # Significant gap (>2 min): trim to last 4 messages, add gap marker
            gap_min = int(gap / 60)
            self.log(f"[GAP] {gap_min}m since last engagement — trimming history, adding marker")

            # Keep just the last 2 exchanges for reference
            old = self.conversation_history[-4:] if len(self.conversation_history) >= 4 else self.conversation_history[:]
            self.conversation_history = old

            # Insert a gap marker as an assistant message so the LLM sees the break
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[{gap_min} minutes passed since our last exchange. I continued "
                           f"listening to the room in the background.]"
            })
        elif gap > 30:
            # Short gap (30s-2min): just add a brief marker
            self.log(f"[GAP] {int(gap)}s since last engagement — adding brief marker")
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[Brief pause — {int(gap)} seconds since we last spoke.]"
            })

    # =========================================================================
    # DETECTION — Whitelist approach
    # =========================================================================

    def has_wake_word(self, t: str) -> bool:
        return bool(WAKE_WORD_REGEX.search(t))

    def strip_wake_word(self, t: str) -> str:
        m = WAKE_WORD_REGEX.search(t)
        return m.group(1).strip().lstrip(",").lstrip(".").lstrip("!").strip() if m else t.strip()

    def is_disengage(self, t: str) -> bool:
        lower = t.lower().strip()
        return any(s in lower for s in DISENGAGE_SIGNALS)

    def is_exit(self, t: str) -> bool:
        lower = t.lower().strip()
        return any(k in lower for k in EXIT_KEYWORDS)

    def is_direct_address(self, t: str) -> bool:
        """Check for phrases that always mean the user is talking to the speaker."""
        lower = t.lower()
        return any(p in lower for p in DIRECT_ADDRESS_PHRASES)

    def in_reply_window(self) -> bool:
        if not self.last_speak_time:
            return False
        elapsed = time.time() - self.last_speak_time
        return elapsed < REPLY_WINDOW_SECONDS

    def should_respond(self, text: str) -> tuple[bool, str]:
        """
        WHITELIST: only respond when there's positive evidence of being addressed.
        Default is silence. This is the core engagement philosophy.
        """
        # 1. Direct address phrases always trigger
        if self.is_direct_address(text):
            return True, "direct_address"

        # 2. Reply window — conversation momentum
        if self.in_reply_window():
            return True, "reply_window"

        # 3. Question containing "you" — probably addressing the speaker
        if "?" in text and "you" in text.lower():
            return True, "question_with_you"

        # 4. Very short + question mark — "thoughts?" "well?" "and?"
        if "?" in text and len(text.split()) <= 4:
            return True, "short_question"

        # Default: not addressed to us
        return False, "not_addressed"

    # =========================================================================
    # RESPONSE
    # =========================================================================

    async def respond(self, user_input: str):
        t0 = time.time()
        sys_prompt = self.build_context()
        hist = self.conversation_history[-MAX_HISTORY_MESSAGES:]
        ctx_len = len(sys_prompt)

        self.log(f"[RESPOND] '{user_input[:80]}' | ctx:{ctx_len} | hist:{len(hist)}")

        response = await asyncio.to_thread(self._call_conversation, user_input, hist, sys_prompt)
        llm_t = time.time() - t0
        self.log(f"[RESPOND] LLM {llm_t:.1f}s: {response}")

        if not response or not response.strip():
            response = "I've got nothing useful on that one."

        await self.speak(response)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        total_t = time.time() - t0
        self.log(f"[RESPOND] Total:{total_t:.1f}s LLM:{llm_t:.1f}s")
        self._post_response(user_input, response, ctx_len, len(hist), llm_t, total_t)

    # =========================================================================
    # TASK 1: FAST ANALYSIS (every 15s) — "What just changed?"
    # =========================================================================

    async def fast_analysis_loop(self):
        self.log("[FAST] Loop started.")
        await self.worker.session_tasks.sleep(ANALYSIS_INTERVAL_SECONDS)

        while self.is_running:
            try:
                ab = self.capability_worker.get_audio_recording()
                if not ab or len(ab) < 2000:
                    await self.worker.session_tasks.sleep(ANALYSIS_INTERVAL_SECONDS)
                    continue

                pcm = len(self.strip_wav(ab))
                buf_s = pcm / BYTES_PER_SECOND
                self.analysis_cycle_id += 1
                cid = self.analysis_cycle_id

                self.log(f"[FAST] {'='*30} Cycle {cid} at {self.timestamp()} | "
                         f"Buf:{buf_s:.0f}s | State:{self.engagement_state}")

                window = self.slice_window(ab, AUDIO_WINDOW_SECONDS)

                # Build Gemini prompt with current room state + corrections + recent DG transcript
                corr_text = ""
                if self.corrections:
                    corr_text = "USER CORRECTIONS (they told us we were wrong):\n" + "\n".join(self.corrections[-3:])
                # Include recent speech lines so Gemini can cross-reference what DG heard
                dg_context = "\n".join(self.recent_speech[-10:]) if self.recent_speech else "(No speech detected yet.)"
                gemini_prompt = GEMINI_STRUCTURED_PROMPT.format(
                    room_state=self.format_room_state(),
                    corrections=corr_text,
                    dg_transcript=dg_context,
                )

                # Init cycle data for dashboard
                self._cycle_data[cid] = {
                    "audio_window_seconds": min(buf_s, AUDIO_WINDOW_SECONDS),
                    "dg_raw": "", "dg_dedup": "", "dg_echo": "", "dg_final": "",
                    "dg_elapsed": 0, "dg_speakers": set(),
                    "gemini_raw": "", "gemini_final": "", "gemini_elapsed": 0,
                    "gemini_prompt": gemini_prompt[:500],
                    "dg_done": False, "gemini_done": False,
                }

                # Fire both
                self.worker.session_tasks.create(self._fast_deepgram(window, cid))
                self.worker.session_tasks.create(self._fast_gemini(window, cid, gemini_prompt))

                self._post_heartbeat(buf_s)

            except Exception as e:
                self.log_error(f"[FAST] Error: {e}")

            await self.worker.session_tasks.sleep(ANALYSIS_INTERVAL_SECONDS)

    async def _fast_deepgram(self, wav: bytes, cid: int):
        try:
            t0 = time.time()
            result = await asyncio.to_thread(self._call_deepgram, wav)
            elapsed = time.time() - t0
            cd = self._cycle_data.get(cid, {})
            cd["dg_elapsed"] = round(elapsed, 2)

            if not result:
                cd["dg_done"] = True
                self._check_fast_complete(cid)
                return

            raw, speakers = self._extract_transcript(result)
            cd["dg_raw"] = raw
            cd["dg_speakers"] = speakers

            if not raw or not raw.strip():
                self.log(f"[DG] Empty ({elapsed:.1f}s)")
                cd["dg_done"] = True
                self._check_fast_complete(cid)
                return

            self.log(f"[DG] Raw ({elapsed:.1f}s, {len(speakers)} speakers):")
            for l in raw.split("\n"):
                self.log(f"[DG]   {l}")

            deduped = self.dedup_transcript(raw)
            cd["dg_dedup"] = deduped
            if not deduped.strip():
                cd["dg_done"] = True
                self._check_fast_complete(cid)
                return

            filtered = self.filter_echo(deduped)
            cd["dg_echo"] = filtered
            if filtered and filtered.strip():
                cd["dg_final"] = filtered
                self._add_to_log("TRANSCRIPT", filtered)

                # Add to recent_speech buffer AND post each line to dashboard
                for line in filtered.split("\n"):
                    if line.strip():
                        self.recent_speech.append(f"[{self.timestamp()}] {line.strip()}")
                        # Extract speaker ID and text for dashboard
                        m = re.match(r'Speaker (\d+): (.+)', line.strip())
                        if m:
                            self._post_transcript_line(m.group(1), m.group(2), cid, "fast_cycle")
                        else:
                            self._post_transcript_line("?", line.strip(), cid, "fast_cycle")
                self.recent_speech = self.recent_speech[-MAX_RECENT_SPEECH:]

                # Check for corrections
                self.detect_corrections(filtered)

                self.log(f"[DG] ✓ Final ({len(filtered)} chars)")

            cd["dg_done"] = True
            self._check_fast_complete(cid)
        except Exception as e:
            self.log_error(f"[DG] Error: {e}")

    async def _fast_gemini(self, wav: bytes, cid: int, prompt: str):
        try:
            t0 = time.time()
            raw = await asyncio.to_thread(self._call_gemini_audio, wav, prompt)
            elapsed = time.time() - t0
            cd = self._cycle_data.get(cid, {})
            cd["gemini_elapsed"] = round(elapsed, 2)

            if not raw or not raw.strip():
                cd["gemini_done"] = True
                self._check_fast_complete(cid)
                return

            cd["gemini_raw"] = raw
            self.log(f"[GEMINI] ({elapsed:.1f}s):")
            for l in raw.split("\n"):
                if l.strip():
                    self.log(f"[GEMINI]   {l.strip()}")

            lower = raw.lower()
            if "no significant change" in lower:
                self.log("[GEMINI] No changes")
                cd["gemini_done"] = True
                self._check_fast_complete(cid)
                return

            # Parse structured output
            parsed = self.parse_gemini_structured(raw)
            cd["gemini_parsed"] = parsed
            cd["gemini_final"] = raw
            self._add_to_log("ENVIRONMENT", raw)

            cd["gemini_done"] = True
            self._check_fast_complete(cid)
        except Exception as e:
            self.log_error(f"[GEMINI] Error: {e}")

    def _check_fast_complete(self, cid: int):
        cd = self._cycle_data.get(cid, {})
        if not cd.get("dg_done") or not cd.get("gemini_done"):
            return

        # --- FUSION: Cross-validate and rewrite room_state ---
        dg_final = cd.get("dg_final", "")
        dg_speakers = cd.get("dg_speakers", set())
        gemini_parsed = cd.get("gemini_parsed", {})
        gem_count = gemini_parsed.get("speaker_count") if gemini_parsed else None

        # Build fusion note for dashboard
        dg_count = len(dg_speakers) if dg_speakers else 0
        if dg_count and gem_count:
            if dg_count == gem_count:
                fusion_note = f"agree:{dg_count}"
            else:
                fusion_note = f"disagree:dg={dg_count},gem={gem_count},using_dg"
        elif dg_count:
            fusion_note = f"dg_only:{dg_count}"
        elif gem_count:
            fusion_note = f"gem_only:{gem_count}"
        else:
            fusion_note = "no_data"

        if gemini_parsed or dg_speakers:
            self.fuse_and_update_room_state(dg_final, dg_speakers, gemini_parsed)

        # --- EVENT: Gemini reported sounds disappeared ---
        bg_gone = gemini_parsed.get("background_gone", "") if gemini_parsed else ""
        if bg_gone and "none" not in bg_gone.lower() and "n/a" not in bg_gone.lower():
            self._add_event(f"Sounds no longer heard: {bg_gone[:80]}")

        # --- EVENT: Gemini heard names ---
        names_heard = gemini_parsed.get("names_heard", "") if gemini_parsed else ""
        if names_heard and "none" not in names_heard.lower() and "n/a" not in names_heard.lower():
            self.log(f"[GEMINI-NAMES] {names_heard}")
            # Extract potential names from Gemini's response too
            for m in re.finditer(r'([A-Z][a-z]{2,})', names_heard):
                name = m.group(1)
                known = set(self.room_state.get("known_names", []))
                if name not in known and name.lower() not in {"speaker", "none", "high", "medium", "low"}:
                    known.add(name)
                    self.room_state["known_names"] = sorted(known)
                    KNOWN_NAMES.add(name.lower())
                    self._add_event(f"Gemini heard name: '{name}'")

        status = "both_returned" if cd.get("dg_raw") and cd.get("gemini_raw") else \
                 "dg_only" if cd.get("dg_raw") else \
                 "gemini_only" if cd.get("gemini_raw") else "both_failed"

        self.log(f"[FAST] ✓ Cycle {cid} — {status} | Room: {self.room_state.get('speaker_count')} speakers | Fusion: {fusion_note}")

        if self.engagement_state == "IDLE":
            self.fire_confirm("confirm.mp3")

        # --- DASHBOARD: room_state ---
        self._post_room_state(cid, dg_speakers, gem_count, fusion_note)

        # --- DASHBOARD: analysis_cycle (with new fusion fields) ---
        self._post_dashboard("analysis_cycle", {
            "session_id": self.session_id, "cycle_id": cid,
            "timestamp": time.time(), "wall_clock_time": self.wall_clock(),
            "audio_window_seconds": cd.get("audio_window_seconds", 0),
            "deepgram_raw_transcript": cd.get("dg_raw", "")[:2000],
            "deepgram_dedup_result": cd.get("dg_dedup", "")[:2000],
            "deepgram_echo_filtered": cd.get("dg_echo", "")[:2000],
            "deepgram_final_transcript": cd.get("dg_final", "")[:2000],
            "deepgram_elapsed_seconds": cd.get("dg_elapsed", 0),
            "gemini_scene_brief": cd.get("gemini_prompt", "")[:2000],
            "gemini_raw_analysis": cd.get("gemini_raw", "")[:2000],
            "gemini_final_analysis": cd.get("gemini_final", "")[:2000],
            "gemini_elapsed_seconds": cd.get("gemini_elapsed", 0),
            "pipeline_status": status,
            # New fusion fields
            "dg_speaker_ids": sorted(dg_speakers) if dg_speakers else [],
            "gemini_speaker_count": gem_count,
            "fusion_result": fusion_note,
            "room_state_snapshot": dict(self.room_state),
            "corrections_active": self.corrections[-5:],
        })

        # Cleanup
        for k in [k for k in self._cycle_data if k < cid - 3]:
            del self._cycle_data[k]

    # =========================================================================
    # TASK 2: DEEP ANALYSIS (every 3 min) — "What's the full picture?"
    # =========================================================================

    async def deep_analysis_loop(self):
        """
        Every 3 minutes: the INTELLIGENCE ENGINE.

        This is the only place latency DOESN'T matter. Nobody is waiting.
        So we use LLMs for everything: name extraction, voice profiling,
        emotional state, topic extraction, conversation summarization.

        Pipeline:
        1. Send 3-min audio to Deepgram (transcript) AND Gemini (scene analysis) — PARALLEL
        2. Send both results to a fast LLM for extraction: names, voice profiles,
           emotions, topics, chained summary
        3. Update voice_profiles, history_chunks, event_log, room_state
        4. Post everything to dashboard
        """
        self.log("[DEEP] Loop started. Waiting for first 3-minute buffer...")
        await self.worker.session_tasks.sleep(DEEP_ANALYSIS_INTERVAL_SECONDS)

        while self.is_running:
            try:
                ab = self.capability_worker.get_audio_recording()
                if not ab or len(ab) < 2000:
                    await self.worker.session_tasks.sleep(DEEP_ANALYSIS_INTERVAL_SECONDS)
                    continue

                self.deep_cycle_id += 1
                dcid = self.deep_cycle_id
                pcm_s = len(self.strip_wav(ab)) / BYTES_PER_SECOND
                self.log(f"[DEEP] {'='*50}")
                self.log(f"[DEEP] Deep cycle {dcid} at {self.timestamp()} | Buf:{pcm_s:.0f}s")

                deep_window = self.slice_window(ab, DEEP_ANALYSIS_INTERVAL_SECONDS)
                t0 = time.time()

                # ==================================================================
                # STEP 1: Send 3-min audio to BOTH Deepgram AND Gemini in parallel
                # ==================================================================
                self.log(f"[DEEP] Step 1: Parallel Deepgram + Gemini on {DEEP_ANALYSIS_INTERVAL_SECONDS}s")

                dg_result = [None]
                gemini_result = [""]
                dg_time = [0.0]
                gem_time = [0.0]

                async def _deep_dg():
                    t = time.time()
                    dg_result[0] = await asyncio.to_thread(self._call_deepgram, deep_window)
                    dg_time[0] = time.time() - t

                async def _deep_gem():
                    t = time.time()
                    prompt = DEEP_GEMINI_PROMPT.format(room_state=self.format_room_state())
                    gemini_result[0] = await asyncio.to_thread(
                        self._call_gemini_audio, deep_window, prompt)
                    gem_time[0] = time.time() - t

                self.worker.session_tasks.create(_deep_dg())
                self.worker.session_tasks.create(_deep_gem())

                # Wait for both — poll every 0.5s
                deadline = time.time() + 60  # 60s max wait
                while time.time() < deadline:
                    if dg_time[0] > 0 and gem_time[0] > 0:
                        break
                    await self.worker.session_tasks.sleep(0.5)

                dg_elapsed = dg_time[0]
                gem_elapsed = gem_time[0]
                self.log(f"[DEEP] Step 1 done: DG={dg_elapsed:.1f}s, Gemini={gem_elapsed:.1f}s")

                # --- Extract Deepgram transcript ---
                deep_transcript = ""
                deep_speakers = set()
                if dg_result[0]:
                    deep_transcript, deep_speakers = self._extract_transcript(dg_result[0])
                    self.log(f"[DEEP] DG: {len(deep_speakers)} speakers, {len(deep_transcript)} chars")

                # --- Gemini audio analysis ---
                gemini_analysis = gemini_result[0] or ""
                if gemini_analysis:
                    gemini_analysis = gemini_analysis.replace("**", "").replace("*", "").strip()
                    self.log(f"[DEEP] Gemini: {len(gemini_analysis)} chars")
                    for l in gemini_analysis.split("\n"):
                        if l.strip():
                            self.log(f"[DEEP-GEM]   {l.strip()[:120]}")

                if not deep_transcript.strip() and not gemini_analysis:
                    self.log(f"[DEEP] Both empty — skipping")
                    await self.worker.session_tasks.sleep(DEEP_ANALYSIS_INTERVAL_SECONDS)
                    continue

                # --- Post each deep transcript line to dashboard ---
                for line in deep_transcript.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r'Speaker (\d+): (.+)', line)
                    if m:
                        self._post_transcript_line(m.group(1), m.group(2), dcid, "deep_cycle")
                    else:
                        self._post_transcript_line("?", line, dcid, "deep_cycle")

                # ==================================================================
                # STEP 2: LLM extraction — names, profiles, emotions, summary
                # ==================================================================
                self.log(f"[DEEP] Step 2: LLM extraction from DG + Gemini")

                # Build the prompt with full context
                prev_summary = ""
                if self.history_chunks:
                    last = self.history_chunks[-1]
                    prev_summary = last.get("summary", "") or last.get("running_summary", "")
                if not prev_summary:
                    prev_summary = "(This is the first 3-minute chunk.)"

                vp_text = self._format_voice_profiles_for_prompt()

                extraction_prompt = DEEP_EXTRACTION_PROMPT.format(
                    voice_profiles=vp_text,
                    previous_summary=prev_summary,
                    gemini_analysis=gemini_analysis or "(Gemini analysis not available.)",
                    transcript=deep_transcript[:5000] or "(No transcript available — Gemini analysis only.)",
                )

                t_ext = time.time()
                extraction_raw = await asyncio.to_thread(
                    self._call_text_llm, extraction_prompt, 1200)
                ext_elapsed = time.time() - t_ext
                self.log(f"[DEEP] Extraction LLM: {ext_elapsed:.1f}s, {len(extraction_raw)} chars")

                # --- Parse JSON response ---
                extraction = self._parse_extraction_json(extraction_raw)

                if extraction:
                    self.log(f"[DEEP] Extraction parsed: {list(extraction.keys())}")

                    # --- Update voice profiles ---
                    new_vps = extraction.get("voice_profiles", {})
                    if new_vps:
                        old_vps = dict(self.voice_profiles)
                        for sid, vp in new_vps.items():
                            if isinstance(vp, dict):
                                old = old_vps.get(str(sid), {})
                                # Merge: keep old values if new is null/unknown
                                merged = {}
                                for key in ["gender", "estimated_age", "probable_name",
                                            "voice_description", "emotional_state", "role"]:
                                    new_val = vp.get(key)
                                    old_val = old.get(key, "") if isinstance(old, dict) else ""
                                    if new_val and str(new_val).lower() not in ("null", "none", "unknown", ""):
                                        merged[key] = new_val
                                    elif old_val and str(old_val).lower() not in ("null", "none", "unknown", ""):
                                        merged[key] = old_val
                                    else:
                                        merged[key] = new_val or ""
                                self.voice_profiles[str(sid)] = merged
                                self.log(f"[DEEP] Voice profile {sid}: {merged.get('probable_name', '?')} "
                                         f"({merged.get('gender', '?')}, {merged.get('estimated_age', '?')}) "
                                         f"— {merged.get('emotional_state', '?')}")

                                # Event: name newly assigned
                                old_name = old.get("probable_name") if isinstance(old, dict) else None
                                new_name = merged.get("probable_name")
                                if new_name and new_name != old_name and str(new_name).lower() not in ("null", "none", ""):
                                    self._add_event(f"Speaker {sid} identified as '{new_name}' ({merged.get('gender', '?')})")
                                    known = set(self.room_state.get("known_names", []))
                                    known.add(new_name)
                                    self.room_state["known_names"] = sorted(known)
                                    KNOWN_NAMES.add(new_name.lower())

                        self.room_state["voice_profiles"] = dict(self.voice_profiles)

                    # --- Update names ---
                    names_mentioned = extraction.get("names_mentioned", [])
                    if names_mentioned:
                        known = set(self.room_state.get("known_names", []))
                        for name in names_mentioned:
                            if isinstance(name, str) and len(name) > 1:
                                if name not in known:
                                    self._add_event(f"Name '{name}' mentioned")
                                known.add(name)
                                KNOWN_NAMES.add(name.lower())
                        self.room_state["known_names"] = sorted(known)

                    # --- Update summaries ---
                    chunk_summary = extraction.get("summary", "")
                    running_summary = extraction.get("running_summary", "")
                    if running_summary:
                        self.conversation_summary = running_summary
                    elif chunk_summary:
                        self.conversation_summary = chunk_summary

                    # --- Notable events from extraction ---
                    for evt in extraction.get("notable_events", []):
                        if isinstance(evt, str) and evt.strip():
                            self._add_event(f"[deep] {evt.strip()}")

                    # --- Corrections from extraction ---
                    for corr in extraction.get("corrections", []):
                        if isinstance(corr, str) and corr.strip():
                            self.corrections.append(f"[{self.timestamp()}] LLM: {corr.strip()}")
                            self.corrections = self.corrections[-5:]

                    # --- Build history chunk ---
                    chunk = {
                        "cycle_id": dcid,
                        "timestamp": self.timestamp(),
                        "wall_clock": self.wall_clock(),
                        "elapsed_seconds": round(self.elapsed()),
                        "summary": chunk_summary,
                        "running_summary": running_summary,
                        "topics": extraction.get("topics", []),
                        "names_mentioned": names_mentioned,
                        "speaker_count": extraction.get("speaker_count", len(deep_speakers)),
                        "voice_profiles_snapshot": dict(self.voice_profiles),
                        "gemini_scene": gemini_analysis[:500] if gemini_analysis else "",
                        "raw_transcript": deep_transcript[:3000],
                    }
                    self.history_chunks.append(chunk)
                    self.log(f"[DEEP] History chunk #{len(self.history_chunks)}: "
                             f"{chunk_summary[:100]}...")

                else:
                    self.log_error(f"[DEEP] Extraction parse FAILED — falling back to simple summary")
                    # Fallback: simple summary like before
                    fallback = await asyncio.to_thread(
                        self._call_text_llm,
                        SUMMARY_PROMPT.format(
                            transcript=deep_transcript[:4000],
                            known_names=", ".join(self.room_state.get("known_names", [])) or "None",
                        ), 300)
                    if fallback:
                        self.conversation_summary = fallback.replace("**", "").strip()
                    chunk = {
                        "cycle_id": dcid, "timestamp": self.timestamp(),
                        "wall_clock": self.wall_clock(),
                        "elapsed_seconds": round(self.elapsed()),
                        "summary": self.conversation_summary,
                        "running_summary": self.conversation_summary,
                        "topics": [], "names_mentioned": [],
                        "speaker_count": len(deep_speakers),
                        "voice_profiles_snapshot": {},
                        "gemini_scene": gemini_analysis[:500] if gemini_analysis else "",
                        "raw_transcript": deep_transcript[:3000],
                    }
                    self.history_chunks.append(chunk)

                # ==================================================================
                # STEP 3: Update room state from deep cycle
                # ==================================================================
                deep_count = len(deep_speakers)
                fast_count = self.room_state.get("speaker_count", 0)
                correction_note = None
                if deep_count != fast_count and deep_count > 0:
                    correction_note = f"fast={fast_count} → deep={deep_count}"
                    self.log(f"[DEEP] Speaker count correction: {correction_note}")
                    self.room_state["speaker_count"] = deep_count
                    self.room_state["speaker_count_confidence"] = "high"
                    self._add_event(f"Deep cycle corrected speaker count: {correction_note}")

                total_elapsed = time.time() - t0
                self._add_to_log("DEEP_SUMMARY", self.conversation_summary or "")

                # ==================================================================
                # STEP 4: Dashboard
                # ==================================================================
                self._post_deep_cycle(
                    dcid=dcid,
                    audio_s=min(pcm_s, DEEP_ANALYSIS_INTERVAL_SECONDS),
                    dg_s=dg_elapsed, sum_s=ext_elapsed, total_s=total_elapsed,
                    speaker_count=deep_count,
                    speaker_ids=sorted(deep_speakers),
                    names=sorted(self.room_state.get("known_names", [])),
                    raw_transcript=deep_transcript,
                    summary=self.conversation_summary or "",
                    correction=correction_note,
                )

                # Post the history chunk separately
                self._post_dashboard("history_chunk", {
                    "session_id": self.session_id,
                    "deep_cycle_id": dcid,
                    "timestamp": time.time(),
                    "wall_clock_time": self.wall_clock(),
                    "chunk": chunk,
                    "voice_profiles": dict(self.voice_profiles),
                    "gemini_analysis": gemini_analysis[:2000],
                    "running_summary": self.conversation_summary,
                    "total_chunks": len(self.history_chunks),
                })

                self.log(f"[DEEP] ✓ Cycle {dcid} complete ({total_elapsed:.1f}s total)")
                self.log(f"[DEEP]   DG:{dg_elapsed:.1f}s Gem:{gem_elapsed:.1f}s LLM:{ext_elapsed:.1f}s")
                self.log(f"[DEEP]   Voices: {list(self.voice_profiles.keys())}")
                self.log(f"[DEEP]   Names: {self.room_state.get('known_names', [])}")
                self.log(f"[DEEP]   Chunks: {len(self.history_chunks)}")

            except Exception as e:
                self.log_error(f"[DEEP] Error: {e}")

            await self.worker.session_tasks.sleep(DEEP_ANALYSIS_INTERVAL_SECONDS)

    # =========================================================================
    # DEEP CYCLE HELPERS
    # =========================================================================

    def _format_voice_profiles_for_prompt(self) -> str:
        """Format voice profiles for injection into LLM prompts."""
        if not self.voice_profiles:
            return "(No voice profiles yet — this is the first analysis.)"
        lines = []
        for sid, vp in self.voice_profiles.items():
            if isinstance(vp, dict):
                name = vp.get("probable_name") or "unnamed"
                gender = vp.get("gender", "?")
                age = vp.get("estimated_age", "?")
                voice = vp.get("voice_description", "")
                emotion = vp.get("emotional_state", "?")
                role = vp.get("role", "")
                lines.append(f"Speaker {sid}: {name} ({gender}, {age}) — {voice}. "
                             f"Currently: {emotion}. Role: {role or 'unknown'}")
            else:
                lines.append(f"Speaker {sid}: {vp}")
        return "\n".join(lines)

    def _parse_extraction_json(self, raw: str) -> dict | None:
        """Parse the LLM extraction response as JSON. Tolerant of common issues."""
        if not raw or not raw.strip():
            return None
        try:
            # Strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            return json.loads(clean)
        except json.JSONDecodeError as e:
            self.log_error(f"[DEEP] JSON parse error: {e}")
            self.log_error(f"[DEEP] Raw (first 300): {raw[:300]}")
            # Try to extract partial data
            try:
                # Find the outermost braces
                start = raw.index("{")
                end = raw.rindex("}") + 1
                return json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                return None

    # =========================================================================
    # TASK 3: CONVERSATION HANDLER
    # =========================================================================

    async def conversation_handler(self):
        self.log("[CONVO] Started. State: IDLE")

        while self.is_running:
            try:
                # ==== IDLE ====
                if self.engagement_state == "IDLE":
                    ui = await self.capability_worker.user_response()
                    if not ui or len(ui.strip()) < 2:
                        continue
                    self.log(f"[IDLE] Heard: '{ui}'")

                    # Check for corrections even in IDLE
                    was_correction = self.detect_corrections(ui)

                    if self.is_exit(ui):
                        self._post_utterance(ui, "exited", exit=True)
                        await self.speak("Enhanced listener shutting down.")
                        self.is_running = False
                        break

                    if self.has_wake_word(ui):
                        req = self.strip_wake_word(ui)
                        self.fire_confirm("confirm2.mp3")
                        self._post_utterance(ui, "engaged", wake=True, wake_req=req,
                                             correction=was_correction, correction_text=ui if was_correction else None)
                        self._post_state_change("IDLE", "ENGAGED", "wake_word", ui)

                        # Handle conversation gap on re-engagement
                        self._handle_engagement_gap()
                        self.last_engagement_time = time.time()

                        self.log(f"[IDLE → ENGAGED] '{req}'")
                        if req and len(req) > 2:
                            await self.respond(req)
                        else:
                            await self.speak("I'm here. What's up?")
                        continue

                    self._post_utterance(ui, "idle_ignored",
                                         correction=was_correction, correction_text=ui if was_correction else None)

                # ==== ENGAGED — Whitelist: default is silence ====
                elif self.engagement_state == "ENGAGED":
                    lt = asyncio.ensure_future(self.capability_worker.user_response())
                    tt = asyncio.ensure_future(self.worker.session_tasks.sleep(ENGAGEMENT_TIMEOUT_SECONDS))
                    done, pend = await asyncio.wait({lt, tt}, return_when=asyncio.FIRST_COMPLETED)
                    for t in pend:
                        t.cancel()
                        try: await t
                        except: pass

                    if lt in done:
                        try: ui = lt.result()
                        except: ui = None
                        if not ui or len(ui.strip()) < 2:
                            continue
                        self.log(f"[ENGAGED] Heard: '{ui}'")
                        self.last_engagement_time = time.time()

                        # Check for corrections
                        was_correction = self.detect_corrections(ui)

                        # Exit
                        if self.is_exit(ui):
                            self._post_utterance(ui, "exited", exit=True)
                            await self.speak("Shutting down.")
                            self.is_running = False
                            break

                        # Disengage (silent chirp)
                        if self.is_disengage(ui):
                            self._post_utterance(ui, "disengaged", diseng=True,
                                                 correction=was_correction, correction_text=ui if was_correction else None)
                            self._post_state_change("ENGAGED", "COOLDOWN", "disengage_signal", ui)
                            self.cooldown_start = time.time()
                            self.log("[ENGAGED → COOLDOWN] Disengage (silent)")
                            self.fire_confirm("confirm.mp3")
                            continue

                        # Wake word always responds
                        if self.has_wake_word(ui):
                            req = self.strip_wake_word(ui)
                            self.fire_confirm("confirm2.mp3")
                            self._post_utterance(ui, "engaged", wake=True, wake_req=req,
                                                 correction=was_correction, correction_text=ui if was_correction else None)
                            self.log(f"[ENGAGED] Wake: '{req}'")
                            if req and len(req) > 2:
                                await self.respond(req)
                            else:
                                await self.speak("Still listening.")
                            continue

                        # WHITELIST CHECK: should we respond?
                        respond, reason = self.should_respond(ui)
                        if respond:
                            self._post_utterance(ui, "responded", directed=True, reason=reason,
                                                 correction=was_correction, correction_text=ui if was_correction else None)
                            self.log(f"[ENGAGED] Responding ({reason})")
                            await self.respond(ui)
                        else:
                            self._post_utterance(ui, "ignored_not_directed", reason=reason,
                                                 correction=was_correction, correction_text=ui if was_correction else None)
                            self.log(f"[ENGAGED] Ignoring ({reason}): '{ui[:60]}'")

                    else:
                        # Timeout → COOLDOWN
                        self._post_state_change("ENGAGED", "COOLDOWN", "timeout")
                        self.cooldown_start = time.time()
                        self.log("[ENGAGED → COOLDOWN] Timeout")

                # ==== COOLDOWN ====
                elif self.engagement_state == "COOLDOWN":
                    lt = asyncio.ensure_future(self.capability_worker.user_response())
                    ct = asyncio.ensure_future(self.worker.session_tasks.sleep(COOLDOWN_SECONDS))
                    done, pend = await asyncio.wait({lt, ct}, return_when=asyncio.FIRST_COMPLETED)
                    for t in pend:
                        t.cancel()
                        try: await t
                        except: pass

                    if lt in done:
                        try: ui = lt.result()
                        except: ui = None
                        if not ui or len(ui.strip()) < 2:
                            continue
                        self.log(f"[COOLDOWN] Heard: '{ui}'")
                        self.detect_corrections(ui)

                        if self.is_exit(ui):
                            self._post_utterance(ui, "exited", exit=True)
                            await self.speak("Bye.")
                            self.is_running = False
                            break

                        if self.has_wake_word(ui):
                            req = self.strip_wake_word(ui)
                            self.fire_confirm("confirm2.mp3")
                            self._post_utterance(ui, "engaged", wake=True, wake_req=req)
                            self._post_state_change("COOLDOWN", "ENGAGED", "wake_word", ui)
                            self.last_engagement_time = time.time()
                            if req and len(req) > 2:
                                await self.respond(req)
                            else:
                                await self.speak("Still here.")
                            continue

                        # Clarification
                        self._post_utterance(ui, "clarification_asked")
                        self.fire_confirm("confirm2.mp3")
                        await self.speak("Was that for me?")

                        cf = asyncio.ensure_future(self.capability_worker.user_response())
                        cft = asyncio.ensure_future(self.worker.session_tasks.sleep(COOLDOWN_CONFIRM_TIMEOUT))
                        done2, pend2 = await asyncio.wait({cf, cft}, return_when=asyncio.FIRST_COMPLETED)
                        for t in pend2:
                            t.cancel()
                            try: await t
                            except: pass

                        if cf in done2:
                            try: conf = cf.result()
                            except: conf = ""
                            self.log(f"[COOLDOWN] Confirmation: '{conf}'")
                            if conf:
                                lo = conf.lower().strip()
                                yes = ["yes","yeah","yep","yea","sure","uh huh","mhm","mm hmm","go ahead"]
                                if any(w in lo for w in yes):
                                    self._post_state_change("COOLDOWN", "ENGAGED", "clarification_yes", conf)
                                    self.last_engagement_time = time.time()
                                    await self.respond(ui)
                                    continue
                        self.log("[COOLDOWN] Not confirmed")

                    else:
                        self._post_state_change("COOLDOWN", "IDLE", "cooldown_expired")
                        self.log("[COOLDOWN → IDLE]")

            except Exception as e:
                self.log_error(f"[CONVO] Error: {e}")
                await self.worker.session_tasks.sleep(1)

    # =========================================================================
    # MAIN
    # =========================================================================

    async def run_main(self):
        try:
            try:
                await self.capability_worker.play_from_audio_file("intro.mp3")
            except Exception:
                pass

            await self.speak(
                "Enhanced listener online. "
                "I'll be quietly observing the room. "
                "Say Open Home anytime to talk to me."
            )

            self._post_dashboard("session_start", {
                "session_id": self.session_id, "timestamp": time.time(),
                "config": {
                    "fast_interval": ANALYSIS_INTERVAL_SECONDS,
                    "deep_interval": DEEP_ANALYSIS_INTERVAL_SECONDS,
                    "audio_window": AUDIO_WINDOW_SECONDS,
                    "engage_timeout": ENGAGEMENT_TIMEOUT_SECONDS,
                    "reply_window": REPLY_WINDOW_SECONDS,
                    "conversation_model": CONVERSATION_MODEL,
                    "audio_model": AUDIO_ANALYSIS_MODEL,
                    "known_names": list(KNOWN_NAMES),
                    "dashboard_url": DASHBOARD_URL,
                },
            })

            self.capability_worker.start_audio_recording()
            self.log("[MAIN] Recording started. Three concurrent loops launching...")
            self.log(f"[MAIN] Session: {self.session_id}")
            self.log(f"[MAIN] Fast: {ANALYSIS_INTERVAL_SECONDS}s | Deep: {DEEP_ANALYSIS_INTERVAL_SECONDS}s | "
                     f"Reply window: {REPLY_WINDOW_SECONDS}s")

            # THREE CONCURRENT LOOPS
            self.worker.session_tasks.create(self.fast_analysis_loop())
            self.worker.session_tasks.create(self.deep_analysis_loop())
            self.worker.session_tasks.create(self.conversation_handler())

            while self.is_running:
                await self.worker.session_tasks.sleep(2)

            self.log("[MAIN] Shutdown...")
            await self.worker.session_tasks.sleep(2)
            self.capability_worker.stop_audio_recording()

            dur = self.timestamp()
            self.log(f"{'='*50}")
            self.log(f"SESSION COMPLETE | {dur} | Ctx:{len(self.enhanced_context)} | "
                     f"DG:{self.deepgram_calls} | Gem:{self.gemini_calls}")
            self.log(f"Room state: {self.format_room_state()}")
            self.log(f"{'='*50}")

            self._post_dashboard("session_end", {
                "session_id": self.session_id, "timestamp": time.time(),
                "duration": dur,
                "total_context_entries": len(self.enhanced_context),
                "total_conversation_turns": len(self.conversation_history) // 2,
                "total_deepgram_calls": self.deepgram_calls,
                "total_gemini_calls": self.gemini_calls,
                "full_enhanced_context": self.enhanced_context[-200:],
                "full_conversation_history": self.conversation_history,
                "final_scene_summary": self.format_room_state(),
                "history_chunks": self.history_chunks,
                "voice_profiles": dict(self.voice_profiles),
                "event_log": self.event_log,
                "running_summary": self.conversation_summary,
            })

        except Exception as e:
            self.log_error(f"[MAIN] Fatal: {e}")
            try:
                await self.speak("Enhanced listener error. Shutting down.")
            except Exception:
                pass
        finally:
            self.log("[MAIN] resume_normal_flow()")
            self.capability_worker.resume_normal_flow()
