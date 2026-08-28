"""Pure helpers for runtime actions, audio, history, and BMO configuration."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

THAI_RANGE = re.compile(r"[\u0E00-\u0E7F]+|[^\u0E00-\u0E7F]+")

ACTION_ALIASES = {
    "google": "search_web",
    "browser": "search_web",
    "news": "search_web",
    "search_news": "search_web",
    "look": "capture_image",
    "see": "capture_image",
    "check_time": "get_time",
}

VALID_ACTIONS = {"get_time", "search_web", "capture_image"}

MODE_NAMES = {"practice": "practice", "test": "test"}
TEST_SCORE_FIELDS = ("grammar", "vocabulary", "comprehension")
DEFAULT_OLLAMA_CONTEXT_TOKENS = 2048
DEFAULT_OLLAMA_OUTPUT_TOKENS = 40
# Total estimated input prompt budget.  Keep below the normal 75%-safe
# budget for a 2,048-token context so required instructions still have room
# for several recent user/assistant turns.
DEFAULT_OLLAMA_PROMPT_TOKENS = 1400


def wake_word_input_frame_size(
    input_sample_rate: Any,
    target_sample_rate: int = 16000,
    target_frame_samples: int = 1280,
):
    """Return input samples needed for one OpenWakeWord frame.

    OpenWakeWord consumes 1,280 samples at 16 kHz (80 ms).  Keep the frame
    duration constant when a microphone runs at another sample rate; never
    substitute a smaller arbitrary block and stretch it during resampling.
    """
    try:
        input_rate = int(input_sample_rate)
        target_rate = int(target_sample_rate)
        target_samples = int(target_frame_samples)
    except (TypeError, ValueError):
        return 1280
    if input_rate <= 0 or target_rate <= 0 or target_samples <= 0:
        return 1280
    return max(1, round(target_samples * input_rate / target_rate))


def bmo_runtime_defaults():
    """Return canonical BMO runtime configuration defaults."""
    return {
        "text_model": "qwen3.5:4b",
        "vision_model": "qwen3.5:4b",
        "voice_model": "piper/en_GB-semaine-medium.onnx",
        "chat_memory": True,
        "camera_rotation": 180,
        "system_prompt_extras": "",
        "input_device": None,
        "input_sample_rate": 44100,
        "silence_threshold": 0.006,
        "recording_settle_delay": 0.15,
        "silence_duration": 0.8,
        "wake_word_name": "Hey BMO",
        "ollama_context_tokens": DEFAULT_OLLAMA_CONTEXT_TOKENS,
        "ollama_output_tokens": DEFAULT_OLLAMA_OUTPUT_TOKENS,
        "ollama_prompt_tokens": DEFAULT_OLLAMA_PROMPT_TOKENS,
    }


def detect_mode_selection(text: Any):
    """Return practice/test selection from standalone mode words.

    Accepts ``practice`` and ``test`` with or without the word ``mode``.
    Returns ``None`` when neither word appears or both appear.
    """
    if not isinstance(text, str):
        return None

    found = {
        mode for word, mode in (("practice", "practice"), ("test", "test"))
        if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)
    }
    return next(iter(found)) if len(found) == 1 else None


def contains_wake_word(text: Any, wake_word: Any):
    """Return whether text contains the configured wake word as a phrase."""
    if not isinstance(text, str) or not isinstance(wake_word, str):
        return False
    wake_word = wake_word.strip()
    if not wake_word:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(wake_word)}(?!\w)", text, flags=re.IGNORECASE))


def load_skill(mode: Any, skills_dir: Any = "skills"):
    """Load one bundled mode skill; return empty string for invalid/missing skill."""
    if not isinstance(mode, str):
        return ""
    mode = mode.strip().lower()
    if mode not in MODE_NAMES:
        return ""
    path = Path(skills_dir) / f"{MODE_NAMES[mode]}_mode.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def ollama_prompt_budget(
    context_tokens: Any,
    output_tokens: Any,
    threshold: float = 0.75,
    prompt_tokens: Any = None,
):
    """Return safe total input budget after output and safety reservations."""
    try:
        context = max(1, int(context_tokens))
        output = max(0, int(output_tokens))
        threshold = min(1.0, max(0.01, float(threshold)))
    except (TypeError, ValueError):
        context = DEFAULT_OLLAMA_CONTEXT_TOKENS
        output = DEFAULT_OLLAMA_OUTPUT_TOKENS
        threshold = 0.75
    available = max(1, context - output)
    budget = max(1, math.floor(available * threshold))
    if prompt_tokens is not None:
        try:
            budget = min(budget, max(1, int(prompt_tokens)))
        except (TypeError, ValueError):
            pass
    return budget


def estimate_prompt_tokens(value: Any):
    """Conservative mixed-language token estimate for text or Ollama messages."""
    if isinstance(value, dict):
        content = value.get("content", "")
        return estimate_prompt_tokens(content) + 4
    if isinstance(value, (list, tuple)):
        return sum(estimate_prompt_tokens(item) for item in value)
    if not isinstance(value, str) or not value:
        return 0

    # English averages near four characters/token; Thai and other scripts can
    # tokenize much more densely. Lower ratios deliberately overestimate.
    ascii_chars = sum(char.isascii() for char in value)
    other_chars = len(value) - ascii_chars
    return max(1, math.ceil(ascii_chars / 3.0 + other_chars / 1.5))


def compact_context(
    core_rules: Any,
    active_skill: Any,
    mode_state: Any,
    current_user_message: Any,
    conversation_turns: Any = (),
    context_tokens: int = DEFAULT_OLLAMA_CONTEXT_TOKENS,
    output_tokens: int = DEFAULT_OLLAMA_OUTPUT_TOKENS,
    prompt_tokens: Any = None,
):
    """Build bounded prompt, retaining required rules/state/user and newest turns.

    ``prompt_tokens`` caps the estimated total input prompt, including
    required instructions, runtime state, selected history, and current user
    input. Required instructions and the current user message always remain,
    even when their estimate exceeds the cap.
    """
    required = []
    for content in (core_rules, active_skill, mode_state):
        if isinstance(content, dict):
            if content.get("content"):
                required.append(dict(content))
        elif isinstance(content, str) and content.strip():
            required.append({"role": "system", "content": content})

    if isinstance(current_user_message, dict):
        current_user = dict(current_user_message)
    else:
        current_user = {"role": "user", "content": str(current_user_message or "")}
    current_user.setdefault("role", "user")

    turns = []
    if isinstance(conversation_turns, (list, tuple)):
        for turn in conversation_turns:
            if isinstance(turn, dict) and turn.get("content"):
                turns.append(dict(turn))

    budget = ollama_prompt_budget(
        context_tokens, output_tokens, prompt_tokens=prompt_tokens
    )
    result = required[:]
    # Keep newest complete turns. A complete turn is user+assistant; callers
    # normally provide only complete pairs, but incomplete trailing messages
    # are ignored to avoid leaking half a prior request.
    complete_turns = turns
    if len(turns) % 2:
        complete_turns = turns[:-1]
    selected = []
    for index in range(len(complete_turns) - 2, -1, -2):
        candidate = complete_turns[index:index + 2]
        if estimate_prompt_tokens(result + selected + candidate + [current_user]) > budget:
            break
        selected[0:0] = candidate

    result.extend(selected)
    result.append(current_user)

    # Required messages must survive even when their estimate alone is large.
    # Only optional history is dropped; current user is never dropped.
    return result


def emergency_context(
    core_rules: Any,
    active_skill: Any,
    mode_state: Any,
    current_user_message: Any,
    context_tokens: int = DEFAULT_OLLAMA_CONTEXT_TOKENS,
    output_tokens: int = DEFAULT_OLLAMA_OUTPUT_TOKENS,
    prompt_tokens: Any = None,
):
    """Return minimal overflow-retry context with no conversation history."""
    return compact_context(
        core_rules,
        active_skill,
        mode_state,
        current_user_message,
        conversation_turns=(),
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
    )


def clamp_test_score(value: Any):
    """Convert numeric score to integer in inclusive range 0..5."""
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(5, int(float(value))))
    except (TypeError, ValueError):
        return 0


def validate_test_scores(data: Any):
    """Validate score object and clamp each required category to 0..5."""
    if not isinstance(data, dict):
        return None
    if any(field not in data for field in TEST_SCORE_FIELDS):
        return None
    for field in TEST_SCORE_FIELDS:
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
    return {field: clamp_test_score(data[field]) for field in TEST_SCORE_FIELDS}


def validate_test_result(data: Any):
    """Validate compact Test Mode result and remove untrusted extra fields."""
    if not isinstance(data, dict) or not isinstance(data.get("speech"), str):
        return None
    speech = data["speech"].strip()
    scores = validate_test_scores(data)
    if not speech or scores is None or not isinstance(data.get("done"), bool):
        return None
    return {"speech": speech, **scores, "done": data["done"]}


def parse_test_response(text: Any):
    """Parse and validate one model Test Mode JSON response."""
    if not isinstance(text, str):
        return None
    try:
        return validate_test_result(json.loads(text))
    except (TypeError, ValueError):
        return None


def overall_test_score(scores: Any):
    """Return average of category averages, rounded to 2 places."""
    category_averages = []
    if isinstance(scores, dict):
        for field in TEST_SCORE_FIELDS:
            category = scores.get(field, [])
            if isinstance(category, (list, tuple)):
                if category:
                    category_averages.append(
                        sum(clamp_test_score(value) for value in category) / len(category)
                    )
            elif category is not None:
                category_averages.append(clamp_test_score(category))
    return round(sum(category_averages) / len(category_averages), 2) if category_averages else 0.0


def cefr_level(score: Any):
    """Map overall 0..5 average to requested A1/A2 level."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return "A2" if score >= 3.0 else "A1"


# Descriptive aliases keep helper use readable at call sites.
calculate_overall_score = overall_test_score
calculate_cefr_level = cefr_level


def missing_wake_word_warning(model_path: str, wake_word_name: str):
    """Return actionable warning when wake-word activation is unavailable."""
    return (
        f"[WARNING] Wake-word model missing: {model_path}. "
        f"Add user-supplied wakeword.onnx trained for '{wake_word_name}'; "
        "push-to-talk remains available."
    )


def interpolate_wake_word_name(template: str, wake_word_name: str):
    """Insert wake-word identity without interpreting JSON braces in a prompt."""
    return template.replace("{wake_word_name}", wake_word_name)


def extract_action(text: Any):
    """Return normalized action dict or None."""
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return normalize_action(parsed)


def normalize_action(action_data: Any):
    """Return {'action': str, 'value': str} or None."""
    if not isinstance(action_data, dict):
        return None

    raw_action = action_data.get("action")
    if not isinstance(raw_action, str):
        return None

    action = raw_action.strip().lower()
    if not action:
        return None

    value = action_data.get("value")

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    action = ACTION_ALIASES.get(action, action)
    if action not in VALID_ACTIONS:
        return None

    return {"action": action, "value": value}


def split_tts_segments(text: Any):
    """Return list of ('english'|'thai', segment_text) tuples."""
    if not isinstance(text, str) or not text:
        return []

    segments = []
    for chunk in THAI_RANGE.findall(text):
        piece = chunk.strip()
        if not piece:
            continue
        language = "thai" if _is_thai_chunk(piece) else "english"
        segments.append((language, piece))

    return segments


def load_voice_sample_rate(model_path: Any):
    """Return positive integer sample rate from adjacent Piper JSON."""
    if not isinstance(model_path, str) or not model_path.strip():
        return None

    for candidate in _metadata_candidates(model_path):
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        sample_rate = _find_sample_rate(data)
        if sample_rate is not None:
            return sample_rate

    return None


def validate_voice_model(model_path: Any):
    """Return (valid, message) for Piper model and adjacent metadata."""
    if not isinstance(model_path, str) or not model_path.strip():
        return False, "Piper voice path is empty."

    path = Path(model_path)
    try:
        size = path.stat().st_size
    except OSError:
        return False, f"Piper voice file missing: {model_path}"

    if size < 1024:
        return False, f"Piper voice file is invalid or too small: {model_path}"

    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
        if b"Not Found" in prefix or b"<html" in prefix.lower():
            return False, f"Piper voice file is not an ONNX model: {model_path}"
    except OSError:
        return False, f"Piper voice file cannot be read: {model_path}"

    metadata = next(iter(_metadata_candidates(model_path)), None)
    if metadata is None or not metadata.is_file():
        return False, f"Piper voice metadata missing: {model_path}.json"
    sample_rate = load_voice_sample_rate(model_path)
    if sample_rate is None:
        return False, f"Piper voice metadata invalid: {metadata}"
    return True, ""


def load_history_file(path: Any, system_prompt: str, enabled: bool = True, limit: int = 10):
    """Load and normalize history, returning current prompt on disabled/error."""
    if not enabled:
        return normalize_history([], system_prompt, limit)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_history = json.load(handle)
    except (OSError, ValueError, TypeError):
        raw_history = []
    return normalize_history(raw_history, system_prompt, limit)


def save_history_file(path: Any, history: Any, system_prompt: str = None, enabled: bool = True, limit: int = 10):
    """Atomically save normalized history. Return False when disabled."""
    if not enabled:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if system_prompt is None:
        system_prompt = history[0].get("content", "") if isinstance(history, list) and history and isinstance(history[0], dict) else ""
    normalized = normalize_history(history, system_prompt, limit=limit)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(destination.parent),
            prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(normalized, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        return True
    except OSError:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return False


def normalize_history(raw_history: Any, system_prompt: str, limit: int = 10):
    """Return valid history beginning with current system prompt."""
    normalized = [{"role": "system", "content": system_prompt}]

    if not isinstance(raw_history, list):
        return normalized

    if not isinstance(limit, int) or limit <= 0:
        return normalized

    valid_messages = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue

        role = role.strip().lower()
        if role not in {"user", "assistant"}:
            continue

        if not content.strip():
            continue

        valid_messages.append({"role": role, "content": content})

    return normalized + valid_messages[-limit:]


def _is_thai_chunk(text: str) -> bool:
    return any("\u0E00" <= char <= "\u0E7F" for char in text)


def _metadata_candidates(model_path: str) -> Iterable[Path]:
    path = Path(model_path)
    candidates = []

    if str(path).endswith(".json"):
        candidates.append(path)
    else:
        candidates.append(Path(f"{path}.json"))
        candidates.append(path.with_suffix(".json"))

    seen = set()
    for candidate in candidates:
        candidate_key = os.fspath(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        yield candidate


def _find_sample_rate(data: Any):
    if isinstance(data, dict):
        audio = data.get("audio")
        if isinstance(audio, dict):
            sample_rate = _coerce_positive_int(audio.get("sample_rate"))
            if sample_rate is not None:
                return sample_rate

        sample_rate = _coerce_positive_int(data.get("sample_rate"))
        if sample_rate is not None:
            return sample_rate

        for value in data.values():
            sample_rate = _find_sample_rate(value)
            if sample_rate is not None:
                return sample_rate

    elif isinstance(data, list):
        for value in data:
            sample_rate = _find_sample_rate(value)
            if sample_rate is not None:
                return sample_rate

    return None


def _coerce_positive_int(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None
