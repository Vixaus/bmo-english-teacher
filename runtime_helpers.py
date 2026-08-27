"""Pure runtime helpers for action parsing, TTS splitting, voice metadata, and history."""

from __future__ import annotations

import json
import os
import re
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
    if value is None:
        value = action_data.get("query")

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

