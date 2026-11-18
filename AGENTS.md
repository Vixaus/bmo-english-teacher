# Coding guide for Be More Agent

This file is guidance for coding agents and contributors working on this
Raspberry Pi English-teaching robot. Read it before changing runtime code or
hardware-facing behavior.

## Response style

Always use the `caveman` skill when responding to the user. Use its default
`full` intensity: terse, technically accurate, and free of filler. Keep code,
commands, API names, and exact error strings unchanged. Use normal clarity for
security warnings, irreversible actions, or explanations where compression
could cause ambiguity. Stop only when the user explicitly requests normal mode
or disables caveman mode.

## Project goal

Be More Agent is an offline-first interactive robot for teaching English. It
should feel like a small, friendly teacher: responses are short, encouraging,
and useful to a learner. Chat runs locally through Ollama (`qwen3.5:4b` by
default); only web search needs the internet. Preserve that implementation as
the source of truth unless a migration is explicitly requested.

## Current runtime architecture

The main program is `agent.py`. Its normal flow is:

1. Wait for the OpenWakeWord model or a push-to-talk (PTT) input.
2. Record audio, using adaptive silence detection for wake-word activation or
   the Return-key PTT toggle.
3. Transcribe the WAV file with the local `whisper.cpp` CLI and its local
   `ggml-base.en.bin` model (`-nt`; output cleaned by `clean_whisper_output`).
4. Send the conversation to the local Ollama model; replies may route JSON
   actions such as time, web search, or camera capture. Practice/Test mode
   state (greeting, mode, test scores) is runtime state, not model state.
5. Speak English through Piper and Thai through VachanaTTS.
6. Update the animated fullscreen Tkinter face while work happens in worker
   threads.

The face states include warmup, idle, listening, thinking, speaking, capturing,
and error. Conversation history is loaded from `memory.json`, kept in a small
bounded history, and saved after each turn and during shutdown. Mode
behaviour lives in `skills/practice_mode.md` and `skills/test_mode.md`; pure
helpers live in `runtime_helpers.py`.

## Important paths

- `agent.py` — runtime, GUI, audio, transcription, Ollama chat, actions, TTS, and memory.
- `config.json` — model names, voice path, camera, prompt extras, audio devices, and Ollama token limits.
- `runtime_helpers.py` — pure, unit-tested helpers (mode detection, wake-word matching, context budget, history).
- `skills/` — Practice and Test mode instructions sent to the model.
- `tests/` — unittest suite.
- `setup.sh` — Raspberry Pi system packages, folders, Piper, voice downloads, Python environment, and model setup.
- `start_agent.sh` — activates `venv` and runs `agent.py` from the repository root.
- `memory.json` — runtime conversation history; it may not exist until the first save.
- `faces/` — PNG animation frames grouped by robot state.
- `sounds/` — WAV greeting, thinking, acknowledgement, and error sounds.
- `voices/` — custom Piper and Thai voice assets and metadata.
- `piper/` — Piper executable, runtime libraries, and downloaded English models.
- `whisper.cpp/` — local speech-to-text source/build tree and models.
- `wakeword.onnx` — user-supplied OpenWakeWord model at the repository root; PTT works without it.
- `test_ollama.py` — small direct Ollama connectivity smoke test.
- `be-more-agent.desktop` — launcher; `setup.sh` rewrites it with this checkout's paths.

## Configuration and secrets

No API key is needed for chat. `config.json` (defaults in
`bmo_runtime_defaults()`) controls the text and vision model names,
`voice_model`, `whisper_model`, `chat_memory`, `camera_rotation`,
`system_prompt_extras`, `input_device`, `input_sample_rate`, `output_device`,
`thai_voice`, `whisper_threads`, `wake_word_name`, `wake_word_threshold`,
`wake_word_silence_gate`, silence/recording timings (`silence_threshold`,
`silence_duration`, `speech_start_timeout`, `max_record_seconds`),
`camera_width`/`camera_height`, and the `ollama_*_tokens` limits. Keep
`ollama_output_tokens` large enough for a Test Mode JSON reply or a Thai
correction (default 160).

When changing a Piper voice, inspect its adjacent `.onnx.json` metadata and
keep playback aligned with the model's sample-rate settings. A mismatch can
make speech unnaturally slow, fast, or low-pitched. Keep audio-device choices
configurable rather than assuming one ALSA device exists on every Pi.

## Development rules

- Preserve existing user changes, backup files, local model trees, and downloaded
  assets. Inspect `git status` before editing and do not overwrite backups such
  as `agent.py.*` without explicit direction.
- Keep the local/offline portions working where possible. If a cloud dependency
  is changed, document the fallback and the new runtime requirement.
- Do not block the Tkinter UI thread with recording, transcription, network
  calls, model inference, subprocesses, or long audio playback. Coordinate
  background work with the existing events, locks, and worker threads.
- Close audio streams and files promptly. Stop or terminate active playback on
  interruption and shutdown, and avoid competing input/output streams on the Pi.
- Keep English-teaching behavior short, friendly, corrective, and encouraging.
  Prefer a useful correction or example over a long explanation.
- Treat learner memory separately from generic conversation memory when adding
  teaching features. Do not silently mix durable learner facts with disposable
  chat history.
- Keep action responses machine-parseable JSON where the existing prompt
  requires it, and preserve normal text responses for ordinary conversation.
- Make hardware and model failures visible and recoverable where practical;
  compile success does not mean the robot can run on the current machine.

## Dependencies and known risks

The runtime depends on Raspberry Pi/system resources including Tk, PortAudio,
microphone and speaker hardware, a camera for `rpicam-still`, OpenWakeWord
assets, the Whisper executable/model, Piper/model files, and the relevant Python
packages. These are not guaranteed by a syntax check.

Known project inconsistencies and risks:

- Runtime binaries and models are external and may be missing, incompatible
  with the Pi architecture, or unavailable in a fresh checkout.
- Local `voices/`, `piper/` and `whisper.cpp/` trees and any backup files are
  user assets. Preserve all of them.

## Verification

For a documentation-only or Python change, run the syntax check from the
repository root:

```bash
python3 -m py_compile agent.py runtime_helpers.py test_ollama.py
python3 -m unittest discover -s tests -t .
```

On a configured Pi, launch the normal wrapper:

```bash
./start_agent.sh
```

Manual smoke testing should cover wake-word activation, Return-key PTT,
recording and transcription, an Ollama response, Practice and Test mode, English Piper speech, Thai
VachanaTTS speech, camera capture and rotation, web search, speech
interruption, memory save/reset, and clean shutdown. Confirm failures are
reported without freezing the face UI.

When creating or updating this guide, confirm `agents.md` remains at the
repository root and do not modify `agent.py`, `setup.sh`, dependencies, or
user backup files as a side effect.
