# BMO English Teacher

Offline-first English-teaching robot for Raspberry Pi 4 (4 GB+) or Pi 5.

Runtime uses local Ollama for text and vision, local Whisper.cpp for speech-to-text, Piper for English speech, and VachanaTTS for Thai explanations. Web search is optional network behavior, so system is offline-first, not fully offline.

## Features

- `Hey BMO` spoken greeting and intended wake phrase through a user-supplied OpenWakeWord model.
- Return-key push-to-talk (PTT) fallback when no wake-word model exists.
- Local Ollama model: `qwen3.5:4b` for text and vision.
- BMO English voice through Piper; Thai correction explanations through VachanaTTS.
- Camera capture through `rpicam-still`, with configurable rotation.
- Animated Tkinter face, interruption with Space, and bounded conversation history.
- Practice Mode: short A1–A2 CEFR communication practice with major-error correction.
- Test Mode: five questions with grammar, vocabulary, comprehension, overall score, and A1/A2 result.
- Saying `practice` or `test` (with or without `mode`) switches mode anytime and resets only active mode progress.

## Hardware and software

Use microphone, speaker, display, Raspberry Pi camera, Raspberry Pi OS, Ollama, and required system packages. `setup.sh` installs Python/system dependencies and builds Whisper.cpp when needed.

## Installation

Install Ollama, then pull exact runtime models:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:4b
```

Run setup:

```bash
chmod +x setup.sh start_agent.sh
./setup.sh
./start_agent.sh
```

Setup preserves existing voice files, Whisper trees, and backups. It does not download a default wake-word model. Setup needs internet for packages/models/assets; runtime chat and speech remain local.

## Wake word

Provide an OpenWakeWord ONNX model trained for the phrase `Hey BMO` at:

```text
wakeword.onnx
```

An existing `Hey Jarvis` model is incompatible and will not detect `Hey BMO`. A successful wake-word trigger immediately accepts the greeting and speaks the mode choices; no second spoken wake phrase is required. Without `wakeword.onnx`, press Return to start/stop PTT recording; PTT speech must include `Hey BMO` before mode selection. Startup prints a warning and keeps PTT available.

## Voice assets

Bundled default Piper files:

```text
piper/en_GB-semaine-medium.onnx
piper/en_GB-semaine-medium.onnx.json
```

The bundled voice is the default. An optional custom BMO voice can override it by setting `voice_model` to these files:

```text
voices/bmo-custom.onnx
voices/bmo-custom.onnx.json
```

Both custom files must be valid. Tiny HTTP error files such as `Not Found` are rejected. JSON must contain a positive Piper sample rate (`audio.sample_rate` or `sample_rate`). Runtime reads sample rate from metadata; it does not assume 22050 Hz. If output hardware rejects that rate, runtime resamples audio to the device rate. Setup does not require the optional custom voice.

Thai speech requires:

```text
voices/th_m_1.onnx
```

and the `vachanatts` Python package. Setup reports Thai TTS as unavailable when this voice is missing. Mixed Thai explanation plus English correction is spoken in source order.

## Whisper.cpp

Runtime expects:

```text
whisper.cpp/build/bin/whisper-cli
whisper.cpp/models/ggml-small.en.bin
```

Setup keeps Whisper shared libraries relocatable with an `$ORIGIN` runpath, rebuilds `whisper-cli`, downloads the English-only `ggml-small.en.bin` model when missing, then transcribes a temporary silent WAV with language forced to English. Setup fails with Whisper output if its runtime libraries or model cannot run. Runtime also supplies `whisper.cpp/build/bin` through `LD_LIBRARY_PATH` as a fallback for older builds and always forces `-l en`.

## Configuration

`config.json` fields:

```json
{
  "text_model": "qwen3.5:4b",
  "vision_model": "qwen3.5:4b",
  "voice_model": "piper/en_GB-semaine-medium.onnx",
  "chat_memory": true,
  "camera_rotation": 180,
  "system_prompt_extras": "",
  "input_device": null,
  "input_sample_rate": 44100,
  "wake_word_name": "Hey BMO",
  "ollama_context_tokens": 2048,
  "ollama_output_tokens": 40,
  "ollama_prompt_tokens": 1400
}
```

`chat_memory` controls canonical local history file `memory.json`. When false, runtime neither reads nor writes it. Reset-memory requests clear active conversation and restore current system prompt. History is saved after completed turns and during shutdown, atomically.

`input_device` accepts an ALSA/device index or a case-insensitive name fragment. `input_sample_rate` is preferred input rate; runtime checks compatible rates. `camera_rotation` is passed to image rotation; current default is 180 degrees.

## English modes

After `Hey BMO`, choose `Practice` or `Test` (optionally say `Practice Mode` or `Test Mode`). A physical wake trigger opens this menu immediately; PTT requires saying `Hey BMO` in recorded speech. Saying either mode word anytime switches mode. Runtime owns active mode, recent mode context, test question index, and scores; these values are not model memory. Selecting a mode clears its active context and test progress. Active mode requests use only recent mode context; completed turns remain in bounded `memory.json` for existing generic-memory behavior but are not restored into a new active mode.

Practice Mode uses `skills/practice_mode.md`: one short A1–A2 question or response at a time, major grammar corrections only, no scores. Test Mode uses `skills/test_mode.md`: five questions, one at a time. Each answer receives 0–5 grammar, vocabulary, and comprehension scores. Overall score is the average of all category scores; below 3.0 is A1, 3.0 or higher is A2.

Before each Qwen request, runtime estimates mixed-language tokens and limits the total input prompt to `ollama_prompt_tokens`; this includes core safety rules, active skill, runtime state, selected history, and the current user message. Required instructions and the current user message always remain. With `ollama_context_tokens: 2048` and `ollama_output_tokens: 40`, the normal safe budget is 1,506 tokens; the configured 1,400-token cap leaves room for recent Practice/Test turns while reserving output and headroom. `ollama_output_tokens` limits generated tokens; compact defaults reduce CPU latency while preserving short teaching replies. Ollama keeps the model loaded between requests. If Ollama still reports context overflow, runtime makes one minimal retry. A second failure clears active context only, preserves `memory.json`, and speaks `Let us start this question again.` No model summary is generated for compaction.

## Project structure

```text
agent.py                  Runtime, GUI, audio, Ollama, actions, memory
runtime_helpers.py        Hardware-independent parsing/config/history helpers
skills/                   Practice and Test Mode prompt skills
config.json               Runtime configuration
memory.json               Local bounded conversation history
setup.sh                  Raspberry Pi setup
start_agent.sh            Virtualenv launcher
faces/                    Animation frames by state
sounds/                   Greeting/thinking/ack/error WAV files
voices/                   Optional custom Piper and Thai voice assets
piper/                    Piper binary and bundled default voice
whisper.cpp/              Whisper source, build, and model
wakeword.onnx             Optional user-supplied Hey BMO model
```

## Actions and web search

Ollama may emit strict JSON for time, search, or camera actions. Normal teaching text remains normal text. Search needs internet; unavailable search produces a short recoverable message. Camera and model work run in worker threads so Tkinter stays responsive.

## Development checks

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile agent.py runtime_helpers.py test_ollama.py
bash -n setup.sh start_agent.sh
```

Manual Pi smoke test should cover PTT/wake word, Whisper, Ollama, English/Thai/mixed speech, camera rotation, search failure, Space interruption, reset, restart persistence, and clean shutdown.

## License and disclaimer

Software is MIT licensed. BMO and Adventure Time are trademarks/copyright of their respective owners. This fan project is not affiliated with or endorsed by Cartoon Network or Warner Bros. Discovery.
