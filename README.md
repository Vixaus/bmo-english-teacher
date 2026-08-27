# BMO English Teacher

Offline-first English-teaching robot for Raspberry Pi 4 (4 GB+) or Pi 5.

Runtime uses local Ollama for text and vision, local Whisper.cpp for speech-to-text, Piper for English speech, and VachanaTTS for Thai explanations. Web search is optional network behavior, so system is offline-first, not fully offline.

## Features

- `Hello BMO` spoken greeting and intended wake phrase through a user-supplied OpenWakeWord model.
- Return-key push-to-talk (PTT) fallback when no wake-word model exists.
- Local Ollama models: `qwen2.5:3b` and `moondream`.
- BMO English voice through Piper; Thai correction explanations through VachanaTTS.
- Camera capture through `rpicam-still`, with configurable rotation.
- Animated Tkinter face, interruption with Space, and bounded conversation history.
- Practice Mode and Test Mode remain placeholders: `Skills coming soon`.

## Hardware and software

Use microphone, speaker, display, Raspberry Pi camera, Raspberry Pi OS, Ollama, and required system packages. `setup.sh` installs Python/system dependencies and builds Whisper.cpp when needed.

## Installation

Install Ollama, then pull exact runtime models:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama pull moondream
```

Run setup:

```bash
chmod +x setup.sh start_agent.sh
./setup.sh
./start_agent.sh
```

Setup preserves existing voice files, Whisper trees, and backups. It does not download a default wake-word model. Setup needs internet for packages/models/assets; runtime chat and speech remain local.

## Wake word

Provide an OpenWakeWord ONNX model trained for the phrase `Hello BMO` at:

```text
wakeword.onnx
```

An existing `Hey Jarvis` model is incompatible and will not detect `Hello BMO`. Without `wakeword.onnx`, press Return to start/stop PTT recording. Startup prints a warning and keeps PTT available.

## Voice assets

Canonical BMO Piper files:

```text
voices/bmo-custom.onnx
voices/bmo-custom.onnx.json
```

Both must be valid. Tiny HTTP error files such as `Not Found` are rejected. JSON must contain a positive Piper sample rate (`audio.sample_rate` or `sample_rate`). Runtime reads sample rate from metadata; it does not assume 22050 Hz. If output hardware rejects that rate, runtime resamples audio to the device rate.

Thai speech requires:

```text
voices/th_m_1.onnx
```

and the `vachanatts` Python package. Setup reports Thai TTS as unavailable when this voice is missing. Mixed Thai explanation plus English correction is spoken in source order.

## Whisper.cpp

Runtime expects:

```text
whisper.cpp/build/bin/whisper-cli
whisper.cpp/models/ggml-base.bin
```

Setup clones/builds Whisper.cpp only when its tree is absent and builds `whisper-cli`/downloads `ggml-base.bin` when missing. Existing trees are not overwritten.

## Configuration

`config.json` fields:

```json
{
  "text_model": "qwen2.5:3b",
  "vision_model": "moondream",
  "voice_model": "voices/bmo-custom.onnx",
  "chat_memory": true,
  "camera_rotation": 180,
  "system_prompt_extras": "",
  "input_device": null,
  "input_sample_rate": 44100,
  "wake_word_name": "Hello BMO"
}
```

`chat_memory` controls canonical local history file `memory.json`. When false, runtime neither reads nor writes it. Reset-memory requests clear active conversation and restore current system prompt. History is saved after completed turns and during shutdown, atomically.

`input_device` accepts an ALSA/device index or a case-insensitive name fragment. `input_sample_rate` is preferred input rate; runtime checks compatible rates. `camera_rotation` is passed to image rotation; current default is 180 degrees.

## Project structure

```text
agent.py                  Runtime, GUI, audio, Ollama, actions, memory
runtime_helpers.py        Hardware-independent parsing/config/history helpers
config.json               Runtime configuration
memory.json               Local bounded conversation history
setup.sh                  Raspberry Pi setup
start_agent.sh            Virtualenv launcher
faces/                    Animation frames by state
sounds/                   Greeting/thinking/ack/error WAV files
voices/                   Piper and Thai voice assets
piper/                    Piper binary and models
whisper.cpp/              Whisper source, build, and model
wakeword.onnx             Optional user-supplied Hello BMO model
```

## Actions and web search

Ollama may emit strict JSON for time, search, or camera actions. Normal teaching text remains normal text. Search needs internet; unavailable search produces a short recoverable message. Camera and model work run in worker threads so Tkinter stays responsive.

## Development checks

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile agent.py test_ollama.py
bash -n setup.sh start_agent.sh
```

Manual Pi smoke test should cover PTT/wake word, Whisper, Ollama, English/Thai/mixed speech, camera rotation, search failure, Space interruption, reset, restart persistence, and clean shutdown.

## License and disclaimer

Software is MIT licensed. BMO and Adventure Time are trademarks/copyright of their respective owners. This fan project is not affiliated with or endorsed by Cartoon Network or Warner Bros. Discovery.
