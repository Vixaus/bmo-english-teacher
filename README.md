# BMO English Teacher 🤖

BMO English Teacher is a small AI English teacher for Raspberry Pi.

It lets you practice English by talking with BMO using your voice. BMO can listen, understand, correct mistakes, and speak back to you.

Most features run locally, so internet is not needed for normal conversations. Internet is mainly used for setup, downloading models, and optional web search.

## What can it do?

* Talk with you using voice
* Help you practice simple English
* Correct important English mistakes
* Explain some corrections in Thai
* Test your English with 5 questions
* Use a camera to look at things
* Show an animated BMO face
* Remember recent conversations
* Search the web when internet is available

## Learning Modes

### Practice Mode

Practice Mode is for normal English practice.

BMO asks simple questions and talks with you one step at a time.

If you make an important mistake, BMO can correct it and give you a simple example.

### Test Mode

Test Mode gives you 5 questions.

BMO checks:

* Grammar
* Vocabulary
* Understanding

At the end, you get an A1 or A2 English level result.

## How it works

The basic flow is:

```text
You speak
   ↓
Whisper turns your voice into text
   ↓
Qwen AI understands your message
   ↓
BMO creates an answer
   ↓
Piper / VachanaTTS turns the answer into voice
   ↓
BMO speaks to you
```

The project uses:

* **Ollama + Qwen 3.5 4B** for AI
* **Whisper.cpp** for speech-to-text
* **Piper** for English voice
* **VachanaTTS** for Thai voice
* **OpenWakeWord** for "Hey BMO"
* **Tkinter** for the BMO screen
* **Raspberry Pi Camera** for vision

## Hardware

Recommended:

* Raspberry Pi 4 with 4 GB RAM or more
* Raspberry Pi 5
* Microphone
* Speaker
* Screen
* Raspberry Pi Camera (optional)

The project is made for Raspberry Pi OS.

## Installation

First, clone this project:

```bash
git clone https://github.com/Vixaus/bmo-english-teacher.git
cd bmo-english-teacher
```

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download the AI model:

```bash
ollama pull qwen3.5:4b
```

### 2. Run setup

```bash
chmod +x setup.sh start_agent.sh
./setup.sh
```

The setup script will install the main tools and Python packages.

It will also set up Whisper and the default English voice.

### 3. Start BMO

```bash
./start_agent.sh
```

## How to use

After BMO starts, you can say:

```text
Hey BMO
```

Then choose:

```text
Practice
```

or:

```text
Test
```

You can also say:

```text
Practice Mode
```

or:

```text
Test Mode
```

### Keyboard controls

If the "Hey BMO" wake word is not available, you can use the keyboard.

* **Enter** — start or stop voice recording
* **Space** — stop BMO while it is speaking
* **Esc** — leave fullscreen mode

When using Enter instead of the wake word, start by saying:

```text
Hey BMO
```

## Camera

BMO can use the Raspberry Pi camera.

For example, you can ask BMO about something in front of the camera.

The camera rotation can be changed inside:

```text
config.json
```

## Configuration

Main settings are inside:

```text
config.json
```

Example:

```json
{
  "text_model": "qwen3.5:4b",
  "vision_model": "qwen3.5:4b",
  "voice_model": "piper/en_GB-semaine-medium.onnx",
  "whisper_model": "whisper.cpp/models/ggml-base.en.bin",
  "chat_memory": true,
  "camera_rotation": 180,
  "wake_word_name": "Hey BMO"
}
```

You can change things like:

* AI model
* Voice
* Microphone
* Camera rotation
* Chat memory
* Wake word

## Project Structure

```text
agent.py
```

Main BMO program.

```text
runtime_helpers.py
```

Helper functions used by BMO.

```text
skills/
```

Rules for Practice Mode and Test Mode.

```text
faces/
```

BMO face animations.

```text
sounds/
```

BMO sound effects.

```text
config.json
```

Main settings.

```text
setup.sh
```

Installs and prepares the project.

```text
start_agent.sh
```

Starts BMO.

## Notes

The "Hey BMO" wake word needs a working `wakeword.onnx` model.

If you do not have one, you can still use **Enter** to talk to BMO.

Thai voice also needs the Thai voice model. The normal English voice works without it.

## License

This project uses the MIT License.

BMO and Adventure Time belong to their respective owners.

This is a fan project and is not connected to or officially supported by Cartoon Network or Warner Bros. Discovery.
