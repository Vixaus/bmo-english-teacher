#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 BMO English Teacher Setup Script${NC}"

# 1. Install System Dependencies (The "Hidden" Requirements)
echo -e "${YELLOW}[1/6] Installing System Tools (apt)...${NC}"
sudo apt update
sudo apt install -y python3-tk python3-dev python3-venv libasound2-dev portaudio19-dev liblapack-dev libblas-dev cmake build-essential espeak-ng rpicam-apps curl wget git

# 2. Create Folders
echo -e "${YELLOW}[2/6] Creating Folders...${NC}"
mkdir -p piper
mkdir -p voices # Optional custom BMO and Thai voice assets
mkdir -p sounds/greeting_sounds
mkdir -p sounds/thinking_sounds
mkdir -p sounds/ack_sounds
mkdir -p sounds/error_sounds
mkdir -p faces/idle
mkdir -p faces/listening
mkdir -p faces/thinking
mkdir -p faces/speaking
mkdir -p faces/error
mkdir -p faces/warmup

# 3. Download Piper (Architecture Check)
echo -e "${YELLOW}[3/6] Setting up Piper TTS...${NC}"
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    # FIXED: Using the specific 2023.11.14-2 release known to work on Pi
    if [ ! -x "piper/piper" ]; then
        curl --fail --location -o piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
        tar -xvf piper.tar.gz -C piper --strip-components=1
        rm -f piper.tar.gz
    fi
else
    echo -e "${RED}⚠️  Not on Raspberry Pi (aarch64). Skipping Piper download.${NC}"
fi

# 4. Download Voice Models
echo -e "${YELLOW}[4/6] Downloading Voice Models...${NC}"
# Download default Piper voice as fallback
cd piper
if [ ! -f en_GB-semaine-medium.onnx ]; then
    curl --fail --location -o en_GB-semaine-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx
fi
if [ ! -f en_GB-semaine-medium.onnx.json ]; then
    curl --fail --location -o en_GB-semaine-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json
fi
cd ..

# Download optional Custom BMO Voice
echo -e "${YELLOW}Checking optional custom BMO voice...${NC}"
if [ ! -f voices/bmo-custom.onnx ]; then
    if ! curl --fail --location -o voices/bmo-custom.onnx "https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx"; then
        echo -e "${YELLOW}Optional custom BMO voice unavailable; bundled Piper voice remains default.${NC}"
    fi
fi
if [ ! -f voices/bmo-custom.onnx.json ]; then
    if ! curl --fail --location -o voices/bmo-custom.onnx.json "https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx.json"; then
        echo -e "${YELLOW}Optional custom BMO voice metadata unavailable; bundled Piper voice remains default.${NC}"
    fi
fi
if [ -f voices/bmo-custom.onnx ] || [ -f voices/bmo-custom.onnx.json ]; then
    if [ ! -s voices/bmo-custom.onnx ] || [ "$(wc -c < voices/bmo-custom.onnx)" -lt 1024 ]; then
        echo -e "${YELLOW}Invalid optional BMO voice model. Bundled Piper voice remains default.${NC}"
    elif [ ! -f voices/bmo-custom.onnx.json ] || ! python3 -c 'import json, sys; d=json.load(open("voices/bmo-custom.onnx.json")); r=d.get("audio", d).get("sample_rate"); sys.exit(0 if isinstance(r, int) and r > 0 else 1)' 2>/dev/null; then
        echo -e "${YELLOW}Invalid optional BMO voice metadata. Bundled Piper voice remains default.${NC}"
    else
        echo -e "${GREEN}Optional custom BMO voice found.${NC}"
    fi
else
    echo -e "${YELLOW}No optional custom BMO voice found; bundled Piper voice remains default.${NC}"
fi

if [ ! -f voices/th_m_1.onnx ]; then
    echo -e "${YELLOW}Thai TTS not ready: add voices/th_m_1.onnx before using VachanaTTS.${NC}"
else
    echo -e "${GREEN}Thai TTS voice found.${NC}"
fi

# 5. Install Python Libraries
echo -e "${YELLOW}[5/6] Installing Python Libraries...${NC}"
# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
# Force rebuild sounddevice to link against the newly installed PortAudio dev libraries
pip install --force-reinstall --no-cache-dir sounddevice
pip install -r requirements.txt

# Whisper.cpp source/build/model setup. Existing trees are preserved.
if [ ! -d whisper.cpp ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git whisper.cpp
fi
# Keep shared libraries beside whisper-cli discoverable after the repository is
# moved. Older builds may contain an absolute runpath and a CMake cache for the
# old checkout location. Only stale generated CMake metadata is discarded.
WHISPER_SOURCE_DIR="$PWD/whisper.cpp"
WHISPER_BUILD_DIR="$WHISPER_SOURCE_DIR/build"
WHISPER_CACHE="$WHISPER_BUILD_DIR/CMakeCache.txt"
if [ -f "$WHISPER_CACHE" ] && ! grep -Fq "CMAKE_HOME_DIRECTORY:INTERNAL=$WHISPER_SOURCE_DIR" "$WHISPER_CACHE"; then
    echo -e "${YELLOW}Whisper build cache belongs to another checkout; regenerating it.${NC}"
    rm -f "$WHISPER_CACHE"
    rm -rf "$WHISPER_BUILD_DIR/CMakeFiles"
fi
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
cmake --build whisper.cpp/build --config Release -j"$(nproc)" --target whisper-cli
if [ ! -f whisper.cpp/models/ggml-base.en.bin ]; then
    curl --fail --location -o whisper.cpp/models/ggml-base.en.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
fi

# Run one real transcription so a missing runtime library or invalid model
# stops setup with a recoverable error instead of producing empty transcripts.
WHISPER_BIN_DIR="$PWD/whisper.cpp/build/bin"
WHISPER_CLI="$WHISPER_BIN_DIR/whisper-cli"
WHISPER_MODEL="$PWD/whisper.cpp/models/ggml-base.en.bin"
WHISPER_SMOKE_WAV="$(mktemp "${TMPDIR:-/tmp}/bmo-whisper-smoke.XXXXXX.wav")"
WHISPER_SMOKE_LOG="$(mktemp "${TMPDIR:-/tmp}/bmo-whisper-smoke.XXXXXX.log")"
python3 - "$WHISPER_SMOKE_WAV" <<'PY'
import struct
import sys
import wave

with wave.open(sys.argv[1], "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(struct.pack("<16000h", *([0] * 16000)))
PY
if ! LD_LIBRARY_PATH="$WHISPER_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$WHISPER_CLI" -m "$WHISPER_MODEL" -l en -t 1 -f "$WHISPER_SMOKE_WAV" -nt \
    >"$WHISPER_SMOKE_LOG" 2>&1; then
    echo -e "${RED}Whisper smoke test failed. Rebuild Whisper or inspect its libraries/model:${NC}"
    tail -n 20 "$WHISPER_SMOKE_LOG"
    rm -f "$WHISPER_SMOKE_WAV" "$WHISPER_SMOKE_LOG"
    exit 1
fi
rm -f "$WHISPER_SMOKE_WAV" "$WHISPER_SMOKE_LOG"
echo -e "${GREEN}Whisper transcription smoke test passed.${NC}"

# 6. Pull AI Models
echo -e "${YELLOW}[6/6] Checking AI Models...${NC}"
if command -v ollama &> /dev/null; then
    ollama pull qwen3.5:4b
else
    echo -e "${RED}❌ Ollama not found. Please install it manually.${NC}"
fi

# 7. OpenWakeWord Model
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Wake-word model missing. Add user-supplied wakeword.onnx trained for 'Hey BMO'; push-to-talk remains available.${NC}"
fi

echo -e "${GREEN}✨ BMO setup complete! Run 'source venv/bin/activate' then 'python agent.py'${NC}"

for required in piper/piper piper/en_GB-semaine-medium.onnx piper/en_GB-semaine-medium.onnx.json whisper.cpp/build/bin/whisper-cli whisper.cpp/models/ggml-base.en.bin; do
    if [ ! -e "$required" ]; then
        echo -e "${RED}Missing required asset: $required${NC}"
    fi
done
if [ ! -f wakeword.onnx ]; then
    echo -e "${YELLOW}PTT fallback ready. Add wakeword.onnx trained for Hey BMO for wake-word activation.${NC}"
fi
