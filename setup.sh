#!/usr/bin/env bash
set -e

VENV_DIR="venv"

echo "=== Creating virtual environment in ./$VENV_DIR ==="
python3 -m venv "$VENV_DIR"

echo "=== Installing dependencies ==="
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# sounddevice needs PortAudio for live playback
if command -v apt-get &>/dev/null; then
    echo "=== Installing PortAudio system lib ==="
    sudo apt-get install -y libportaudio2
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the venv first:"
echo "  source venv/bin/activate"
echo ""
echo "Then run:"
echo "  uvicorn server:app --host 0.0.0.0 --port 8000"
echo "  python test_api.py"
