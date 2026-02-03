#!/bin/bash
# Activate script - source this to activate the virtual environment

if [ -d ".venv" ]; then
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
    echo "Python: $(which python3)"
    echo "Pip: $(which pip)"
    echo ""
    echo "Ready to run:"
    echo "  python3 main.py"
    echo ""
    echo "To deactivate later: deactivate"
else
    echo "❌ Virtual environment not found!"
    echo "Run ./setup.sh first to create it"
    return 1
fi
