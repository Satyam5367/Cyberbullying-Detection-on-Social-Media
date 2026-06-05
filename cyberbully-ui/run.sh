#!/bin/bash
echo "============================================================"
echo "  CyberGuard AI - Setup and Launch"
echo "============================================================"
echo "[1/3] Checking Python..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }
echo "[2/3] Installing Flask..."
pip3 install flask --quiet
echo "[3/3] Starting server..."
echo ""
echo "  Open your browser at: http://127.0.0.1:5000"
echo "  Press Ctrl+C to stop"
echo ""
python3 app.py
