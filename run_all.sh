#!/bin/bash
set -e

echo "[1/6] Data Processing..."
python src/load_data.py

echo "[2/6] Model Training..."
python src/baseline_models.py

echo "[3/6] Portfolio Construction..."
python src/construct_portfolio.py

echo "[4/6] Performance Evaluation..."
python src/performance.py

echo "[5/6] Generating Deck..."
python src/generate_deck.py

echo "[6/6] All steps completed successfully!" 