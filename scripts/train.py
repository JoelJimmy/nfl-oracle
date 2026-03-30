"""scripts/train.py — run from project root: python -m scripts.train"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.train import train
if __name__ == "__main__":
    train()
