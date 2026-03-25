# Phishing Detection System

## Overview
Multi-agent phishing detection system using:
- URL analysis
- NLP (TF-IDF + DistilBERT)
- Domain intelligence
- Fusion using Dempster-Shafer theory

## Features
- Chrome extension integration
- Real-time detection
- Whitelisting

## Note
Model files are not included due to size.
Download them separately or retrain using provided scripts.

## Run Project

pip install -r requirements.txt
uvicorn api:app --reload