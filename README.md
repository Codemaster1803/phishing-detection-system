# Phishing Detection System

## Overview
Multi-agent phishing detection system using:
- URL analysis
- NLP (TF-IDF + DistilBERT)
- Domain intelligence
- Fusion using Dempster-Shafer theory

## Agents 
## 🧠 NLP Analysis Agent

The NLP Agent analyzes the **text content of a webpage** to detect phishing attempts based on language patterns, intent, and semantic signals.

---

### 🔍 Objective

The goal of this agent is to identify phishing indicators such as:

* Urgency phrases (e.g., "verify now", "account suspended")
* Requests for sensitive information
* Social engineering patterns
* Fake login or security alerts

---

### ⚙️ Approach

The NLP pipeline follows a structured approach:

#### 1. Data Collection

* A **custom dataset** was created using web scraping.
* Sources include:

  * Phishing feeds (e.g., PhishTank-style data)
  * Legitimate websites (e.g., Tranco domains)

---

#### 2. Data Preprocessing

* HTML/text cleaning
* Removal of scripts, styles, and noise
* Normalization of text
* Dataset balancing (phishing vs legitimate)

Preprocessing scripts are available in:

```
agents/nlp_agent/preprocessing/
```

---

#### 3. Model Architecture

The NLP agent uses a **hybrid model approach**:

* **DistilBERT** → for deep contextual understanding
* **TF-IDF + ML model** → for fast and lightweight inference

This ensures:

* High accuracy (BERT)
* Low latency (TF-IDF fallback)

---

#### 4. Training Pipeline

To regenerate dataset and retrain the model:

```
cd agents/nlp_agent/preprocessing
python build_dataset.py
python balance_dataset.py
```

---

### 📦 Model Usage

Due to size constraints, trained models are **not included in the repository**.

To use the NLP agent:

1. Train the model using the provided scripts
   AND
2. Place trained model files inside:

```
agents/nlp_agent/models/
```
---


### ⚡ Key Features

* Custom-built dataset (not pre-downloaded)
* Hybrid NLP model (accuracy + performance)
* Handles real-world phishing language
* Fully integrated with multi-agent fusion system

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