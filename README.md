# 🔐 Phishing Detection System (Multi-Agent Architecture)

A **production-inspired phishing detection system** that combines multiple intelligent agents to detect malicious websites in real-time using a Chrome Extension and FastAPI backend.

---

## 🚀 Overview

This system uses a **multi-agent architecture** where each agent analyzes a different aspect of a website:

* 🌐 URL structure
* 🧠 Page content (NLP)
* 🌍 Domain intelligence
* ⚖️ Decision fusion (final verdict)

The results are combined using **Dempster-Shafer Theory** to produce a robust and explainable phishing prediction.

---

## 🏗️ Architecture

```text
Chrome Extension
       ↓
   FastAPI Backend
       ↓
 ┌───────────────┐
 │  URL Agent     │
 │  NLP Agent     │
 │  Domain Agent  │
 └───────────────┘
       ↓
  Fusion Agent
       ↓
 Final Verdict
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI (Python)
* **ML/NLP:** Scikit-learn, Transformers (DistilBERT)
* **Data Processing:** Pandas, NumPy
* **Browser Integration:** Chrome Extension (JavaScript)
* **APIs:** WHOIS, DNS, SSL, VirusTotal (optional)

---

## 🧠 Agents Overview

---

### 🔗 URL Analysis Agent

Analyzes URL structure using machine learning and feature engineering.

#### Key Features:

* URL length, entropy, and structure
* Suspicious keywords (login, verify, secure)
* Subdomain and query analysis
* Obfuscation and encoding detection
* WHOIS enrichment and domain metadata

#### Model:

* Ensemble of:

  * Random Forest
  * Extra Trees
  * Gradient Boosting

---

### 🧠 NLP Analysis Agent

Analyzes webpage text to detect phishing intent using semantic understanding.

#### Objective:

Detect:

* Urgency language (“verify now”)
* Credential requests
* Social engineering patterns

#### Pipeline:

**1. Data Collection**

* Custom dataset built via scraping
* Includes phishing and legitimate content

**2. Preprocessing**

* Text cleaning
* Noise removal
* Dataset balancing

**3. Models**

* DistilBERT (deep understanding)
* TF-IDF + ML model (fast fallback)

#### Training:

```bash
cd agents/nlp_agent/preprocessing
python build_dataset.py
python balance_dataset.py
```

#### Output:

```json
{
  "score": 0.87,
  "label": "Phishing"
}
```

---

### 🌐 Domain Intelligence Agent

Evaluates domain trust using real-world security signals.

#### Checks Performed:

1. Domain age (WHOIS)
2. SSL certificate validity
3. DNS records (A, MX)
4. Suspicious TLD detection
5. IP-based URL detection
6. Domain pattern analysis (brand + keywords)
7. VirusTotal (optional)

#### Features:

* No ML required (rule-based intelligence)
* Explainable risk factors
* Fast and reliable

---

### ⚖️ Fusion Agent (Decision Engine)

Combines outputs from all agents into a final verdict.

#### Method:

* **Dempster-Shafer Evidence Theory**
* Weighted score aggregation

#### Inputs:

* URL Agent score
* NLP Agent score
* Domain Agent score

#### Output:

* Final label (Safe / Suspicious / Phishing)
* Confidence level
* Explanation
* Conflict analysis between agents

---

## 🔄 Workflow

1. Chrome Extension captures:

   * URL
   * Page text

2. API receives request (`/analyze`)

3. Agents run in parallel:

   * URL Agent
   * NLP Agent
   * Domain Agent

4. Fusion Agent combines results

5. Final verdict returned to extension

---

## 📦 Project Structure

```text
Phishing_detection/
│
├── agents/
│   ├── url_agent/
│   ├── nlp_agent/
│   │   ├── preprocessing/
│   │   └── models/ (ignored)
│   ├── domain_agent/
│   └── fusion_agent/
│
├── api/
│   └── api.py
│
├── extension/
│
├── models/ (ignored)
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Start API server

```bash
uvicorn api:app --reload
```

---

### 3. Load Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable Developer Mode
3. Click **Load Unpacked**
4. Select `extension/` folder

---

## ⚠️ Notes

* Trained models are **not included** due to size
* Datasets are **not included** (custom scraped)

### To use models:

* Train using scripts
  OR
* Place models in:

```bash
models/
```

---

## 📊 Output Example

```json
{
  "url": "http://secure-login-paypal.xyz",
  "final_label": "Phishing",
  "final_probability": 0.84,
  "confidence": "High",
  "explanation": "Suspicious URL + new domain + phishing text",
  "agents": {
    "url_agent": { "score": 0.82 },
    "nlp_agent": { "score": 0.91 },
    "domain_agent": { "score": 0.75 }
    }
}
```

---

## 🔥 Key Features

* Multi-agent architecture
* Hybrid ML + rule-based detection
* Custom dataset creation
* Real-time Chrome extension integration
* Explainable AI decisions
* Parallel processing for performance

---

## 🚀 Future Improvements

* Real-time threat intelligence APIs
* Model optimization for latency
* UI improvements in extension
* Online learning / model updates
* Deployment using Docker / Cloud

---

## 👨‍💻 Author

Developed as an end-to-end phishing detection system combining:

* Machine Learning
* NLP
* Cybersecurity principles
* Full-stack integration
