# 🧠 Local Prompt Evaluator

A **full-stack, locally-running AI framework** that evaluates the quality of prompt-response pairs using a fine-tuned **DistilBERT** model. No paid API calls needed for evaluation — everything runs on your machine.

---

## 🎯 What It Does

You provide:
- **A Prompt** — the question or instruction given to an AI
- **An AI Response** — the answer produced by any AI model

The system returns:
- **Score** — quality rating from `0.0` to `5.0`
- **Quality Label** — `poor` / `fair` / `good` / `excellent`
- **Confidence** — how certain the model is (0–100%)

---

## ✨ Features

- ✅ **Local inference** — no API calls, fully offline evaluation
- ✅ **Fine-tuned DistilBERT** — trained on the `nvidia/HelpSteer2` dataset
- ✅ **REST API** — FastAPI backend for easy integration
- ✅ **React Frontend** — clean UI to evaluate prompts interactively
- ✅ **Batch Evaluation** — evaluate multiple prompt-response pairs at once
- ✅ **Response Comparison** — compare multiple AI responses to the same prompt and rank them

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | DistilBERT (HuggingFace Transformers) |
| **Training Data** | nvidia/HelpSteer2 |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React.js + Vite |
| **Deep Learning** | PyTorch |
| **Language** | Python 3.10+ / Node.js |

---

## 📁 Project Structure

```
local-prompt-evaluator/
├── api/
│   └── app.py              # FastAPI REST API
├── evaluation/
│   └── evaluator.py        # Core inference engine
├── training/
│   ├── train.py            # Training loop
│   ├── model.py            # DistilBERT model definition
│   ├── dataset.py          # PyTorch dataset loader
│   └── prepare_data.py     # Data preparation script
├── frontend/               # React.js Vite frontend
│   └── src/
├── models/
│   ├── prompt_evaluator/   # Saved fine-tuned model
│   └── tokenizer/          # Saved tokenizer
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
├── logs/
│   └── training_history.csv
├── config.py               # Central configuration
├── main.py                 # CLI entry point
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd local-prompt-evaluator
```

### 2. Set Up Python Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Quick Start

### Start the Backend (Terminal 1)
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Backend will be available at: **http://localhost:8000**

### Start the Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
Frontend will be available at: **http://localhost:3000**

---

## 🖥️ Usage

### Option 1: Frontend UI
1. Open **http://localhost:3000** in your browser
2. Enter your **Prompt** in the first text box
3. Enter the **AI Response** in the second text box
4. Click **Evaluate**
5. View your score, quality label, and confidence

### Option 2: REST API

**Evaluate a single prompt-response pair:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "response": "Machine learning is a subset of AI that enables systems to learn from data."
  }'
```

**Response:**
```json
{
  "score": 4.2,
  "quality": "good",
  "confidence": 0.84,
  "binary_score": 0.84,
  "timestamp": "2026-02-23T11:00:00"
}
```

### Option 3: Python Script
```python
from evaluation.evaluator import PromptEvaluator

evaluator = PromptEvaluator()

result = evaluator.evaluate(
    prompt="Explain neural networks.",
    response="Neural networks are computing systems inspired by the human brain."
)

print(f"Score: {result['score']:.2f}/5.0")
print(f"Quality: {result['quality']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

---

## 📊 Quality Scoring

| Label | Score Range | Binary Score |
|---|---|---|
| ⭐ Excellent | 4.0 – 5.0 | ≥ 0.8 |
| 🟢 Good | 3.0 – 3.99 | ≥ 0.6 |
| 🟡 Fair | 2.0 – 2.99 | ≥ 0.4 |
| 🔴 Poor | 0.0 – 1.99 | < 0.4 |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/evaluate` | Evaluate a prompt-response pair |
| `POST` | `/evaluate/batch` | Evaluate multiple pairs |
| `POST` | `/compare` | Compare multiple responses to one prompt |
| `GET` | `/model/info` | Get model metadata |

---

## 🏋️ Training

### Prepare Data
```bash
python main.py prepare
```

### Train the Model
```bash
python main.py train
```

### Configuration (`config.py`)
| Parameter | Default | Description |
|---|---|---|
| `epochs` | `1` | Number of training epochs |
| `batch_size` | `8` | Training batch size |
| `learning_rate` | `2e-5` | AdamW learning rate |
| `max_samples` | `500` | Dataset sample limit |
| `max_length` | `512` | Token sequence length |

### Training Results
| Metric | Train | Test |
|---|---|---|
| Accuracy | 80.4% | 82.6% |
| F1 Score | 0.890 | 0.905 |
| Loss | 0.552 | 0.456 |

---

## 🔧 Configuration

All settings are managed in `config.py`. You can also override via environment variables:

```bash
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-5
DEVICE=cuda         # Use GPU if available
API_PORT=8000
```

---

## 📦 Dependencies

```
torch
transformers
datasets
pandas
scikit-learn
tqdm
fastapi
uvicorn
pydantic
```

---

## 📈 Model Architecture

- **Base Model:** `distilbert-base-uncased` (66M parameters)
- **Task:** Binary classification → scaled to 0–5 score
- **Input:** `Prompt: {prompt}\n\nResponse: {response}` (max 512 tokens)
- **Output:** Probability score → quality label

---

## 🗒️ License

This project is for educational and research purposes.

---

## 👨‍💻 Author

Built as a full-stack prompt evaluation framework using the **LLM-as-a-judge** pattern with local DistilBERT fine-tuning.
