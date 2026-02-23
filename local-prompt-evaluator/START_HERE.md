# Quick Start Guide - Local Prompt Evaluator

## ⚠️ IMPORTANT: You Have TWO Projects!

### OLD Project (Don't use this one)
- Location: `D:\Prompt Engineering\prompt-evaluation-framework\`
- Uses: Flask backend with Gemini API
- Frontend: Old React app

### NEW Project (Use this one!) ✅
- Location: `D:\Prompt Engineering\local-prompt-evaluator\`
- Uses: FastAPI backend with local DistilBERT
- Frontend: New React app with Vite

## 🚀 How to Start the NEW Project

### Step 1: Start Backend API

Open Terminal 1:
```bash
cd "D:\Prompt Engineering\local-prompt-evaluator"
python main.py api
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 2: Install Frontend Dependencies (First Time Only)

Open Terminal 2:
```bash
cd "D:\Prompt Engineering\local-prompt-evaluator\frontend"
npm install
```

### Step 3: Start Frontend

In Terminal 2:
```bash
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:3000/
```

### Step 4: Open Browser

Go to: `http://localhost:3000`

You should see:
- Header: "Local Prompt Evaluator"
- Status: "API Online" (green dot)
- Three tabs: Single Evaluation, Batch Processing, Compare Responses

## ❌ Common Mistakes

### Mistake 1: Running Old Frontend
```bash
# WRONG - This is the old project
cd "D:\Prompt Engineering\prompt-evaluation-framework\frontend"
npm run dev
```

```bash
# CORRECT - This is the new project
cd "D:\Prompt Engineering\local-prompt-evaluator\frontend"
npm run dev
```

### Mistake 2: Backend Not Running
If you see "API Offline" or "API Disconnected":
- Make sure Terminal 1 is running `python main.py api`
- Check that it shows "Uvicorn running on http://0.0.0.0:8000"

### Mistake 3: Model Not Trained
If API returns "Model not loaded":
```bash
# Train the model first (2-4 hours)
python main.py train
```

## 🔧 Troubleshooting

### "Cannot find path frontend"
You're in the wrong directory. Use:
```bash
cd "D:\Prompt Engineering\local-prompt-evaluator\frontend"
```

### "Module 'uvicorn' not found"
Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend shows "Backend API is not responding"
1. Check Terminal 1 - is the API running?
2. Go to `http://localhost:8000/health` - should return JSON
3. Check the API URL in `frontend/src/services/api.js`

## ✅ Success Checklist

- [ ] Backend running on Terminal 1 (port 8000)
- [ ] Frontend running on Terminal 2 (port 3000)
- [ ] Browser shows "API Online" status
- [ ] Can evaluate prompts and see results

## 📝 Quick Test

1. Go to `http://localhost:3000`
2. Click "Single Evaluation" tab
3. Enter:
   - Prompt: "What is AI?"
   - Response: "AI is artificial intelligence"
4. Click "Evaluate"
5. See score and quality badge

If this works, you're all set! 🎉
