# Local Prompt Evaluator - Frontend

Modern React frontend for the Local Prompt Evaluator with a beautiful dark theme UI.

## Features

- 🎨 **Beautiful Dark Theme** - Vibrant colors with smooth animations
- 📝 **Single Evaluation** - Evaluate individual prompt-response pairs
- 📦 **Batch Processing** - Evaluate multiple items at once with CSV export
- 🔄 **Response Comparison** - Compare multiple responses side-by-side
- 📊 **Score Visualization** - Circular progress indicators and quality badges
- ⚡ **Fast & Responsive** - Built with Vite for optimal performance
- 📱 **Mobile-Friendly** - Fully responsive design

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Lightning-fast build tool
- **Axios** - HTTP client for API calls
- **CSS Variables** - Themeable design system
- **Inter Font** - Clean, modern typography

## Getting Started

### Prerequisites

- Node.js 16+ installed
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

## Project Structure

```
frontend/
├── index.html              # HTML entry point
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
├── src/
│   ├── main.jsx           # React entry point
│   ├── App.jsx            # Main app with tab navigation
│   ├── App.css            # Global styles
│   ├── components/
│   │   ├── PromptInput.jsx        # Single evaluation form
│   │   ├── ResultsDisplay.jsx     # Results visualization
│   │   ├── ScoreCard.jsx          # Circular score display
│   │   ├── BatchEvaluator.jsx     # Batch processing
│   │   └── CompareResponses.jsx   # Response comparison
│   └── services/
│       └── api.js         # API service layer
```

## API Integration

The frontend connects to the backend API at `http://localhost:8000`. Make sure the backend is running before starting the frontend.

### API Endpoints Used

- `POST /evaluate` - Single evaluation
- `POST /batch-evaluate` - Batch processing
- `POST /compare` - Response comparison
- `GET /health` - Health check
- `GET /model-info` - Model metadata

## Features Guide

### Single Evaluation

1. Enter your prompt in the first textarea
2. Enter the AI response in the second textarea
3. Click "Evaluate"
4. View score, quality, and confidence
5. Copy or export results

### Batch Processing

1. Add multiple prompt-response pairs
2. Click "Evaluate X Items"
3. View results table with summary statistics
4. Export to CSV

### Response Comparison

1. Enter a single prompt
2. Add multiple responses (2-10)
3. Click "Compare"
4. See ranked results with best response highlighted

## Customization

### Colors

Edit CSS variables in `src/App.css`:

```css
:root {
  --primary: #00D9FF;      /* Cyan */
  --secondary: #FF6B9D;    /* Pink */
  --success: #00E676;      /* Green */
  --warning: #FFB300;      /* Amber */
  --danger: #FF5252;       /* Red */
}
```

### API URL

Edit `src/services/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## Troubleshooting

### "API Offline" Status

- Make sure the backend is running: `python main.py api`
- Check the API URL in `src/services/api.js`
- Verify CORS is enabled in backend

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

Edit `vite.config.js` to change the port:

```javascript
server: {
  port: 3001,  // Change from 3000
}
```

## Screenshots

### Single Evaluation
Beautiful form with real-time character count and validation.

### Results Display
Circular progress indicator with quality badges and confidence meter.

### Batch Processing
Table view with summary statistics and CSV export.

### Response Comparison
Side-by-side comparison with ranking and best response highlight.

## License

Part of the Local Prompt Evaluator project.

## Support

For issues or questions, check the main project README.
