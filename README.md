FarmAssist Pro
Real-time AI for farmers: upload a photo to identify plant/soil/pest, diagnose issues, get exact treatments with “Buy Online” links, and translate to English/Hindi/Kannada.

Features
Plant/Soil/Pest image analysis (Gemini)
Exact plant name, health score, issues, treatments (chemical/organic)
“Buy Online” links mapped to products
AI chat support
Translations: en/hi/kn (via Gemini)
Stack
Backend: FastAPI, google-generativeai, Pillow
Frontend: Next.js (App Router), Tailwind/UI kit
Quick Start
Backend

backend/.env
text

GEMINI_API_KEY=YOUR_KEY
HOST=0.0.0.0
PORT=8000
Install & run
text

cd backend
pip install -r requirements.txt
python start.py
API docs: http://localhost:8000/docs

Frontend

frontend/.env.local
text

NEXT_PUBLIC_API_URL=http://localhost:8000
Install & run
text

cd frontend
npm install
npm run dev
App: http://localhost:3000

Key Endpoints
POST /api/validate/image (file, language)
POST /api/analyze/plant (file, language)
POST /api/analyze/soil (file, language)
POST /api/analyze/pest (file, language)
POST /api/chat/support ({ query, language })
Example

text

curl -X POST http://localhost:8000/api/analyze/plant \
  -F "file=@/path/to/plant.jpg" -F "language=en"
Notes
No googletrans (Python 3.13 safe). Translation via Gemini.
Set CORS origin in production.
