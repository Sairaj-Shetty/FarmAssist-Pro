from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os, json, re, logging, io
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image as PILImage
import google.generativeai as genai

# ====== Load ENV and configure logging ======
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farmassist")

# ====== AI setup ======
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="FarmAssist Pro",
    version="3.0.0",
    description="AI-powered farming assistant with real-time image analysis"
)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODELS ===========================
class TextQueryRequest(BaseModel):
    query: str
    language: str = "en"
    context: Optional[Dict[str, Any]] = None

# ======================== HELPERS ===========================
def build_model(name: str = "gemini-1.5-flash"):
    return genai.GenerativeModel(name)

def safe_json_parse(text: str) -> Dict[str, Any]:
    # Try find first JSON block
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {}

def translate_text(text: str, target_lang: str) -> str:
    if not text or target_lang == "en":
        return text
    try:
        lang_name = "Hindi" if target_lang == "hi" else "Kannada" if target_lang == "kn" else "English"
        prompt = f"Translate this to {lang_name}. Return only the translated text:\n\n{text}"
        resp = build_model().generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        logger.warning(f"translate_text error: {e}")
        return text

def translate_to_en(text: str, source_lang: str) -> str:
    if not text or source_lang == "en":
        return text
    try:
        prompt = f"Translate this to English. Return only the translated text:\n\n{text}"
        resp = build_model().generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        logger.warning(f"translate_to_en error: {e}")
        return text

def translate_list(items: List[str], lang: str) -> List[str]:
    return [translate_text(x, lang) for x in (items or [])]

def get_medicine_link(primary_text: str) -> str:
    t = (primary_text or "").lower()
    mapping = [
        ("mancozeb", "https://www.bighaat.com/products/indofil-m-45-mancozeb-75-wp-500-gm"),
        ("propiconazole", "https://www.bighaat.com/products/tilt-propiconazole-25-ec"),
        ("copper oxychloride", "https://www.bighaat.com/products/copper-oxychloride-50-wp-500gm"),
        ("carbendazim", "https://www.bighaat.com/products/bavistin-carbendazim-50-wp-100-gm"),
        ("sulphur", "https://www.bighaat.com/products/sulfex-80-wdg-sulphur-1-kg"),
        ("imidacloprid", "https://www.bighaat.com/products/command-imidacloprid-17-8-sl"),
        ("chlorpyrifos", "https://www.bighaat.com/collections/insecticides"),
        ("acetamiprid", "https://www.bighaat.com/collections/insecticides"),
        ("thiamethoxam", "https://www.bighaat.com/collections/insecticides"),
        ("neem", "https://www.bighaat.com/products/neem-oil-azadirachtin-300-ppm"),
        ("urea", "https://www.bighaat.com/collections/fertilizers"),
        ("npk", "https://www.bighaat.com/collections/fertilizers"),
        ("dap", "https://www.bighaat.com/collections/fertilizers"),
    ]
    for key, url in mapping:
        if key in t:
            return url
    return "https://www.bighaat.com/collections/pest-disease-management"

def ask_gemini_json(prompt: str, image: PILImage.Image) -> Dict[str, Any]:
    resp = build_model().generate_content([prompt, image])
    parsed = safe_json_parse(resp.text or "")
    return parsed

# ========================= Prompts =========================
PLANT_PROMPT = """
You are an agricultural plant pathologist.
Analyze this plant/crop image and return STRICT JSON only (no extra text) in this schema:
{
  "plant": { "common_name": "", "scientific_name": "", "growth_stage": "" },
  "health": { "score": 0, "observations": [""] },
  "issues": [
    { "name": "", "type": "disease|pest|deficiency", "severity": "mild|moderate|severe", "evidence": [""] }
  ],
  "treatment": {
    "chemical": [ { "product": "", "dose": "", "notes": "" } ],
    "organic":  [ { "method": "", "dose": "", "notes": "" } ]
  },
  "prevention": ["short actionable tips"]
}
Be exact and specific. If uncertain, leave fields empty or omit entries. Do not fabricate.
"""

SOIL_PROMPT = """
You are a soil scientist.
Analyze this soil image and return STRICT JSON only (no extra text) in this schema:
{
  "soil": { "type": "", "color": "", "texture": "", "structure": "" },
  "health": { "ph_estimate": "range like 6.5-7.0", "organic_matter": "low|medium|high", "moisture": "low|medium|high" },
  "nutrients": { "nitrogen": "", "phosphorus": "", "potassium": "", "micronutrients": ["optional"] },
  "suitable_crops": ["", ""],
  "recommendations": ["short actionable tips"],
  "fertilizer_plan": [ { "product": "", "dose": "", "notes": "" } ]
}
Be factual. Do not fabricate if image doesn't support an estimate.
"""

PEST_PROMPT = """
You are an entomologist/plant protection expert.
Analyze this pest image and return STRICT JSON only (no extra text) in this schema:
{
  "pest": { "common_name": "", "scientific_name": "", "life_stage": "", "size_estimate": "" },
  "damage": { "severity": 0, "pattern": "", "affected_parts": [""] },
  "hosts": { "primary": ["", ""], "secondary": [""] },
  "control": {
    "chemical": [ { "product": "", "dose": "", "notes": "" } ],
    "organic":  [ { "method": "", "dose": "", "notes": "" } ]
  },
  "prevention": ["short actionable tips"]
}
Be exact and practical. Do not make up unknowns.
"""

# ========================= API =========================
@app.get("/")
def root():
    return {"message": "FarmAssist Pro API Running", "version": "3.0.0"}

@app.post("/api/validate/image")
async def validate_image(file: UploadFile = File(...), language: str = Form("en")):
    try:
        img_bytes = await file.read()
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        prompt = "Is this image related to agriculture (plants/crops/soil/pests/farming)? Answer strictly 'Yes' or 'No'."
        resp = build_model().generate_content([prompt, image])
        is_agri = "yes" in (resp.text or "").strip().lower()
        message = "Image validated successfully" if is_agri else "Please upload an agriculture-related image"
        return {"success": True, "is_agricultural": is_agri, "message": translate_text(message, language)}
    except Exception as e:
        logger.error(f"/validate error: {e}")
        return {"success": False, "is_agricultural": False, "message": "Validation failed, please try again."}

@app.post("/api/analyze/plant")
async def analyze_plant(file: UploadFile = File(...), language: str = Form("en")):
    try:
        img_bytes = await file.read()
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        data = ask_gemini_json(PLANT_PROMPT, image)
        if not data or "plant" not in data:
            raise HTTPException(status_code=422, detail="AI could not extract plant details. Try a clearer image.")

        plant = data.get("plant", {})
        health = data.get("health", {})
        issues = data.get("issues", [])
        treatment = data.get("treatment", {})
        prevention = data.get("prevention", [])

        # Summary + link
        common = plant.get("common_name") or "Unknown plant"
        sci = plant.get("scientific_name") or ""
        score = health.get("score", 0)
        first_issue = (issues[0].get("name") if issues else "") or ""
        first_chem = ((treatment.get("chemical") or [{}])[0].get("product") or "")
        summary_en = f"Identified: {common}{f' ({sci})' if sci else ''}. Health score: {score}/100."
        if first_issue:
            summary_en += f" Primary issue: {first_issue}."
        if first_chem:
            summary_en += f" Recommended treatment: {first_chem}."
        buy_url = get_medicine_link(first_chem or first_issue or common)

        # Translate user-facing fields (keep names mostly as-is)
        summary = translate_text(summary_en, language)
        for it in issues:
            it["name"] = translate_text(it.get("name",""), language)
            it["type"] = translate_text(it.get("type",""), language)
            it["severity"] = translate_text(it.get("severity",""), language)
            it["evidence"] = translate_list(it.get("evidence", []), language)
        for c in (treatment.get("chemical") or []):
            c["product"] = translate_text(c.get("product",""), language)
            c["dose"] = translate_text(c.get("dose",""), language)
            c["notes"] = translate_text(c.get("notes",""), language)
        for o in (treatment.get("organic") or []):
            o["method"] = translate_text(o.get("method",""), language)
            o["dose"] = translate_text(o.get("dose",""), language)
            o["notes"] = translate_text(o.get("notes",""), language)
        prevention = translate_list(prevention, language)

        return {
            "success": True,
            "plant": plant,
            "health": health,
            "issues": issues,
            "treatment": treatment,
            "prevention": prevention,
            "summary": summary,
            "medicine_links": {"url": buy_url}
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"/analyze/plant error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze plant image.")

@app.post("/api/analyze/soil")
async def analyze_soil(file: UploadFile = File(...), language: str = Form("en")):
    try:
        img_bytes = await file.read()
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        data = ask_gemini_json(SOIL_PROMPT, image)
        if not data or "soil" not in data:
            raise HTTPException(status_code=422, detail="AI could not extract soil details. Try a clearer image.")

        soil = data.get("soil", {})
        health = data.get("health", {})
        nutrients = data.get("nutrients", {})
        suitable_crops = data.get("suitable_crops", [])
        recommendations = data.get("recommendations", [])
        fertilizer_plan = data.get("fertilizer_plan", [])

        soil_type = soil.get("type", "Soil")
        ph = health.get("ph_estimate", "")
        om = health.get("organic_matter", "")
        summary_en = f"{soil_type} detected. pH: {ph or 'N/A'}, Organic matter: {om or 'N/A'}. Recommended crops: {', '.join(suitable_crops[:3]) if suitable_crops else 'N/A'}."
        summary = translate_text(summary_en, language)

        suitable_crops = translate_list(suitable_crops, language)
        recommendations = translate_list(recommendations, language)
        for f in fertilizer_plan:
            f["product"] = translate_text(f.get("product",""), language)
            f["dose"] = translate_text(f.get("dose",""), language)
            f["notes"] = translate_text(f.get("notes",""), language)

        first_fert = (fertilizer_plan[0]["product"] if fertilizer_plan else "") or "fertilizer"
        buy_url = get_medicine_link(first_fert)

        return {
            "success": True,
            "soil": soil,
            "health": health,
            "nutrients": nutrients,
            "suitable_crops": suitable_crops,
            "recommendations": recommendations,
            "fertilizer_plan": fertilizer_plan,
            "summary": summary,
            "medicine_links": {"url": buy_url}
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"/analyze/soil error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze soil image.")

@app.post("/api/analyze/pest")
async def analyze_pest(file: UploadFile = File(...), language: str = Form("en")):
    try:
        img_bytes = await file.read()
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        data = ask_gemini_json(PEST_PROMPT, image)
        if not data or "pest" not in data:
            raise HTTPException(status_code=422, detail="AI could not extract pest details. Try a clearer image.")

        pest = data.get("pest", {})
        damage = data.get("damage", {})
        hosts = data.get("hosts", {})
        control = data.get("control", {})
        prevention = data.get("prevention", [])

        cname = pest.get("common_name", "Pest")
        sev = damage.get("severity", 0)
        chem_first = (control.get("chemical") or [{}])[0].get("product", "")
        summary_en = f"{cname} detected. Damage severity: {sev}%. Recommended treatment: {chem_first or 'See control measures'}."
        summary = translate_text(summary_en, language)

        for c in (control.get("chemical") or []):
            c["product"] = translate_text(c.get("product",""), language)
            c["dose"] = translate_text(c.get("dose",""), language)
            c["notes"] = translate_text(c.get("notes",""), language)
        for o in (control.get("organic") or []):
            o["method"] = translate_text(o.get("method",""), language)
            o["dose"] = translate_text(o.get("dose",""), language)
            o["notes"] = translate_text(o.get("notes",""), language)
        prevention = translate_list(prevention, language)

        buy_url = get_medicine_link(chem_first or cname)

        return {
            "success": True,
            "pest": pest,
            "damage": damage,
            "hosts": hosts,
            "control": control,
            "prevention": prevention,
            "summary": summary,
            "medicine_links": {"url": buy_url}
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"/analyze/pest error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze pest image.")

@app.post("/api/chat/support")
async def chat_support(req: TextQueryRequest):
    try:
        query_en = translate_to_en(req.query, req.language)
        prompt = f"""
You are an Indian Agricultural Advisor.
Farmer asked: {query_en}
Provide clear, practical guidance under 180 words. Use short bullets where useful.
Return just the answer text.
"""
        resp = build_model().generate_content(prompt)
        answer_en = (resp.text or "").strip()
        answer = translate_text(answer_en, req.language)
        return { "success": True, "response": { "answer": answer }, "timestamp": datetime.now().isoformat() }
    except Exception as e:
        logger.error(f"/chat/support error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)