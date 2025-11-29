from __future__ import annotations

"""ai_mindmap_backend.py — robust and feature-rich FastAPI micro-service
========================================================================
* **AI-Powered Language Detection**: The Gemini model itself detects and returns the language code.
* **Resilient JSON extraction**: supports ```json …```, ``` …```, and raw JSON.
* **Advanced styling**: Custom themes, fonts, arrows, and auto-emojis.
* **Multi-format export**: PNG, SVG, and PDF support.
"""

import os
import re
import json
import tempfile
import uuid
import asyncio
import logging
import pprint
from pathlib import Path
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions as gapi_exc
import graphviz
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_GENAI_API_KEY missing — set it in .env or env vars")

genai.configure(api_key=API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash-latest")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))
STATIC_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

###############################################################################
# Prompt sent to Gemini (Now includes language detection instruction)
###############################################################################

PROMPT = (
    "Tu es un expert en organisation et en structuration d'idées. "
    "Ton objectif est de renvoyer un objet JSON principal. Cet objet doit contenir deux clés de premier niveau : "
    "1. `langue_detectee`: une chaîne contenant le code de langue ISO 639-1 de la langue principale du texte de l'image (ex: 'fr', 'en', 'de'). "
    "2. `mind_map`: un objet JSON hiérarchique représentant la carte mentale. "
    "Dans l'objet `mind_map`, pour chaque idée, la clé est le nom de l'idée. "
    "La valeur est un objet contenant 'importance' ('haute', 'moyenne', ou 'basse') et 'sous_taches' (un objet contenant les idées enfants, suivant cette même structure)."
)

###############################################################################
# Style Definitions & Helper Functions
###############################################################################

def get_contrasting_font_color(hex_color: str) -> str:
    if not hex_color.startswith('#'): return "black"
    try:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return 'white' if luminance < 140 else 'black'
    except (ValueError, IndexError): return "black"

THEMES = {
    "default": {"bgcolor": "white", "fontcolor": "black", "root": "#FF7F50", "node": "#ADD8E6", "details": "#FFFFE0", "importance": {"haute": "#FFB6C1", "moyenne": "#90EE90", "basse": "#D3D3D3"}},
    "dark": {"bgcolor": "#2E2E2E", "fontcolor": "white", "root": "#FF6B6B", "node": "#4ECDC4", "details": "#FFE66D", "importance": {"haute": "#D94848", "moyenne": "#40C057", "basse": "#495057"}},
    "corporate": {"bgcolor": "white", "fontcolor": "#003366", "root": "#003366", "node": "#D8E2F3", "details": "#F5F5F5", "importance": {"haute": "#BF0A30", "moyenne": "#FFC72C", "basse": "#E1E1E1"}}
}
EMOJI_MAP = {"tâche": "📝", "idée": "💡", "projet": "🚀", "réunion": "👥", "bug": "🐞", "question": "❓", "important": "❗", "urgent": "🔥", "analyse": "📊", "design": "🎨", "code": "💻", "test": "🧪", "déploiement": "📦", "marketing": "📈", "vente": "💰", "budget": "💲", "client": "🧑‍💼", "sécurité": "🔒", "optimisation": "⚙️", "rapport": "📄", "anxiété": "😟", "stress": "🤯", "social": "🤝", "famille": "👨‍👩‍👧‍👦", "schule": "🏫", "kurs": "🎓"}

MULTI_LANG_LEGEND = {
    "fr": {"title": "Légende", "haute": "Haute Importance", "moyenne": "Moyenne Importance", "basse": "Basse Importance"},
    "en": {"title": "Legend", "haute": "High Importance", "moyenne": "Medium Importance", "basse": "Low Importance"},
    "de": {"title": "Legende", "haute": "Hohe Wichtigkeit", "moyenne": "Mittlere Wichtigkeit", "basse": "Geringe Wichtigkeit"},
    "es": {"title": "Leyenda", "haute": "Alta Importancia", "moyenne": "Importancia Media", "basse": "Baja Importancia"},
    "it": {"title": "Leggenda", "haute": "Alta Importanza", "moyenne": "Media Importanza", "basse": "Bassa Importanza"},
}

def get_emoji(text: str) -> str:
    text_lower = text.lower()
    for keyword, emoji in EMOJI_MAP.items():
        if keyword in text_lower: return emoji
    return ""
    

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)

def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text.strip()
    try: return json.loads(candidate)
    except json.JSONDecodeError as exc: raise ValueError("Gemini response did not contain valid JSON") from exc

def _normalise(map_data: Dict[str, Any]) -> Dict[str, Any]:
    def build_tasks(data: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        task_list = []
        if not isinstance(data, dict): return []
        for name, content in data.items():
            if not isinstance(content, dict): continue
            task = {
                "id": str(uuid.uuid4()), "nom_tache": name,
                "importance": content.get("importance", "defaut").lower(),
                "details": content.get("details", ""), "emoji": "", "dependances": [],
                "sous_taches": build_tasks(content.get("sous_taches"))
            }
            task_list.append(task)
        return task_list
    
    if "idee_principale" in map_data and "taches" in map_data: return map_data
    
    root_name, tasks_data = "Mind Map", map_data
    if len(map_data) == 1:
        root_key, root_value = next(iter(map_data.items()))
        root_name = root_key
        tasks_data = root_value.get("sous_taches", {}) if isinstance(root_value, dict) else {}
    
    return {"idee_principale": {"id": "root", "nom": root_name}, "taches": build_tasks(tasks_data)}



def analyse_image(path: str) -> Tuple[Dict[str, Any], str]:
    """Calls Gemini, returns a tuple of (normalized_map_data, language_code)."""
    try:
        with Image.open(path) as img:
            response = MODEL.generate_content([PROMPT, img], request_options={"timeout": 120})
        log.info("Received response from Gemini. Text: %s", response.text)
    except gapi_exc.GoogleAPIError as exc:
        raise RuntimeError(f"Gemini API error: {exc.message}") from exc
    
    raw_response = _extract_json(response.text)
    
    lang_code = raw_response.get("langue_detectee", "en")
    
    normalized_data = _normalise(mind_map_part)
    
    return normalized_data, lang_code

def _render_graph(data: Dict[str, Any], lang_code: str, layout: int, output_format: str, theme_name: str, arrow_style: str, outfile_base: Path) -> None:
    theme = THEMES.get(theme_name, THEMES["default"])
    colors = {"haute": theme["importance"]["haute"], "moyenne": theme["importance"]["moyenne"], "basse": theme["importance"]["basse"], "defaut": theme["node"]}
    legend_labels = MULTI_LANG_LEGEND.get(lang_code, MULTI_LANG_LEGEND["en"])

    dot = graphviz.Digraph(comment="MindMap", format=output_format)
    dot.attr(dpi="220", fontname="Arial", fontsize="12", charset="utf-8", overlap="false", splines="true", bgcolor=theme["bgcolor"], fontcolor=theme["fontcolor"])
    
    engine, rankdir = {1: ("dot", "TB"), 2: ("circo", None), 3: ("dot", "LR")}.get(int(layout), ("dot", "TB"))
    dot.engine = engine
    if rankdir: dot.attr(rankdir=rankdir)
    
    dot.attr('node', style='filled', shape='box', fontname='Arial', fontsize='14')
    dot.attr('edge', color=theme["fontcolor"], arrowhead=arrow_style, arrowsize='0.7')
    
    id_map, root = {}, data["idee_principale"]
    root_id = root.get("id", "root")
    id_map[root_id] = root_id
    dot.node(root_id, f"🎯\n{root['nom']}", shape="ellipse", fillcolor=theme["root"], fontsize='18', fontcolor=get_contrasting_font_color(theme["root"]))

    def visit(tasks: List[Dict[str, Any]], parent: str) -> None:
        for t in tasks:
            tid, label = t["id"], t.get("nom_tache", "")
            id_map[tid] = tid
            emoji = t.get("emoji") or get_emoji(label)
            fill_color = colors.get(t.get("importance", "defaut"), colors["defaut"])
            font_color = get_contrasting_font_color(fill_color)
            dot.node(tid, f"{emoji} {label}".strip(), fillcolor=fill_color, fontcolor=font_color)
            dot.edge(parent, tid)
            if t.get("details"):
                details_id, details_color = f"details_{tid}", theme["details"]
                details_font_color = get_contrasting_font_color(details_color)
                dot.node(details_id, f"📝 {t['details']}", shape="note", fillcolor=details_color, fontcolor=details_font_color)
                dot.edge(tid, details_id, style="dashed")
            if t.get("sous_taches"): visit(t["sous_taches"], tid)
    visit(data.get("taches", []), root_id)
    
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label=legend_labels["title"], color=theme["fontcolor"], fontcolor=theme["fontcolor"], fontsize='16')
        c.attr('node', shape='box', style='filled')
        c.node('legend_haute', legend_labels["haute"], fillcolor=colors["haute"], fontcolor=get_contrasting_font_color(colors["haute"]))
        c.node('legend_moyenne', legend_labels["moyenne"], fillcolor=colors["moyenne"], fontcolor=get_contrasting_font_color(colors["moyenne"]))
        c.node('legend_basse', legend_labels["basse"], fillcolor=colors["basse"], fontcolor=get_contrasting_font_color(colors["basse"]))
        c.edge('legend_haute', 'legend_moyenne', style='invis')
        c.edge('legend_moyenne', 'legend_basse', style='invis')
    
    try:
        dot.render(outfile_base, cleanup=True, view=False)
    except graphviz.ExecutableNotFound as exc:
        raise RuntimeError("Graphviz 'dot' not found") from exc


class MindmapResponse(BaseModel):
    data: Dict[str, Any]
    image_url: str
    format: str
    detected_language: str

app = FastAPI(title="AI Mind-Map Backend")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.post("/mindmap", response_model=MindmapResponse)
async def mindmap_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="PNG or JPEG sketch/image"),
    layout: int = Form(1, description="Layout: 1=Top-Down, 2=Circle, 3=Left-Right"),
    output_format: str = Form("png", description="Format de sortie: png, svg, pdf"),
    theme: str = Form("default", description="Thème de couleurs: default, dark, corporate"),
    arrow_style: str = Form("normal", description="Style des flèches: normal, odot, vee, diamond")
):
    if image.content_type not in ("image/png", "image/jpeg"): raise HTTPException(400, "Only PNG or JPEG accepted")
    if output_format not in ("png", "svg", "pdf"): raise HTTPException(400, "Unsupported output format")

    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{Path(image.filename or '').suffix}"
    try:
        with open(tmp_path, "wb") as f: f.write(await image.read())
    except IOError as e:
        raise HTTPException(500, "Could not save uploaded file.") from e

    try:
        map_data, lang_code = await asyncio.get_running_loop().run_in_executor(None, analyse_image, str(tmp_path))
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}") from exc
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: log.warning(f"Failed to clean up temp file: {tmp_path}")

    out_base = STATIC_DIR / str(uuid.uuid4())
    background_tasks.add_task(_render_graph, map_data, lang_code, layout, output_format, theme, arrow_style, out_base)
    image_url = f"/static/{out_base.name}.{output_format}"

    try:
        return MindmapResponse(data=map_data, image_url=image_url, format=output_format, detected_language=lang_code)
    except Exception as e:
        log.exception("CRITICAL: Pydantic validation failed. Data causing issue:")
        log.error(pprint.pformat(map_data))
        raise HTTPException(status_code=500, detail=f"Internal error validating response: {e}")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(f"{Path(__file__).stem}:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True, log_level="debug")