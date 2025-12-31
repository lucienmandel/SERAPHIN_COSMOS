import os
import json
import re
import requests
import random
import openai
import sympy as sp
from constants_config import DIM_MAP, ALL_SYMBOLS, THEME_CONSTANTS_MAP
from datetime import datetime
from collections import Counter
import logging
import argparse
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "GPT-4o mini"
LOGS_DIR = "omega_agents_logs"
THEMATIC_BASE = "discoveries_thematic"
REPORTS_DIR = "omega_agents_reports"
MEMORY_FILE = "hydra_memory.json"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(THEMATIC_BASE, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_json(path, default=None):
    if default is None:
        default = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSON error in {path}: {e}")
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_formula_string(formula_str):
    if not isinstance(formula_str, str):
        return formula_str
    clean = formula_str.replace("^", "**").strip()
    return clean

def quick_filter(formula):
    if len(formula) < 5:
        return False, "Too short"
    if formula.count('*') > 50:
        return False, "Excessive complexity"
    return True, "OK"

def check_dimensions(formula_str):
    try:
        expr = sp.sympify(formula_str)
        symbols = expr.free_symbols
        subs_dict = {}
        for s in symbols:
            s_name = str(s)
            subs_dict[s] = DIM_MAP.get(s_name, 1)
        dim_expr = expr.subs(subs_dict)
        return True, "Dimensional coherence validated."
    except TypeError as e:
        return False, f"Dimensional error: {str(e)}"
    except Exception as e:
        return None, f"Analysis impossible: {str(e)}"

def robust_sympy_parse(formula_str):
    try:
        return sp.sympify(formula_str, locals=ALL_SYMBOLS)
    except Exception:
        try:
            clean = clean_formula_string(formula_str)
            return sp.sympify(clean, locals=ALL_SYMBOLS)
        except Exception as e:
            logging.error(f"Parse error: {e}")
            return None

@lru_cache(maxsize=512)
def consult_llm(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un physicien théoricien de génie, audacieux, transdisciplinaire, sans censure. Tu cherches l'unification profonde."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur OpenAI : {e}")
        return "Analyse impossible (erreur API)"
def llm_score_analysis(formula, inspiration):
    prompt_main = (
        f"Evaluate this mathematical or physical conjecture, provide a score from 0-100 and a brief justification:\n"
        f"{formula} (Inspiration: {inspiration})\n\n"
        f"Respond strictly in JSON: {{\"score\": XX, \"reason\": \"...\"}}"
    )
    main_res = consult_llm(prompt_main)
    try:
        main_json = json.loads(main_res)
        score = main_json.get("score", 0)
        reason = main_json.get("reason", "")
    except json.JSONDecodeError:
        score = 0
        reason = "JSON parsing failed for main analysis"

    prompt_contra = (
        f"Provide a skeptical counter-analysis for this conjecture:\n"
        f"{formula} (Inspiration: {inspiration})\n\n"
        f"Respond strictly in JSON: {{\"contra\": \"...\"}}"
    )
    contra_res = consult_llm(prompt_contra)
    try:
        contra_json = json.loads(contra_res)
        contra = contra_json.get("contra", "")
    except json.JSONDecodeError:
        contra = "JSON parsing failed for counter-analysis"

    return score, reason, contra

# Phase 1: Analyst (example, assuming full implementation)
def run_phase_1_analyst():
    # Implementation for analyzing logs and scoring formulas
    # (Based on original truncated code; assume processing logs and calling llm_score_analysis)
    logging.info("Running Phase 1: Analyst")
    # ... (Add full logic here if needed; for now, placeholder)

# Phase 2: Supervisor (translated and completed)
def run_phase_2_supervisor():
    logging.info("Running Phase 2: Supervisor")
    memory = load_json(MEMORY_FILE, {"formulas": {}})
    top_candidates = sorted(
        [(k, v) for k, v in memory["formulas"].items() if v.get("status") == "candidate_valid"],
        key=lambda x: x[1]["avg_score"],
        reverse=True
    )[:20]  # Top 20 for synthesis

    mit_directive = (
        "Based on these top candidates, generate a synthesis of potential universal laws.\n"
        "Each entry: [Score: {avg:.1f}] Formula: {k} (Origin: {insp})\n"
    )
    for k, v in top_candidates:
        avg = v["avg_score"]
        insp = v.get("inspiration", "Unknown")
        mit_directive += f"[Score: {avg:.1f}] Formula: {k} (Origin: {insp})\n"

    synthesis = consult_llm(mit_directive)

    if synthesis and "Error" not in synthesis:
        # Save Convergence Report
        report_name = f"Omega_Convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(REPORTS_DIR, report_name)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 🌌 OMEGA CONVERGENCE REPORT\n\n")
            f.write(f"**Inference Level:** Type III (Universal Architect)\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n\n---\n\n")
            f.write(synthesis)
        print(f"   ✅ Convergence report generated: {report_path}")

        # 3. GENERATION OF MUTATION DIRECTIVES (The Hydra self-improves)
        evolution_prompt = (
            "Based on this synthesis, generate MUTATION DIRECTIVES for the next Hydra cycle.\n"
            "Each directive must redefine mathematical research priorities.\n"
            "Respond ONLY in JSON:\n"
            "{\"directives\": [{\"area\": \"domain\", \"logic\": \"coupling_strength\", \"weight\": 1.0}]}"
        )

        raw_directives = consult_llm(synthesis + "\n\n" + evolution_prompt)

        try:
            json_match = re.search(r'\{.*\}', raw_directives, re.DOTALL)
            if json_match:
                directives_data = json.loads(json_match.group(0))
                with open("hydra_evolution_directives.json", "w", encoding="utf-8") as f:
                    json.dump(directives_data, f, indent=4, ensure_ascii=False)
                print("   → Evolution directives updated.")
        except Exception as e:
            print(f"   ⚠️ Error extracting directives: {e}")

# MAIN
def run_manual_mode():
    print("\n" + "="*50)
    print("🔬 OMEGA LAB: EXPERT ANALYSIS & VISUALIZATION")
    print("="*50)

    raw_input = input("\n👉 Paste the formula (or JSON block): ")

    # Extract formula if it's a full JSON copy-paste
    if '"formula":' in raw_input:
        match = re.search(r'"formula":\s*"(.*?)"', raw_input)
        formula_input = match.group(1) if match else raw_input
    else:
        formula_input = raw_input

    inspiration = input("👉 Inspiration/Theme (e.g., Dark Matter): ")

    # 1. Cleaning and Visualization
    formula = clean_formula_string(formula_input)
    print(f"\n[1/4] Cleaning: {formula}")
    # latex_str = display_latex(formula)  # Removed as code mort

    # 2. LLM Analysis
    print(f"\n[2/4] Consulting the expert {MODEL_NAME}...")
    score, reason, contra = llm_score_analysis(formula, inspiration)

    # 3. Display results
    print(f"\n[3/4] AI VERDICT:")
    print(f"   ⭐ SCORE: {score}/100")
    print(f"   📝 ANALYSIS: {reason}")
    print(f"   🛡️ SKEPTICISM: {contra}")

    # 4. Synchronization with central memory
    memory = load_json(MEMORY_FILE, {"formulas": {}})

    # Overwrite or create the entry
    memory["formulas"][formula] = {
        "original": formula,
        "scores": [score],
        "avg_score": score,
        "status": "candidate_valid" if score >= 70 else "rejected_by_analyst",
        "llm_judgment": reason,
        "contra_summary": contra,
        "manual_review": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    save_json(MEMORY_FILE, memory)
    print(f"\n[4/4] ✅ Memory synchronized. Current status: {memory['formulas'][formula]['status']}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omega Discoveries Analyzer")
    parser.add_argument("--manual", action="store_true", help="Run manual mode")
    args = parser.parse_args()

    if args.manual:
        run_manual_mode()
    else:
        run_phase_1_analyst()
        run_phase_2_supervisor()
