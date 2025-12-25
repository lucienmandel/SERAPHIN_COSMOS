import os
import json
import requests
from datetime import datetime

# === CONFIGURATION ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# === MOTS-CLÉS POSITIFS (même liste que le script principal) ===
positive_keywords = [
    "plausible", "intéressant", "cohérent", "prometteur", "profond",
    "fascinant", "étonnant", "innovant", "pertinent", "novateur",
    "convaincant", "solide", "élégant", "audacieux", "original",
    "remarquable", "excellent", "brillant", "stimulant",
    "80", "85", "90", "95", "100", "très bon", "bon potentiel"
]

def is_positive(analysis):
    """Retourne True si l'analyse contient un mot-clé positif"""
    return any(word in analysis.lower() for word in positive_keywords)

def consult_llm(formula_str, context_tag):
    prompt = f"""
    En tant que physicien théoricien de très haut niveau en 2025, analyse cette conjecture issue d'une évolution agentique autonome :

    Formule : {formula_str}

    Inspiration : {context_tag}

    1. Ressemblance avec un concept connu ou émergent ?
    2. Cohérence théorique, élégance et audace ?
    3. Plausibilité sur 100 ?
    4. Potentiel comme avancée partielle sur un problème unsolved ?

    Réponds en français, précis, technique, sans complaisance, et avec une note chiffrée.
    """
    try:
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json().get('response', "Pas de réponse").strip()
        else:
            return f"Erreur HTTP {response.status_code}"
    except Exception as e:
        return f"Erreur LLM : {str(e)}"

# === CHARGEMENT DES DÉCOUVERTES ===
discoveries = []
log_dir = "omega_agents_logs"
for filename in os.listdir(log_dir):
    if filename.startswith("discovery_p") and filename.endswith(".json"):
        with open(os.path.join(log_dir, filename), "r", encoding='utf-8') as f:
            data = json.load(f)
            if "différée" in data.get("analysis", ""):  # seulement celles non analysées
                discoveries.append(data)

# Tri par score intrinsèque (SymPy)
discoveries.sort(key=lambda x: x['score'], reverse=True)

print(f"\n=== ANALYSE CRITIQUE DES {len(discoveries)} MEILLEURES CONJECTURES ===")
print(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

N = min(100, len(discoveries))  # nombre à analyser (change si tu veux plus/moins)
for i, d in enumerate(discoveries[:N], 1):
    score_sympy = d['score']
    family = d.get('family', 'Unknown')
    subfamily = d.get('subfamily', 'Unknown')

    print(f"#{i}/{N} | Score SymPy : {score_sympy} | Famille : {family}/{subfamily}")
    print(f"Agent : {d['agent_id']} | Pulse {d['pulse']} | Profondeur {d['depth']}")
    print(f"Inspiration : {d['inspiration']}")
    print(f"Formule : {d['formula']}")
    print(f"LaTeX : ${d['latex']}$\n")

    print("Critique théorique en cours...\n")
    true_analysis = consult_llm(d['formula'], d['inspiration'])
    print(f"{true_analysis}\n")

    # Évaluation automatique
    if is_positive(true_analysis):
        print("🌟 JUGEMENT POSITIF DU CRITIQUE THÉORIQUE 🌟\n")
    else:
        print("⚖️ Jugement neutre ou réservé.\n")

    print("=" * 120)

print(f"\nAnalyse terminée. {N} conjectures évaluées par le physicien théoricien.")
print("Les plus prometteuses (avec jugement positif) méritent une étude approfondie.")
