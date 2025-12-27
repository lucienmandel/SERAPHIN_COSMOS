import os
import json
import requests
import shutil  # pour déplacer les fichiers
from datetime import datetime

# === CONFIGURATION ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

positive_keywords = [
    "plausible", "intéressant", "cohérent", "prometteur", "profond",
    "fascinant", "étonnant", "innovant", "pertinent", "novateur",
    "convaincant", "solide", "élégant", "audacieux", "original",
    "remarquable", "excellent", "brillant", "stimulant",
    "80", "85", "90", "95", "100", "très bon", "bon potentiel"
]

def is_positive(analysis):
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

# === CHARGEMENT DES DÉCOUVERTES À ANALYSER (depuis omega_agents_logs) ===
discoveries = []
log_dir = "omega_agents_logs"

for filename in os.listdir(log_dir):
    if filename.startswith("discovery_p") and filename.endswith(".json"):
        file_path = os.path.join(log_dir, filename)
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            # On analyse seulement celles pas encore critiquées
            if "différée" in data.get("analysis", "") and 'true_analysis' not in data:
                discoveries.append((file_path, data))

# Tri par score
discoveries.sort(key=lambda x: x[1]['score'], reverse=True)

print(f"\n=== ANALYSE CRITIQUE DES {len(discoveries)} NOUVELLES CONJECTURES ===")
print(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

N = min(100, len(discoveries))
moved_count = 0

for i, (source_path, d) in enumerate(discoveries[:N], 1):
    score = d['score']
    family = d.get('family', 'Uncategorized')
    subfamily = d.get('subfamily', 'General')

    print(f"#{i}/{N} | Score : {score} | Famille : {family}/{subfamily}")
    print(f"Formule : {d['formula']}")
    print(f"LaTeX : ${d['latex']}$")
    print(f"Inspiration : {d['inspiration']}\n")

    print("Critique théorique en cours...\n")
    true_analysis = consult_llm(d['formula'], d['inspiration'])
    print(f"{true_analysis}\n")

    if is_positive(true_analysis):
        print("🌟 JUGEMENT POSITIF DU CRITIQUE THÉORIQUE 🌟\n")
    else:
        print("⚖️ Jugement neutre ou réservé.\n")

    # === SAUVEGARDE DE L'ANALYSE ===
    d['true_analysis'] = true_analysis
    d['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === DÉPLACEMENT VERS LE DOSSIER THÉMATIQUE DÉFINITIF ===
    thematic_dir = f"discoveries_thematic/{family}/{subfamily}"
    os.makedirs(thematic_dir, exist_ok=True)
    final_path = os.path.join(thematic_dir, os.path.basename(source_path))

    with open(final_path, "w", encoding='utf-8') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

    # Suppression de la copie dans omega_agents_logs
    os.remove(source_path)

    print(f"→ Déplacée définitivement vers {family}/{subfamily}/ et supprimée du dossier temporaire\n")
    moved_count += 1

    print("=" * 120)

    # === MISE À JOUR DE LA MÉMOIRE CENTRALE DE L'HYDRE ===
    MEMORY_FILE = "hydra_memory.json"

    # Charge ou crée la mémoire
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    else:
        memory = {"formulas": {}}

    # Met à jour avec les formules analysées dans ce run
    for i, (source_path, d) in enumerate(discoveries[:N], 1):
        try:
            simplified = sp.simplify(sp.sympify(d['formula']))
            key = str(simplified)
        except:
            key = d['formula']

        status = "validated" if is_positive(d.get('true_analysis', '')) else "rejected"

        if key not in memory["formulas"]:
            memory["formulas"][key] = {
                "original": d['formula'],
                "simplified": key,
                "count": 1,
                "status": status,
                "last_seen": datetime.now().strftime("%Y-%m-%d"),
                "family": d.get('family', 'Unknown'),
                "subfamily": d.get('subfamily', 'Unknown'),
                "llm_judgment": d.get('true_analysis', 'Non analysée')
            }
        else:
            memory["formulas"][key]["count"] += 1
            memory["formulas"][key]["status"] = status
            memory["formulas"][key]["last_seen"] = datetime.now().strftime("%Y-%m-%d")
            if 'true_analysis' in d:
                memory["formulas"][key]["llm_judgment"] = d['true_analysis']

    # Sauvegarde
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4, ensure_ascii=False)

    print(f"\nMémoire centrale mise à jour : {len(memory['formulas'])} formules uniques connues.")

print(f"\nAnalyse terminée : {moved_count} conjectures analysées et archivées définitivement.")
print("omega_agents_logs est maintenant nettoyé — prêt pour le prochain run !")
print("Toutes les formules critiquées sont dans discoveries_thematic/ (organisées par famille).")
