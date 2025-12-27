import os
import json
import sympy as sp
import requests
from datetime import datetime
from collections import defaultdict, Counter

# ==============================================================================
# SUPERVISEUR AGENTIQUE — MÉTA-ANALYSE DES DÉCOUVERTES DE L'HYDRE OMÉGA
# Version 26 Décembre 2025 — Intelligence collective et synthèse créative
# ==============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# Dossiers sources
THEMATIC_BASE = "discoveries_thematic"
LOGS_DIR = "omega_agents_logs"

# Rapport final
REPORT_PATH = f"omega_agents_reports/Rapport_Superviseur_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

# Mots-clés pour jugement LLM
positive_keywords = [
    "plausible", "intéressant", "cohérent", "prometteur", "profond",
    "fascinant", "étonnant", "innovant", "pertinent", "novateur",
    "convaincant", "solide", "élégant", "audacieux", "original",
    "remarquable", "excellent", "brillant", "80", "85", "90", "95", "100"
]

def consult_llm(prompt):
    """Envoie un prompt à Llama3 et retourne la réponse"""
    try:
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json().get('response', "Pas de réponse").strip()
        else:
            return f"Erreur HTTP {response.status_code}"
    except Exception as e:
        return f"Erreur LLM : {str(e)}"

def is_positive(analysis):
    return any(word in analysis.lower() for word in positive_keywords)

# === CHARGEMENT DE TOUTES LES DÉCOUVERTES ===
print("=== CHARGEMENT DES DÉCOUVERTES DE L'HYDRE OMÉGA ===\n")

discoveries = []

# 1. Depuis les dossiers thématiques (archive finale)
for root, _, files in os.walk(THEMATIC_BASE):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                discoveries.append(data)

# 2. Depuis omega_agents_logs (nouvelles non classées)
for file in os.listdir(LOGS_DIR):
    if file.startswith("discovery_p") and file.endswith(".json"):
        path = os.path.join(LOGS_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            discoveries.append(data)

print(f"{len(discoveries)} découvertes chargées (thématiques + logs).\n")

# === GROUPEMENT PAR SIMILARITÉ (clustering intelligent) ===
print("=== CLUSTERING PAR SIMILARITÉ SYMBOLIQUE ===\n")

clusters = defaultdict(list)

for d in discoveries:
    try:
        simplified = sp.simplify(sp.sympify(d['formula']))
        key = str(simplified)
    except:
        key = d['formula']  # fallback

    clusters[key].append(d)

# Garder seulement les clusters avec au moins 2 formules (intérêt pour comparaison)
multi_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

print(f"{len(multi_clusters)} clusters de formules similaires détectés (≥2 occurrences).\n")

# === ANALYSE SUPERVISEUR : COMPARAISON ET SYNTHÈSE ===
print("=== MÉTA-ANALYSE SUPERVISEUR — COMPARAISON DES AGENTS ===\n")

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write("# RAPPORT DU SUPERVISEUR AGENTIQUE\n\n")
    report.write(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
    report.write(f"Découvertes analysées : {len(discoveries)}\n")
    report.write(f"Clusters multiples détectés : {len(multi_clusters)}\n\n")
    report.write("## Clusters de convergence et désaccord entre agents\n\n")

    for i, (key, cluster) in enumerate(multi_clusters.items(), 1):
        report.write(f"### Cluster #{i} — {len(cluster)} variantes similaires\n\n")
        report.write(f"Forme simplifiée dominante :\n\n${sp.latex(sp.sympify(key))}$\n\n")

        agents = Counter(d['agent_id'] for d in cluster)
        inspirations = set(d['inspiration'] for d in cluster)

        report.write(f"**Agents impliqués** : {', '.join(agents.keys())} (fréquences : {dict(agents)})\n")
        report.write(f"**Inspirations** : {', '.join(inspirations)}\n\n")

        report.write("**Variantes proposées par les agents** :\n\n")
        for j, d in enumerate(cluster, 1):
            report.write(f"- **{d['agent_id']}** (pulse {d['pulse']}) : ${d['latex']}$\n")
            if 'true_analysis' in d:
                report.write(f"  → Critique : {d['true_analysis'][:200]}...\n")
            report.write("\n")

        # === SYNTHÈSE CRÉATIVE PAR LE SUPERVISEUR ===
        synthesis_prompt = f"""
Tu es le Superviseur Agentique, un méta-théoricien qui analyse les désaccords entre plusieurs agents évolutifs.

Ils ont produit {len(cluster)} variantes proches d'une même forme physique :

Forme dominante simplifiée : {key}

Voici les variantes :

"""
        for j, d in enumerate(cluster, 1):
            synthesis_prompt += f"\nVariante {j} (agent {d['agent_id']}) :\n{d['formula']}\nInspiration : {d['inspiration']}\n"

        synthesis_prompt += f"""
Ta mission :
1. Identifier les points de convergence et de divergence entre ces variantes.
2. Proposer une synthèse hybride optimale qui combine le meilleur de chaque variante.
3. Donner une plausibilité sur 100 et un potentiel comme avancée théorique.

Sois audacieux, créatif, et sans complaisance.
Réponds en français, structuré et technique.
"""

        print(f"Superviseur analyse le cluster #{i}...")
        synthesis = consult_llm(synthesis_prompt)

        report.write("## Synthèse du Superviseur\n\n")
        report.write(f"{synthesis}\n\n")
        report.write("---\n\n")

        if is_positive(synthesis):
            print(f"🌟 SYNTHÈSE POSITIVE POUR LE CLUSTER #{i} 🌟\n")
        else:
            print(f"⚖️ Synthèse réservée pour le cluster #{i}\n")

        synthesis_prompt += f"""
        
Enfin, en tant que Superviseur, donne 3 recommandations concrètes pour faire évoluer l’Hydre Oméga :
1. Nouveaux types de termes ou hybridations à explorer en priorité
2. Inspirations ou thèmes à privilégier
3. Mutations ou mécanismes à renforcer ou éviter
        
        Sois précis et opérationnel.
        """

# === GÉNÉRATION DE DIRECTIVES ÉVOLUTIVES AVANCÉES POUR L'HYDRE ===
    directives = []

    for i, (key, cluster) in enumerate(multi_clusters.items(), 1):
        # Extraire les inspirations dominantes
        inspirations = Counter(d['inspiration'] for d in cluster).most_common(3)
        insp_list = [insp for insp, count in inspirations]

        # Extraire les termes communs (approximation simple)
        terms = set()
        for d in cluster:
            formula = d['formula']
            if "log" in formula:
                terms.add("log_holographic")
            if "sin" in formula:
                terms.add("periodic_sin")
            if "phi**8" in formula or "phi**6" in formula:
                terms.add("high_order")
            if "G * phi" in formula:
                terms.add("gravitational")

        directives.append({
            "cluster_id": i,
            "dominant_form": key,
            "inspirations": insp_list,
            "common_terms": list(terms),
            "recommendations": [
                f"Privilégier les inspirations : {', '.join(insp_list[:2])}",
                f"Explorer davantage les termes : {', '.join(list(terms)[:3])}",
                "Augmenter les hybridations entre log et périodique",
                "Tester des termes d'ordre supérieur (ϕ⁸, ϕ⁶) avec suppression Planck"
            ]
        })

    # Export JSON pour l'Hydre
    directives_file = "hydra_evolution_directives.json"
    with open(directives_file, "w", encoding="utf-8") as f:
        json.dump(directives, f, indent=4, ensure_ascii=False)

    print(f"\nDirectives évolutives avancées exportées dans {directives_file}")
    print(f"{len(directives)} clusters analysés → recommandations prêtes pour l'Hydre")
print(f"\n[TERMINE] Rapport du Superviseur généré → {REPORT_PATH}")
print("Le Superviseur a comparé les agents, identifié leurs désaccords,")
print("et proposé des synthèses créatives.")
print("Ouvre le rapport Markdown pour découvrir les idées émergentes.")
