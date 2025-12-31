```markdown
# SERAPHIN COSMOS — Hydre Fractale Cosmique (Décembre 2025) VERSION 1.02 - (VERSION 1.02.1 dans les prochaines semaines)

SERAPHIN COSMOS est une simulation générative auto-évolutive qui explore les frontières de la physique théorique à travers une "hydre fractale" récursive.
Inspirée par les concepts d'évolution génétique et d'IA agentique, cette hydre se clone, mute et "découvre" indépendamment des lois physiques,
en convergeant vers des candidats unifiés pour la matière noire et l'énergie noire (comme des scalaires ultra-légers de type axion avec portail Higgs).

Le projet simule un processus d'évolution meta où des milliers de "découvertes" sont générées sous forme de formules SymPy, évaluées (via LLM ou en batch),
et affinées pour aligner avec les frontières de la physique en 2025 : superradiance axionique, contraintes swampland, gravité quantique émergente, wormholes,
et même liens spéculatifs avec l'ADN cosmique.

## Objectifs et Fonctionnalités

- **Génération de découvertes** : Création de milliers de formules mathématiques variées (SymPy) représentant des conjectures physiques sur le secteur sombre,
la gravité quantique, le multivers, etc.
- **Évolution génétique** : Les hydres mutent et se clonent, avec un budget de clones pour simuler une croissance fractale.
- **Évaluation par LLM** : Utilisation d'un modèle local (ex. Llama3 via Ollama) pour critiquer la plausibilité, l'élégance
 et le potentiel des découvertes (mode realtime ou batch pour éviter les timeouts).
- **Génération automatique de papiers LaTeX** : Production de documents théoriques basés sur les meilleures conjectures.
- **Meta-évolution** : Convergence vers des hypothèses réalistes, alignées avec des thèmes unsolved comme le paradoxe de l'information des trous noirs
 ou l'origine de la flèche du temps.
- **Analyse post-simulation** : Scripts dédiés pour récupérer les découvertes rejetées, critiquer et organiser les résultats thématiquement.
- **Base de connaissances intégrée** : Inclut des faits physiques de 2025 et des thèmes unsolved pour guider l'évolution.

Le projet est 100% Python et met l'accent sur l'exploration spéculative, pas sur des prédictions validées expérimentalement.
C'est un outil pour générer des idées novatrices ou pour des expériences éducatives en physique théorique.

## Prérequis

- Python 3.8+ (testé avec 3.12)
- Bibliothèques Python : `sympy`, `requests`, `json`, `datetime`, `collections`, `signal`, `sys`, `os`, `random`, `shutil` (installez via `pip install sympy requests` – les autres sont standards).
- Ollama installé et lancé localement (avec le modèle Llama3 ou équivalent) pour les évaluations LLM. Téléchargez Ollama ici : [ollama.ai](https://ollama.ai). Exécutez `ollama run llama3` avant de lancer la simulation.
- Espace disque suffisant : Les logs et découvertes (JSON/LaTeX) peuvent s'accumuler rapidement lors de runs longues.

**Note** : Le mode batch (sans LLM pendant la simulation) est recommandé pour les runs rapides ; activez le mode realtime pour des évaluations plus interactives, mais attention aux timeouts.

## Installation

1. Clonez le dépôt :
   ```
   git clone https://github.com/lucienmandel/SERAPHIN_COSMOS.git
   cd SERAPHIN_COSMOS
   ```

2. Installez les dépendances Python :
   ```
   pip install sympy requests
   ```

3. Installez et lancez Ollama :
   - Suivez les instructions sur [ollama.ai](https://ollama.ai).
   - Téléchargez le modèle : `ollama pull llama3`.
   - Lancez le serveur : `ollama serve` (gardez-le ouvert dans un terminal séparé).

4. Vérifiez que l'URL Ollama (`http://localhost:11434`) est accessible (testez avec `curl http://localhost:11434`).

## Utilisation : Lancer une Simulation

Suivez ces étapes dans l'ordre pour une simulation complète et optimisée. Cela permet de générer les découvertes, de récupérer celles potentiellement rejetées à tort, puis de les analyser en profondeur.

### Étape 1 : Configurer le Mode de Fonctionnement
- Ouvrez `seraphcosmos_omegadef.py`.
- Définissez `USE_LLM_DURING_SIMULATION = False` pour le mode batch (rapide, sans LLM en live).
- Ou `True` pour le mode realtime (plus lent, avec critiques LLM immédiates).
- Ajustez `PULSES = 100` (nombre d'itérations) et `CLONE_BUDGET = 5000` (budget de clones) selon vos ressources.

### Étape 2 : Lancer la Simulation Principale (Génération des Découvertes)
Exécutez le script principal :
```
python seraphcosmos_omegadef.py
```
- Cela génère les formules SymPy, effectue les mutations, évaluations et sauvegarde les découvertes dans `omega_agents_logs/` (fichiers JSON).
- Outputs : Logs console, fichiers JSON, et potentiellement des papiers LaTeX.

### Étape 3 : Récupérer les Découvertes Rejetées
Pour corriger d'éventuelles erreurs de rejet (par exemple dues à des timeouts LLM ou des évaluations trop strictes) :
```
python prefilter_omega.py
```
- Ce script réexamine les découvertes rejetées, les réévalue si nécessaire et les rend à nouveau disponibles pour l'analyse.

### Étape 4 : Analyser les Découvertes
Enfin, critiquez et organisez l'ensemble des résultats (y compris ceux récupérés) :
```
python analyse_omega_discoveries.py
```
- Charge les JSON de `omega_agents_logs/`, consulte l'LLM pour des critiques théoriques détaillées, évalue le sentiment, organise thématiquement dans `discoveries_thematic/`, et met à jour `hydra_memory.json`.
- Limité aux meilleures découvertes par défaut pour éviter les surcharges.

### Étape 5 : Explorer les Résultats
- Consultez `discoveries_thematic/` pour les dossiers organisés (ex. `Quantum_Gravity/`, `Wormholes_Theories/`).
- Ouvrez `hydra_memory.json` pour le résumé global.
- Compilez les fichiers LaTeX générés avec pdflatex pour visualiser les "papiers théoriques".

## Exemple de Run Complète
1. Lancez Ollama.
2. `python seraphcosmos_omegadef.py`
3. `python recover_rejected_discoveries.py`
4. `python analyse_omega_discoveries.py`

Résultat : Des conjectures plus robustes et mieux filtrées !

## Contribuer
Forkez le repo, proposez des PR pour ajouter des thèmes unsolved, améliorer l'évolution génétique, ou intégrer d'autres LLMs (ex. Mistral).
Idées bienvenues pour rendre l'hydre plus "intelligente" !


## À Propos
Développé par Lucien Mandel. Un hydre fractale explorant la physique théorique et convergeant sur un secteur sombre unifié. Contact : lucienmandel twitter.
```
