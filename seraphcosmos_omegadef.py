import os
import random
import requests
import json
import sympy as sp
from datetime import datetime
from collections import defaultdict
import signal
import sys

# ==============================================================================
# SÉRAPHIN : PROTOCOLE AGENTIC OMEGA — VERSION COMPLÈTE (24 Décembre 2025)
# Avec base de connaissances + catégories + 3 nouvelles fonctionnalités
# ==============================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # ou mistral, gemma2, etc.

PULSES = 100
CLONE_BUDGET = 50000000

# ==============================================================================
# MODE DE FONCTIONNEMENT
# ==============================================================================
# False = Mode batch : génération rapide sans LLM pendant la simulation (recommandé pour longues runs)
# True  = Mode realtime : consultation LLM à chaque découverte valide (plus lent, risque timeout)
USE_LLM_DURING_SIMULATION = False

# ==============================================================================
# MOTS-CLÉS POSITIFS POUR L'ÉVALUATION LLM (récompense forte si présent)
# ==============================================================================
positive_keywords = [
    "plausible",
    "intéressant",
    "cohérent",
    "prometteur",
    "profond",
    "fascinant",
    "étonnant",
    "innovant",
    "pertinent",
    "novateur",
    "convaincant",
    "solide",
    "élégant",
    "audacieux",
    "original",
    "remarquable",
    "excellent",
    "brillant",
    "stimulant",
    "80",
    "85",
    "90",
    "95",
    "100",
    "très bon",
    "bon potentiel",
]

# ==============================================================================
# BASE DE CONNAISSANCES (Décembre 2025) — Complète comme demandé
# ==============================================================================
REAL_KNOWLEDGE_2025 = [
    "Ultra-light scalar fields (m ~ 10^{-22} to 10^{-20} eV) as fuzzy dark matter candidates, explaining galaxy cores.",
    "Axion-like particles (ALPs) with periodic potentials, candidates for dark matter and dark energy.",
    "Relaxion mechanism: scalar relaxing Higgs mass during inflation.",
    "Higgs portal couplings g φ H² for detection via Higgs decays.",
    "Quintessence with time-varying equation of state w(z).",
    "Swampland constraints favoring dynamical dark energy.",
    "Superradiance axionique autour de trous noirs en rotation : amplification de champs scalaires ultra-légers menant à des nuages axioniques observables via ondes gravitationnelles (LIGO/Virgo 2025).",
    "Cheveux scalaires sur trous noirs : violations de la no-hair theorem pour scalaires exotiques, prédisant des signatures dans les images EHT de Sgr A* et M87*.",
    "Interactions scalaires-trous noirs supermassifs : rôle des champs ultra-légers dans la stabilisation des jets relativistes et l'accrétion, expliquant les anomalies observées par JWST dans les galaxies primitives.",
    "Axions comme médiateurs de la gravité quantique : contraintes de la swampland distance conjecture appliquées aux trous noirs primordiaux, favorisant des masses ~10^{-10} eV.",
    "Quintessence sombre couplée à la matière noire : modèles avec w(z) oscillant, testés via DESI 2025, montrant une corrélation avec la formation de trous noirs binaires.",
    "Relaxion en cosmologie post-inflation : relaxation dynamique de la masse de Higgs influencée par des scalaires cachés, avec implications pour l'évaporation de trous noirs primordiaux.",
    "Portail Higgs vers secteurs scalaires sombres : détections potentielles de désintégrations invisibles au HL-LHC, liées à des signatures de trous noirs quantiques.",
    "Fractales en spacetime près des horizons de trous noirs : scalaires ultra-légers induisant une géométrie fractale, expliquant l'entropie holographique observée.",
    "Axions et asymétrie baryonique : rôle dans la génération d'antimatière près des trous noirs, avec contraintes de CMB-S4 sur les potentiels périodiques.",
    "Matière noire floue autour de trous noirs : solitons scalaires stables formant des cœurs galactiques, simulant des halos observés par Euclid 2025.",
    "Swampland et trous noirs extrémaux : contraintes sur les scalaires dynamiques favorisant une énergie noire accélérée, testée via ondes gravitationnelles.",
    "Ultra-light scalars comme quintessence holographique : potentiel V(φ) lié à l'entropie de trous noirs, prédisant des transitions de phase cosmiques tardives.",
    "Unification de la gravité quantique via boucles : avancées en LQG montrant une résolution des singularités des trous noirs en 2025.",
    "Gravité émergente de l'intrication quantique : holographie quantique confirmée par simulations qubit, reliant gravité à l'information quantique.",
    "Effets quantiques dans la gravité : corrections quantiques observées dans les ondes gravitationnelles de trous noirs binaires par LIGO advanced.",
    "Multivers inflationnaire : preuves indirectes via anomalies CMB-S4, suggérant bulles d'univers parallèles.",
    "Multivers quantique d'Everett : interprétations renforcées par expériences de décohérence quantique en 2025.",
    "Swampland et multivers : contraintes sur les paysages de cordes favorisant un multivers fini et testable.",
    "Trous de ver traversables : simulations holographiques via ordinateurs quantiques démontrant des tunnels stables en gravité quantique.",
    "Trous de ver entrelacés : liens entre intrication quantique et géométrie spacetime, testés via qubits en 2025.",
    "Trous de ver comme portails multiversels : hypothèses reliant wormholes à des transitions entre univers parallèles.",
    "ADN comme structure holographique : patterns fractals dans l'ADN suggérant un encodage d'informations cosmiques, inspiré de la gravité quantique.",
    "Microtubules et gravité quantique : théorie Orch-OR reliant conscience à des effets quantiques sensibles à la gravité dans l'ADN.",
    "ADN panspermique et multivers : hypothèses sur l'origine extraterrestre de l'ADN portant des 'mémoires' d'univers parallèles.",
    "ADN comme récepteur de trous de ver quantiques : spéculations sur des micro-wormholes magnétiques dans l'ADN pour transfert d'information interstellaire.",
    "Épigénétique et champs cosmiques : modifications ADN influencées par des scalaires exotiques, comme indices cachés dans le génome humain."
    "Asymptotic safety in quantum gravity → contraintes renforcées sur les couplages d'axions ultra-légers et matière noire vectorielle."
    "Nouvelles théories quantiques de la gravité compatibles avec le Modèle Standard, décrivant la gravité via des symétries de jauge unitaires 1D (Aalto University, mai 2025)."
    "Évidence renforcée pour une énergie noire affaiblie (dynamical dark energy) via DESI et Vera C. Rubin Observatory, favorisant des modèles quintessence ou axions ultra-légers oscillants."
    "Découverte d'un trou noir supermassif isolé dans l'univers primitif (JWST, 2025), challengeant les modèles de formation et impliquant une croissance via superradiance axionique."
    "Photons émergents et fractionalisation dans un liquide de spin quantique 3D vrai (quantum spin ice en Ce2Zr2O7, confirmant des prédictions théoriques décennales)."
    "Gravitinos chargés comme candidats viables à la matière noire dans des théories de supergravité."
    "Axions composites lourds (glueball axion-like particles) explorant de nouveaux domaines de couplage-masse via secteurs confinés sombres."
    "Superradiance quantique pleinement décrite autour de trous noirs de Kerr, avec quantification canonique de champs scalaires massifs."
    "Énergie noire précoce couplée (early dark energy) résolvant partiellement les tensions Hubble et σ8."
    "Cheveux quantiques courts sur trous noirs confirmés absents par ondes gravitationnelles (LIGO/Virgo/KAGRA), renforçant le no-hair theorem classique mais laissant place à des effets Planck-scale."
    "Wormholes évolutifs dans fonds cosmologiques FLRW/de Sitter, avec dynamiques influencées par champs scalaires et expansion."
    "Tunnel quantique macroscopique et quantification d'énergie dans circuits supraconducteurs (Nobel 2025 : Clarke, Devoret, Martinis), fondation des qubits supraconducteurs modernes.",
    "Flèche du temps et deuxième principe : entropie croissante comme origine de la direction temporelle.",
    "Temps émergent en gravité quantique : hypothèse que le temps n'est pas fondamental mais émerge de l'intrication (Wheeler-DeWitt, Page-Wootters).",
    "Temps discret en boucle quantique : structure granulaire du temps à l'échelle de Planck.",
    "Temps et holographie : rôle du temps radial en AdS/CFT et émergence dans les espaces de de Sitter.",
    "Problème du temps en gravité quantique : équation de Wheeler-DeWitt sans variable temps explicite.",
    "Temps et conscience : hypothèses que la perception du temps est liée à des processus quantiques dans le cerveau (Orch-OR étendu).",
    "Temps fractal ou multi-fractal : modèles où le temps a une dimension fractale liée à la géométrie quantique.",
    "Temps réversible vs irréversible : lien avec la perte d'information et le paradoxe des trous noirs.",
    "Temps dans le multivers : branchement temporel dans l'interprétation d'Everett.",
    "Temps et information : rôle de l'information quantique dans la définition de la flèche temporelle."
]

UNSOLVED_THEMES = [
    "Quantum gravity reconciliation",
    "Nature of dark matter",
    "Multiverse existence",
    "Fine-tuning constants",
    "Antimatter asymmetry",
    "Dark energy composition",
    "Time arrow direction",
    "Hidden sectors",
    "Supersymmetry",
    "Fractal spacetime",
    "Ultra-light scalars",
    "Axions",
    "Higgs portal",
    "Relaxion",
    "Black hole information paradox",
    "Wormhole traversability",
    "DNA cosmic encoding",
    "Quantum entanglement in gravity"
    "Réconciliation complète de la gravité quantique avec le Modèle Standard (théories compatibles émergentes mais non confirmées)."
    "Identité précise de la matière noire (WIMPs exclus, axions/fuzzy DM favorisés mais non détectés)."
    "Énergie noire dynamique vs constante cosmologique (evidence croissante pour w variant, mais mécanisme inconnu)."
    "Paradoxe de l'information des trous noirs dans un cadre quantique complet."
    "Traversabilité et stabilité des wormholes en gravité quantique."
    "Origine de l'asymétrie matière/antimatière et du problème strong CP."
    "Existence et testabilité du multivers (inflationnaire ou Everett)."
    "Hiérarchie des masses et fine-tuning des constantes (swampland constraints vs landscape string)."
    "Cheveux scalaires ou modifications quantiques violant le no-hair theorem."
    "Liens entre intrication quantique, gravité émergente et holographie (AdS/CFT étendu à dS)."
    "Rôle potentiel de champs ultra-légers dans la conscience ou structures biologiques (théories spéculatives Orch-OR mises à jour)."
    "Détection directe d'ondes gravitationnelles primordiales ou signatures de phase transitions cosmiques.",
    "Nature fondamentale du temps",
    "Origine de la flèche du temps",
    "Temps émergent ou fondamental ?",
    "Temps discret vs continu à l'échelle de Planck",
    "Problème du temps en gravité quantique",
    "Lien entre temps et conscience",
    "Temps fractal en physique",
    "Temps et paradoxe de l'information",
    "Temps dans les modèles multiversels"
]

DISCOVERY_CATEGORIES = {
    "quantum": "Quantum_Laws",
    "classical": "Classical_Physics",
    "multiverse": "Multiverse_Theories",
    "micro_macro_link": "Quantum_Classical_Bridges",
    "big_bang": "Cosmology_Origins",
    "new_matter": "Exotic_Matter_Discoveries",
    "math": "Mathematical_Laws",
    "meta_evolver": "Meta_Evolution",
    "gravity": "Gravity_Laws",
    "quantum_gravity": "Quantum_Gravity",
    "wormholes": "Wormholes_Theories",
    "dna_cosmic": "DNA_Cosmic_Links",
}

# Symboles
x, y, z, t, k = sp.symbols("x y z t k", real=True)
phi, hbar, delta, G, m = sp.symbols("phi hbar delta G m")
f_a, Lambda, M_pl, lambda_q = sp.symbols(
    "f_a Lambda M_pl lambda_q", positive=True, real=True
)
# f_a  : échelle de brisure de symétrie pour axions (decay constant)
# Lambda : échelle d'énergie (ex. énergie noire, vide QCD, etc.)
# M_pl : masse de Planck réduite ≈ 2.4 × 10^18 GeV

# ==============================================================================
# PATCH SYMPY : AJOUT DE LA FONCTION SIGMOID (SymPy n'en a pas nativement)
# ==============================================================================
def _sympy_sigmoid(expr):
    """
    Fonction sigmoïde pour SymPy : σ(x) = 1 / (1 + exp(-x))
    On l'attache à sp.sigmoid pour que toutes les écritures sp.sigmoid(...) fonctionnent.
    """
    return 1 / (1 + sp.exp(-expr))

# Surcharge sp.sigmoid avec notre fonction
sp.sigmoid = _sympy_sigmoid

# Alternative (équivalente mathématiquement, parfois plus stable)
# sp.sigmoid = lambda expr: (1 + sp.tanh(expr / 2)) / 2

# Compteur global des formules découvertes (pour éviter les duplicatas excessifs)
formula_counter = defaultdict(int)  # clé : str(simplified_formula), valeur : nombre de fois vue
MAX_DUPLICATES_ALLOWED = 1  # ajuste selon tes préférences (5 à 15 recommandé)

# ==============================================================================
# DIRECTIVES ÉVOLUTIVES DU SUPERVISEUR (chargées dynamiquement)
# ==============================================================================
evolution_directives = []

def load_evolution_directives():
    global evolution_directives
    directives_file = "hydra_evolution_directives.json"
    if os.path.exists(directives_file):
        with open(directives_file, "r", encoding="utf-8") as f:
            evolution_directives = json.load(f)
        print(f"[ÉVOLUTION AVANCÉE] {len(evolution_directives)} directives du Superviseur chargées")
    else:
        print("[ÉVOLUTION AVANCÉE] Aucune directive trouvée — mode exploration libre")

# ==============================================================================
# ARRÊT PROPRE Ctrl+C
# ==============================================================================
import signal
import sys

# ==============================================================================
# ARRÊT PROPRE AVEC Ctrl+C — SAUVEGARDE DES DÉCOUVERTES EN COURS
# ==============================================================================
global_discoveries = []  # doit être global pour être accessible partout

def graceful_shutdown(sig, frame):
    print("\n" + "="*80)
    print("ARRÊT MANUEL DÉTECTÉ (Ctrl+C)")
    print("Sauvegarde des découvertes en cours avant sortie...")
    print("="*80)

    if global_discoveries:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_path = f"omega_agents_reports/Rapport_Interrompu_{timestamp}.md"
        os.makedirs("omega_agents_reports", exist_ok=True)

        with open(emergency_path, "w", encoding="utf-8") as f:
            f.write("# RAPPORT D'INTERRUPTION — DÉCOUVERTES SAUVEGARDÉES\n\n")
            f.write(f"Date d'interruption : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Découvertes sauvegardées : {len(global_discoveries)}\n\n")
            f.write("## Dernières découvertes (top 20 par score)\n\n")

            sorted_disc = sorted(global_discoveries, key=lambda x: x.get('score', 0), reverse=True)[:20]
            for i, d in enumerate(sorted_disc, 1):
                f.write(f"### {i}. Score : {d.get('score', 'N/A')}\n")
                f.write(f"**Formule** : ${d.get('latex', d['formula'])}$\n")
                f.write(f"**Inspiration** : {d['inspiration']}\n\n")

        print(f"\nRapport d'urgence sauvegardé : {emergency_path}")
    else:
        print("\nAucune découverte à sauvegarder.")

    print("\nHydre Oméga arrêtée proprement. Tu peux relancer plus tard.")
    sys.exit(0)

# Activation du handler
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


# ==============================================================================
# VALIDATEUR PHYSIQUE + DIMENSIONNELLE
# ==============================================================================
class PhysicalValidator:
    @staticmethod
    def check_symmetry(expr):
        is_symmetric = sp.simplify(expr - expr.subs(x, -x)) == 0
        complexity = sp.count_ops(expr)
        return is_symmetric, complexity

    @staticmethod
    def check_dimensional_consistency(expr):
        try:
            dim_rules = {
                x: -1,
                y: -1,
                z: -1,
                t: -1,
                phi: 1,
                m: 1,
                G: -2,
                hbar: 0,
                delta: 1,
                f_a: 1,
                Lambda: 1,
                M_pl: 1,
            }
            dim_expr = expr.copy()

            # Substitution dimensionnelle
            for sym, d in dim_rules.items():
                if d != 0:
                    dim_expr = dim_expr.subs(sym, sp.Symbol("M", positive=True) ** d)
                else:
                    dim_expr = dim_expr.subs(sym, 1)

            # On développe et on regarde si tout est en M**4 (potentiel) ou constant
            expanded = sp.expand(dim_expr)
            terms = expanded.as_ordered_terms() if expanded.is_Add else [expanded]

            for term in terms:
                if term.has(sp.cos, sp.sin, sp.exp, sp.log, sp.Heaviside):
            # On extrait le coefficient devant la fonction transcendante
                    coeff = term.as_coeff_Mul()[0] if term.is_Mul else term
                    if not (coeff.is_Pow and coeff.exp == 4) and coeff != 1:
                        return False, "DIMENSIONAL_FAIL (transcendente)"
                elif not (term.is_Pow and term.exp == 4) and term != 1 and term != 0:
                    return False, "DIMENSIONAL_FAIL (puissance)"

            return True, "DIMENSIONAL_OK"
        except:
            return False, "DIMENSIONAL_FAIL"


# ==============================================================================
# LLM CRITIQUE
# ==============================================================================
class LLMAgent:
    @staticmethod
    def consult(formula_str, context_tag):
        prompt = f"""
        En tant que physicien théoricien expert, analyse cette formule générée par une IA évolutive :
        {formula_str}
        Contexte : {context_tag}
      
        1. Ressemblance avec un concept physique connu ?
        2. Interprétation audacieuse mais cohérente ?
        3. Note de plausibilité sur 100 ?
      
        Réponds en français, technique et concis.
        """
        try:
            payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            if response.status_code == 200:
                return (
                    response.json().get("response", "Pas de réponse du modèle").strip()
                )
            else:
                return f"Erreur HTTP {response.status_code} : le modèle local a répondu avec un code d'erreur."
        except requests.exceptions.Timeout:
            return "Analyse temporairement indisponible (timeout Ollama). Formule considérée comme potentiellement intéressante par défaut."
        except Exception as e:
            return f"Agent critique temporairement indisponible : {str(e)}. Découverte conservée pour analyse ultérieure."


# ==============================================================================
# HYDRE AGENTIQUE
# ==============================================================================
class AgenticHydra:
    def __init__(self, depth=0, energy=100):
        self.depth = depth
        self.energy = energy
        self.id = f"AGENT_{random.randint(1000, 9999)}"
        
    def get_inspiration(self):
        """
        Inspiration enrichie : privilégie les thèmes sous-explorés et interdisciplinaires.
        """
        # Thèmes sous-explorés (moins de 5 formules)
        under_explored = []
        for sub in self.__class__.available_subfamilies:
            sub_path = os.path.join("discoveries_thematic", sub.replace("/", os.sep))
            if os.path.isdir(sub_path):
                count = len([f for f in os.listdir(sub_path) if f.endswith(".json")])
                if count < 5:
                    under_explored.append(f"Thème sous-exploré : {sub}")

        # Mélange : 40% thèmes sous-explorés, 30% unsolved, 30% connaissance
        r = random.random()
        if r < 0.4 and under_explored:
            return random.choice(under_explored)
        elif r < 0.7:
            return "Thème à explorer : " + random.choice(UNSOLVED_THEMES)
        else:
            return "Connaissance de référence : " + random.choice(REAL_KNOWLEDGE_2025)

        # 30% de chance pour un thème sur le temps
        if random.random() < 0.3:
            return "Thème à explorer : " + random.choice([
                "Nature fondamentale du temps",
                "Flèche du temps",
                "Temps émergent en gravité quantique",
                "Temps et conscience",
                "Temps fractal"
            ])

    def classify_discovery_advanced(self, inspiration, formula_str):
        """
        Classifie en grande famille (via DISCOVERY_CATEGORIES) + sous-famille fine.
        Retourne (main_category_name, subcategory)
        """
        insp_lower = inspiration.lower()
        form_lower = formula_str.lower()

        # Détection par mots-clés dans inspiration ou formule
        if any(
            word in insp_lower
            for word in ["axion", "alps", "fuzzy", "periodic", "soliton", "dark matter"]
        ):
            if "cos(" in form_lower:
                return "Exotic_Matter_Discoveries", (
                    "Axion_Like" if "1 - cos" in form_lower else "Periodic_Potential"
                )
            return "Exotic_Matter_Discoveries", "Fuzzy_Dark_Matter"

        if "quintessence" in insp_lower or "dark energy" in insp_lower:
            if "exp(-" in form_lower:
                return "Exotic_Matter_Discoveries", "Ultra_Light_Scalars"
            return "Cosmology_Origins", "Early_Dark_Energy"

        if any(word in insp_lower for word in ["higgs", "portal", "relaxion"]):
            if "phi**6" in form_lower or "phi^6" in form_lower:
                return "Quantum_Laws", "Higgs_Mechanisms"
            return "Quantum_Laws", "Relaxion"

        if any(
            word in insp_lower
            for word in ["black hole", "superradiance", "no-hair", "swampland"]
        ):
            if "log(" in form_lower:
                return "Quantum_Gravity", "Holographic"
            if "m_pl" in form_lower:
                return "Quantum_Gravity", "Planck_Suppressed"
            return "Gravity_Laws", "Superradiance"

        if any(word in insp_lower for word in ["multiverse", "everett", "bubble"]):
            return "Multiverse_Theories", "Inflationary_Bubbles"

        if any(word in insp_lower for word in ["wormhole", "traversable", "er=epr"]):
            return "Wormholes_Theories", "Entangled_ER=EPR"

        if any(
            word in insp_lower
            for word in ["dna", "adn", "microtubules", "conscience", "orch-or"]
        ):
            if "bessel" in form_lower:
                return "DNA_Cosmic_Links", "Fractal_Patterns"
            return "DNA_Cosmic_Links", "Orch_OR"

        if "log(" in form_lower and "entropy" in insp_lower:
            return "Quantum_Gravity", "Holographic"

        # Par défaut
        return "Uncategorized", "General"

    # ==============================================================
    # TEMPLATES PARTAGÉS – DÉFINIS UNE SEULE FOIS POUR TOUS LES AGENTS
    # ==============================================================
    kg_templates = [
        # 1. KG standard (masse)
        lambda: 0.5 * (sp.diff(phi, t)**2 - sp.diff(phi, x)**2 - sp.diff(phi, y)**2 - sp.diff(phi, z)**2) - (m**2 * phi**2) / 2,

        # 2. KG sans masse (conforme)
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * (sp.diff(phi, x)**2 + sp.diff(phi, y)**2 + sp.diff(phi, z)**2),

        # 3. KG avec potentiel quartique
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 - (m**2 * phi**2)/2 + (lambda_q / 4) * phi**4,

        # 4. KG tachyonique (masse négative)
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + (m**2 * phi**2)/2,

        # 5. KG avec potentiel mexicain
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * (phi**2 / delta**2 - 1)**2 / 4,

        # 6. KG avec potentiel axion-like
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * (1 - sp.cos(phi / f_a)),

        # 7. KG avec correction holographique
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * sp.log(1 + (phi / M_pl)**2),

        # 8. KG avec terme non-linéaire cinétique (k-essence)
        lambda: sp.diff(phi, t)**4 / 4 - 0.5 * sp.grad(phi)**2 - m**2 * phi**2 / 2,

        # 9. KG avec potentiel exponentiel (quintessence)
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * sp.exp(-phi / M_pl),

        # 10. KG avec potentiel inverse power-law
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 / (1 + (phi / M_pl)**2)**2,

        # 11. KG avec potentiel log holographique haute énergie
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + (phi / M_pl)**8 * sp.log(1 + phi**2 / M_pl**2) / M_pl**4,

        # 12. KG avec terme cinétique non-standard (ghost condensate)
        lambda: -0.5 * sp.diff(phi, t)**2 + 0.5 * sp.grad(phi)**2 - m**2 * phi**2 / 2,

        # 13. KG avec potentiel fractal/périodique
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + m**4 * sp.sin(phi / delta)**4,

        # 14. KG avec effet quantique explicite (ħ)
        lambda: hbar * sp.diff(phi, t)**2 / (2 * m**2) - 0.5 * sp.grad(phi)**2 - m**2 * phi**2 / 2,

        # 15. KG avec potentiel sigmoidale (transition douce)
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * (1 + sp.tanh(phi / delta)) / 2,

        # 16. KG avec terme cinétique spatial anisotrope
        lambda: 0.5 * sp.diff(phi, t)**2 - sp.diff(phi, x)**2 - 0.1 * (sp.diff(phi, y)**2 + sp.diff(phi, z)**2) - m**2 * phi**2 / 2,

        # 17. KG avec potentiel double exponentiel
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + Lambda**4 * (sp.exp(-phi / M_pl) + sp.exp(phi / M_pl)),

        # 18. KG avec potentiel confiné gaussien
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * sp.grad(phi)**2 + m**4 * sp.exp(-phi**2 / delta**2),
    ]

    axion_templates = [
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a)),  # potentiel axion classique
        lambda: Lambda**4 * (1 - sp.cos(2 * phi / f_a)) / 2,  # premier harmonique
        lambda: Lambda**4 * (1 - sp.cos(3 * phi / f_a)) / 3,  # second harmonique
        lambda: Lambda**4 * sp.sin(phi / f_a)**2,  # forme équivalente
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a + k * x)),  # dépendance spatiale (instabilité)
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a)) * sp.exp(-phi**2 / M_pl**2),  # confinement Planck
        lambda: Lambda**4 * sp.log(1 + sp.cos(phi / f_a)**2),  # log périodique
        lambda: Lambda**4 * sp.tanh(phi / f_a)**2,  # saturation axion-like
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a))**2,  # double barrière
        lambda: Lambda**4 * sp.sech(phi / f_a)**2,  # potentiel plat axion
        lambda: Lambda**4 * (phi / f_a)**2 * sp.sin(phi / f_a),  # hybride polynomial-périodique
        lambda: Lambda**4 * sp.besselj(0, phi / f_a),  # mode radial axion
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a)) * sp.log(1 + phi**2 / M_pl**2),  # axion + holographie
        lambda: Lambda**4 * sp.sin(phi / f_a)**4 / 4,  # puissance supérieure
        lambda: Lambda**4 * sp.Heaviside(f_a - phi) * (1 - sp.cos(phi / f_a)),  # confinement causal
        lambda: Lambda**4 * sp.cos(k * x) * (1 - sp.cos(phi / f_a)),  # modulation spatiale
        lambda: Lambda**4 * sp.exp(-phi**2 / f_a**2) * sp.sin(phi / f_a)**2,  # soliton axion-like
    ]

    # === TEMPLATES HIGGS PORTAL — COUPLAGES SCALAIRE-HIGGS & BEYOND SM ===
    higgs_portal_templates = [
        lambda: - (m**2 / 2) * phi**2 + (lambda_q / 4) * phi**4,  # double puits classique
        lambda: (Lambda**4 / 4) * (phi**2 / delta**2 - 1)**2,  # potentiel mexicain stabilisé
        lambda: Lambda**4 * (phi / M_pl)**6 / 6 + m**4 * (phi / M_pl)**2,  # sextique swampland-compatible
        lambda: m**4 * phi**2 + Lambda**4 * phi**4 / M_pl**4,  # portal quartique supprimé
        lambda: - Lambda**4 * phi**2 / 2 + lambda_q * phi**4 / 4,  # brisure spontanée
        lambda: Lambda**4 * sp.tanh(phi / delta)**4,  # transition douce Higgs-like
        lambda: m**4 * sp.log(1 + phi**2 / delta**2),  # portal logarithmique
        lambda: Lambda**4 * sp.sech(phi / delta)**2 * phi**2,  # puits plat + masse
        lambda: (phi / M_pl)**4 * Lambda**4 + m**2 * phi**2,  # hybride Planck + masse
        lambda: Lambda**4 * (phi**2 / delta**2)**2 * sp.exp(-phi**2 / delta**2),  # confinement gaussien
        lambda: - m**2 * phi**2 + Lambda**4 * (phi / M_pl)**4,  # portal non-renormalisable
        lambda: Lambda**4 * sp.cos(phi / f_a)**2 + m**4 * phi**2,  # oscillation + masse
        lambda: Lambda**4 * sp.Heaviside(delta - phi) * phi**4,  # seuil de brisure
        lambda: m**4 * sp.sin(phi / delta)**2 + Lambda**4 * phi**4 / M_pl**4,  # périodique + portal
        lambda: Lambda**4 * sp.log(cosh(phi / delta)),  # potentiel hyperbolique Higgs-like
        lambda: (phi / M_pl)**8 * Lambda**4 / 24 + m**4 * phi**2,  # très haut ordre portal
        lambda: Lambda**4 * sp.tanh(phi / M_pl)**2 * sp.exp(-phi / M_pl),  # portal évolutif
    ]

    quintessence_templates = [
        lambda: Lambda**4 * sp.exp(-phi / M_pl),  # tracker classique
        lambda: Lambda**4 / (1 + (phi / M_pl)**2)**2,  # hilltop quintessence
        lambda: Lambda**4 * sp.sech(phi / M_pl)**2,  # potentiel plat sech
        lambda: Lambda**4 * sp.exp(-2 * phi / M_pl),  # steep exponential
        lambda: Lambda**4 * sp.tanh(phi / M_pl)**2,  # transition douce
        lambda: Lambda**4 * sp.log(1 + sp.exp(-phi / M_pl)),  # soft exponential
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a))**2,  # natural quintessence
        lambda: Lambda**4 * sp.exp(-phi**2 / M_pl**2),  # gaussien centré
        lambda: Lambda**4 * sp.Heaviside(M_pl - phi) * sp.exp(phi / M_pl),  # asymétrique
        lambda: Lambda**4 / (1 + sp.exp(phi / M_pl)),  # sigmoïde inversée
        lambda: Lambda**4 * sp.sech(phi / f_a)**4,  # ultra-plat
        lambda: Lambda**4 * sp.cos(phi / M_pl)**2,  # oscillation cosmologique
        lambda: Lambda**4 * sp.log(1 + (phi / M_pl)**4),  # log polynomial
        lambda: Lambda**4 * sp.tanh(phi / (2 * M_pl))**4,  # plus lent
        lambda: Lambda**4 * sp.exp(-sp.Abs(phi / M_pl)),  # symétrique
        lambda: Lambda**4 * (phi / M_pl)**2 * sp.exp(-phi / M_pl),  # hybride
    ]

    # === TEMPLATES GRAVITY — CORRECTIONS NON-RENORMALISABLES & ÉMERGENCE ===
    gravity_templates = [
        lambda: G * phi**4 / M_pl**2,  # classique non-renormalisable
        lambda: (phi / M_pl)**4 * Lambda**4,  # suppression Planck
        lambda: G * m**2 * phi**2 / M_pl**2,  # masse induite
        lambda: (phi / M_pl)**6 * Lambda**4 * M_pl**2,  # sextique compensé
        lambda: G * phi**6 / M_pl**4,  # haut ordre gravitationnel
        lambda: Lambda**4 * sp.log(1 + G * phi**2 / M_pl**2),  # log gravitationnel
        lambda: (phi / M_pl)**2 * phi**2 * Lambda**2,  # phi^4 / M_pl^2
        lambda: G * phi**4 / sp.sqrt(x**2 + y**2 + z**2 + delta**2),  # gravité à courte portée
        lambda: (phi / M_pl)**8 * Lambda**4 / M_pl**4,  # très haut ordre
        lambda: G * m**4 * phi**2 / M_pl**4,  # couplage masse-gravité
        lambda: Lambda**4 * sp.tanh(G * phi**2 / M_pl**2),  # saturation gravitationnelle
        lambda: (phi / M_pl)**4 * sp.exp(-phi / M_pl),  # gravité émergente
        lambda: G * phi**2 * sp.log(1 + phi**2 / M_pl**2) / M_pl**2,
        lambda: Lambda**4 * sp.Heaviside(M_pl - phi) * (phi / M_pl)**4,
        lambda: G * phi**8 / M_pl**6,  # ultra haut ordre
        lambda: (phi / M_pl)**4 * sp.cos(phi / f_a)**2 * Lambda**4,  # oscillation gravitationnelle
    ]

    # === TEMPLATES SWAMPLAND — CONTRAINTES DE CONSISTENCE QUANTIQUE ===
    swampland_templates = [
        lambda: (phi / M_pl)**4 * sp.log(1 + phi**2 / M_pl**2) * Lambda**4,  # log swampland
        lambda: Lambda**4 * sp.tanh(phi / f_a)**2,  # distance conjecture
        lambda: (phi / M_pl)**6 * sp.log(phi / M_pl + 1) / 6,  # de Sitter conjecture
        lambda: Lambda**4 * sp.exp(-c * phi / M_pl),  # avec c ~ O(1) swampland
        lambda: (phi / M_pl)**8 * sp.tanh(phi / M_pl)**4 / M_pl**4,
        lambda: Lambda**4 * sp.log(1 + sp.exp(-phi / M_pl)),  # soft dS
        lambda: (phi / M_pl)**4 * sp.Heaviside(M_pl - phi),
        lambda: Lambda**4 * sp.sech(phi / M_pl)**6,  # ultra-plat (danger swampland)
        lambda: (phi / M_pl)**10 * Lambda**4 / 120,  # haut ordre contrôlé
        lambda: Lambda**4 * sp.log(cosh(phi / M_pl)),  # hyperbolic swampland
        lambda: (phi / M_pl)**2 * sp.exp(-phi / M_pl) * Lambda**2,
        lambda: Lambda**4 * sp.tanh(phi / M_pl)**6,
        lambda: (phi / M_pl)**4 * sp.log(1 + sp.tanh(phi / M_pl)**2),
        lambda: Lambda**4 * sp.Heaviside(phi) * sp.exp(-phi / f_a),
        lambda: (phi / M_pl)**12 * sp.sech(phi / M_pl)**4 / 720,
        lambda: Lambda**4 * sp.cos(phi / f_a)**4,  # oscillation potentiellement instable
    ]

    dna_cosmic_templates = [
        ldna_cosmic_templates = [
        lambda: Lambda**4 * sp.exp(-phi / M_pl),  # mémoire primordiale
        lambda: Lambda**4 * (phi / M_pl)**6,      # puits haut ordre
        lambda: m**4 * sp.log(sp.sin(phi / delta)**2 + 1),  # motif périodique ADN + entropie
        lambda: Lambda**4 * sp.log(1 + (phi / M_pl)**2),  # correction holographique
        lambda: m**4 * sp.sin(phi / delta)**4,    # puissance 4 pour hélicoïdal
        lambda: delta**4 * sp.log(1 + sp.Abs(phi / delta)**1.618),  # section dorée
        lambda: hbar * sp.log(sp.sin(phi / delta)**2 + 1) * m**2,  # effet quantique microtubule (avec ħ neutralisé plus tard)
    ]

    time_templates = [
        lambda: hbar * sp.diff(phi, t)**2 / (2 * m**2),  # énergie cinétique temporelle
        lambda: Lambda**4 * sp.exp(-t / delta),  # décroissance temporelle
        lambda: delta**4 * sp.log(1 + t**2 / delta**2),  # entropie temporelle
        lambda: m**4 * sp.Heaviside(t) * sp.exp(-t / delta),  # évolution causale
        lambda: (phi / M_pl)**4 * sp.sin(t / delta)**2,  # oscillation temporelle
        lambda: hbar * sp.log(1 + sp.diff(phi, t)**2),  # terme entropique cinétique
    ]

    all_templates = (
        kg_templates
        + axion_templates
        + higgs_portal_templates
        + quintessence_templates
        + gravity_templates
        + swampland_templates
        + dna_cosmic_templates
        + time_templates
    )

    # === DÉTECTION DYNAMIQUE DES FAMILLES ET SOUS-FAMILLES DISPONIBLES ===
    thematic_base = "discoveries_thematic"
    available_families = []
    available_subfamilies = []

    if os.path.exists(thematic_base):
        for family in os.listdir(thematic_base):
            family_path = os.path.join(thematic_base, family)
            if os.path.isdir(family_path):
                available_families.append(family)
                for sub in os.listdir(family_path):
                    sub_path = os.path.join(family_path, sub)
                    if os.path.isdir(sub_path):
                        available_subfamilies.append(f"{family}/{sub}")

    print(f"[THÉMATIQUE] {len(available_families)} familles et {len(available_subfamilies)} sous-familles détectées au démarrage")
    print(f"Familles disponibles : {', '.join(available_families)}")

    # === TEMPLATES DYNAMIQUES GÉNÉRÉS PAR LE SUPERVISEUR ===
    supervisor_templates = []

    if evolution_directives:
        for directive in evolution_directives:
            terms = directive.get("common_terms", [])
            if "log_holographic" in terms:
                supervisor_templates.append(lambda: Lambda**4 * sp.log(1 + (phi / M_pl)**2))
            if "periodic_sin" in terms:
                supervisor_templates.append(lambda: m**4 * sp.sin(phi / delta)**2 / 2)
            if "high_order" in terms:
                supervisor_templates.append(lambda: (phi / M_pl)**8 * Lambda**4 * sp.log(1 + phi**2 / M_pl**2))
            if "gravitational" in terms:
                supervisor_templates.append(lambda: (phi / M_pl)**4 * Lambda**4)

        print(f"[ÉVOLUTION AVANCÉE] {len(supervisor_templates)} templates dynamiques créés à partir des directives")

    # Fusion avec les templates classiques (supervisor en priorité)
    all_templates = supervisor_templates + all_templates

    # ==============================================================
    # CHARGEMENT DES PATTERNS APPRIS (mémoire douce de l'Hydre)
    # ==============================================================
    learned_templates = []
    patterns_file = "successful_patterns.json"

    if os.path.exists(patterns_file):
        try:
            with open(patterns_file, "r", encoding="utf-8") as f:
                patterns = json.load(f)

            # On prend seulement les 30 plus récents pour éviter la domination
            for pattern in patterns[-30:]:
                try:
                    expr = sp.sympify(pattern["corrected"])
                    learned_templates.append(lambda: expr)
                except:
                    pass
            print(
                f"[MÉMOIRE] {len(learned_templates)} patterns appris chargés (influence douce activée)"
            )
        except Exception as e:
            print(f"[MÉMOIRE] Erreur lecture patterns : {e}")

    # Mélange équilibré : 30% chance de piocher dans les appris, 70% dans les originaux
    # → Influence sans domination
    combined_templates = all_templates + learned_templates

    # ==============================================================
    def generate_candidate(self):
        """
        Génération contextualisée avec forte emphasis sur la créativité et l'hybridation.
        """
        inspiration = self.get_inspiration()
        insp_lower = inspiration.lower()

        # === 1. Sélection avec forte influence du Superviseur si directives présentes
        if self.__class__.supervisor_templates and random.random() < 0.7:
            base = random.choice(self.__class__.supervisor_templates)()
            print("[ÉVOLUTION GUIDÉE] Utilisation d'un template recommandé par le Superviseur")
        else:
            base = random.choice(self.__class__.all_templates)()

        # === 2. NOMBRE DE MUTATIONS AUGMENTÉ — EXPLORATION PROFONDE (5 à 15 couches) ===
        num_mutations = random.randint(5, 15)  # plus de couches = plus de complexité et d'hybridation
        print(f"[CRÉATIVITÉ BOOSTÉE] Application de {num_mutations} mutations successives")

        for _ in range(num_mutations):
            if random.random() < 0.7:  # 70% chance : mutation guidée par l'inspiration
                if any(word in insp_lower for word in ["axion", "alps", "fuzzy", "dark matter"]):
                    base += random.choice([
                        Lambda**4 * (1 - sp.cos(phi / f_a)),
                        Lambda**4 * (1 - sp.cos(2 * phi / f_a)) / 2,
                        Lambda**4 * sp.sin(phi / f_a)**2,
                        Lambda**4 * (1 - sp.cos(3 * phi / f_a)) / 3,  # harmoniques supérieurs
                    ])

                elif any(word in insp_lower for word in ["quintessence", "dark energy", "w(z)"]):
                    base += random.choice([
                        Lambda**4 * sp.exp(-phi / M_pl),
                        Lambda**4 / (1 + (phi / M_pl)**2)**2,
                        Lambda**4 * sp.sech(phi / M_pl)**2,
                        Lambda**4 * sp.exp(-2 * phi / M_pl),  # plus raide
                    ])

                elif any(word in insp_lower for word in ["black hole", "swampland", "holograph"]):
                    base += random.choice([
                        Lambda**4 * sp.log(1 + (phi / M_pl)**2),
                        (phi / M_pl)**4 * Lambda**4,
                        Lambda**4 * sp.tanh(phi / M_pl)**2,
                        (phi / M_pl)**8 * sp.log(1 + phi**2 / M_pl**2) / M_pl**4,  # haut ordre holographique
                    ])

                elif any(word in insp_lower for word in ["wormhole", "multiverse", "everett"]):
                    base += random.choice([
                        delta**4 * sp.log(1 + phi**2 / delta**2),
                        Lambda**4 * sp.exp(-(phi / delta)**2),
                        delta**4 * sp.Heaviside(delta - phi) * sp.log(1 + phi**2 / delta**2),
                        (phi / M_pl)**4 * sp.tanh(phi / delta)**4,
                    ])

                elif any(word in insp_lower for word in ["dna", "conscience", "orch-or", "microtubules"]):
                    base += random.choice([
                        delta * sp.besselj(0, phi / delta) * sp.sin(k * x),
                        hbar * sp.log(sp.sin(phi / delta)**2 + 1) * m**2,
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**1.618),  # section dorée (Fibonacci)
                        m**4 * sp.besselj(1, phi / delta)**2,
                    ])

                else:
                    # Mutation totalement libre — créativité maximale
                    base += random.choice([
                        (phi / M_pl)**6 * Lambda**4 / 6,
                        m**4 * sp.tanh(phi / delta)**4,
                        delta**4 * sp.log(1 + (phi / delta)**3),
                        Lambda**4 * sp.sech(phi / f_a)**4,
                        G * phi**4 / M_pl**2,
                        hbar**2 * sp.diff(phi, t)**2 / (2 * m**2),  # terme cinétique
                        (phi / M_pl)**10 * Lambda**4 / 120,  # très haut ordre
                    ])

            else:
                # 30% chance : mutation purement aléatoire (exploration sauvage)
                base += random.choice([
                    delta * m**2 * phi**2,
                    m**4 * sp.sin(phi / delta)**4 / 4,
                    hbar**2 * sp.diff(phi, x)**2 / (2 * m**2),
                    delta**4 * sp.log(1 + sp.Abs(phi / delta)**2),
                    (phi / M_pl)**8 * Lambda**4 / 24,
                    G * m**2 * phi**2 / sp.sqrt(x**2 + y**2 + z**2 + delta**2),
                    Lambda**4 * sp.Heaviside(phi - delta),
                ])

        # === 3. Hybridation finale : parfois fusion de deux templates ===
        if random.random() < 0.25:  # 25% chance d'hybridation audacieuse
            extra = random.choice(self.__class__.all_templates)()
            base += 0.5 * extra  # coefficient modéré pour ne pas dominer
            print("[HYBRIDATION] Fusion créative de deux templates")

        # === 4. MUTATIONS TEMPORELLES (exploration du temps) ===
        if random.random() < 0.4:  # 40% de chance — le temps est prioritaire
            print("[TEMPS] Exploration temporelle activée")
            base += random.choice([
                hbar * sp.diff(phi, t)**2 / (2 * m**2),  # énergie cinétique temporelle
                Lambda**4 * sp.exp(-t / delta),          # décroissance temporelle
                delta**4 * sp.log(1 + t**2 / delta**2),  # entropie temporelle croissante
                m**4 * sp.Heaviside(t) * sp.exp(-t / delta),  # évolution causale
                (phi / M_pl)**4 * sp.sin(t / delta)**2,  # oscillation temporelle
                hbar * sp.log(1 + sp.diff(phi, t)**2),   # entropie cinétique temporelle
                Lambda**4 * sp.Heaviside(t) * sp.log(1 + t / delta),  # flèche du temps
                delta**4 * sp.tanh(t / delta)**2,        # saturation temporelle
                hbar * sp.diff(phi, t)**4 / (24 * m**4), # terme non-linéaire temporel
                Lambda**4 * sp.log(sp.exp(-t / delta) + 1),  # fonction soft de temps
                m**4 * sp.cos(t / delta)**4,             # périodicité temporelle
                delta**4 * sp.Heaviside(-t) * sp.exp(t / delta),  # anti-causalité (pour multivers)
                (phi / M_pl)**6 * sp.sin(t / M_pl)**2 / 6,  # oscillation Planckienne
                hbar * sp.log(1 + sp.diff(phi, t)**4),   # entropie temporelle haute énergie
                Lambda**4 * sp.sech(t / delta)**2,       # pic temporel
            ])

                # === EXPLORATION FORCÉE DE NOUVELLES FAMILLES (30% de chance) ===
        if random.random() < 0.3:  # Monte à 0.4 si tu veux encore plus d'exploration
            if self.__class__.available_families:
                random_family = random.choice(self.__class__.available_families)
                print(f"[EXPLORATION LARGE] Inspiration forcée vers famille : {random_family}")
                fam_lower = random_family.lower()

                # ≈15 choix par famille — créativité maximale
                if "wormhole" in fam_lower:
                    base += random.choice([
                        delta**4 * sp.log(1 + phi**2 / delta**2),                          # throat classique
                        Lambda**4 * sp.exp(-(phi / delta)**2),                            # confinement gaussien
                        (phi / M_pl)**4 * sp.tanh(phi / delta)**2,                        # potentiel plat traversable
                        delta**4 * sp.sech(phi / delta)**4,                               # double puits gorge
                        Lambda**4 * sp.Heaviside(delta - phi) * sp.exp(phi / delta),      # asymétrie causale
                        delta**4 * sp.log(cosh(phi / delta)),                             # métrique wormhole-like
                        (phi / delta)**6 * sp.log(1 + phi**2 / delta**2),                 # haut ordre throat
                        Lambda**4 * sp.cos(phi / delta)**2,                               # oscillation dans le throat
                        delta**4 * sp.tanh(phi / delta)**4,                               # barrière douce
                        Lambda**4 * sp.exp(-sp.Abs(phi / delta)),                         # confinement symétrique
                        (phi / M_pl)**4 * sp.sin(k * x) * sp.cos(t / delta),              # onde traversant le throat
                        delta**4 * sp.log(1 + sp.exp(phi / delta)),                       # fonction sigmoidale gorge
                        Lambda**4 * sp.Heaviside(M_pl - phi) * sp.log(1 + phi**2 / M_pl**2),
                        delta**4 * sp.besselj(0, phi / delta),                            # mode radial discret
                        (phi / delta)**8 * sp.sech(phi / delta)**2,                       # très haut ordre stabilisé
                    ])

                elif "dna" in fam_lower or "cosmic_links" in fam_lower:
                    base += random.choice([
                        delta * sp.besselj(0, m * phi / delta) * sp.sin(k * x),
                        hbar * sp.log(sp.sin(phi / delta)**2 + 1) * m**2,
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**1.618),                # section dorée Fibonacci
                        m**4 * sp.besselj(1, phi / delta)**2,
                        delta**4 * sp.sin(phi / delta)**6,                                # motif fractal biologique
                        hbar * sp.cos(k * x) * sp.besselj(2, phi / delta),
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**2.807),                # dimension fractale poumons
                        m**4 * sp.Abs(phi / delta)**1.618,                                # croissance dorée
                        delta * sp.besselj(0, phi / delta) * sp.cos(t / delta),           # oscillation temporelle ADN
                        hbar * sp.log(1 + sp.diff(phi, x)**2),                            # entropie spatiale
                        delta**4 * sp.tanh(phi / delta)**3 * sp.sin(k * x),                # motif hélicoïdal
                        m**4 * sp.exp(-phi**2 / delta**2) * sp.cos(2 * phi / delta),      # soliton oscillant
                        delta**4 * sp.log(1 + sp.sin(phi / delta)**4),
                        hbar * m**2 * sp.Heaviside(phi) * sp.log(1 + phi**2 / delta**2),
                        delta**4 * sp.besselj(3, phi / delta)**2,
                    ])

                elif "multiverse" in fam_lower:
                    base += random.choice([
                        Lambda**4 * sp.Heaviside(phi - M_pl),
                        (phi / M_pl)**4 * sp.log(1 + phi**2 / M_pl**2),
                        delta**4 * sp.tanh(phi / delta)**3,
                        Lambda**4 * sp.exp(-sp.Abs(phi / M_pl)),
                        (phi / M_pl)**6 * sp.Heaviside(M_pl - phi),
                        delta**4 * sp.log(1 + sp.exp(phi / delta)),
                        Lambda**4 * sp.cos(phi / M_pl)**2,
                        (phi / M_pl)**8 * sp.tanh(phi / M_pl)**4,
                        delta**4 * sp.sech(phi / delta)**6,
                        Lambda**4 * sp.Heaviside(phi) * sp.exp(-phi / M_pl),
                        (phi / M_pl)**4 * sp.sin(t / delta),
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**3),
                        Lambda**4 * sp.besselj(0, phi / M_pl),
                        (phi / M_pl)**10 * sp.log(1 + phi**2 / M_pl**2) / 120,
                        delta**4 * sp.Heaviside(delta - phi) * sp.tanh(phi / delta)**2,
                    ])

                elif "fractal" in fam_lower or "mathematical_laws" in fam_lower:
                    base += random.choice([
                        m**4 * sp.sin(phi / delta)**4,
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**1.5),
                        hbar * sp.besselj(1, phi / delta) * sp.cos(k * x),
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**2.807),
                        m**4 * sp.Abs(phi / delta)**1.618,
                        delta**4 * sp.sin(phi / delta)**8 / 8,
                        hbar * sp.log(1 + sp.Abs(sp.diff(phi, x))**1.5),
                        delta**4 * sp.besselj(0, phi / delta)**2,
                        m**4 * sp.exp(-sp.Abs(phi / delta)**0.5),
                        delta**4 * sp.log(1 + sp.sin(k * x)**2 * sp.Abs(phi / delta)**2),
                        hbar * sp.cos(t / delta) * sp.Abs(phi / delta)**1.3,
                        delta**4 * sp.tanh(sp.Abs(phi / delta)**2)**3,
                        m**4 * sp.Heaviside(phi) * sp.Abs(phi / delta)**2.5,
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**3.0),
                        hbar * sp.besselj(2, phi / delta) * sp.sin(k * x),
                    ])

                elif "quantum_gravity" in fam_lower:
                    base += random.choice([
                        (phi / M_pl)**8 * sp.log(1 + phi**2 / M_pl**2) / M_pl**4,
                        G * phi**4 / M_pl**2,
                        Lambda**4 * sp.sech(phi / M_pl)**2,
                        (phi / M_pl)**10 * Lambda**4 / 120,
                        hbar * sp.diff(phi, t)**2 / (2 * m**2),
                        (phi / M_pl)**6 * sp.log(1 + G * phi**2 / M_pl**2) / 6,
                        Lambda**4 * sp.exp(-phi**2 / M_pl**2),
                        G * m**2 * phi**2 / M_pl**2,
                        (phi / M_pl)**12 * sp.tanh(phi / M_pl)**4 / 720,
                        hbar * sp.log(1 + sp.diff(phi, x)**2 + sp.diff(phi, t)**2),
                        Lambda**4 * sp.Heaviside(M_pl - phi),
                        (phi / M_pl)**4 * sp.besselj(0, phi / M_pl),
                        G * phi**6 / M_pl**4,
                        Lambda**4 * sp.cos(phi / M_pl)**4,
                        (phi / M_pl)**8 * sp.sech(phi / M_pl)**6,
                    ])

                elif "holographic" in fam_lower:
                    base += random.choice([
                        Lambda**4 * sp.log(1 + (phi / M_pl)**2),
                        (phi / M_pl)**6 * sp.log(1 + G * phi**2 * M_pl**2) / 6,
                        hbar * sp.diff(phi, t)**2 / (2 * m**2),
                        Lambda**4 * sp.log(1 + sp.diff(phi, x)**2),
                        (phi / M_pl)**8 * sp.log(1 + phi**2 / M_pl**2) / M_pl**4,
                        Lambda**4 * sp.tanh(phi / M_pl)**4,
                        hbar * sp.log(1 + sp.diff(phi, t)**2 + sp.diff(phi, x)**2),
                        (phi / M_pl)**4 * sp.log(1 + sp.exp(phi / M_pl)),
                        Lambda**4 * sp.sech(phi / M_pl)**4,
                        hbar * sp.besselj(0, phi / M_pl) * sp.diff(phi, t),
                        Lambda**4 * sp.log(1 + G * phi**2 / M_pl**2),
                        (phi / M_pl)**10 * sp.log(1 + phi**2 / M_pl**2) / 120,
                        hbar * sp.log(sp.Abs(sp.diff(phi, x))**2 + 1),
                        Lambda**4 * sp.Heaviside(phi) * sp.log(1 + phi**2 / M_pl**2),
                        (phi / M_pl)**6 * sp.cos(k * x)**2,
                    ])

                elif "meta_evolution" in fam_lower:
                    base += random.choice([
                        (phi / M_pl)**4 * sp.exp(-phi / M_pl),
                        delta**4 * sp.log(1 + sp.exp(phi / delta)),
                        m**4 * sp.sigmoid(phi / delta),
                        Lambda**4 * sp.tanh(phi / M_pl)**6,
                        delta**4 * sp.Heaviside(phi) * sp.exp(-phi / delta),
                        (phi / M_pl)**8 * sp.sech(phi / M_pl)**2,
                        m**4 * sp.log(1 + sp.exp(phi / delta)),
                        delta**4 * sp.cos(phi / delta)**4,
                        Lambda**4 * sp.Heaviside(M_pl - phi),
                        (phi / delta)**4 * sp.sin(t / delta),
                        delta**4 * sp.besselj(0, phi / delta)**2,
                        m**4 * sp.exp(-sp.Abs(phi / delta)),
                        Lambda**4 * sp.log(1 + sp.tanh(phi / M_pl)**2),
                        delta**4 * sp.Heaviside(delta - phi) * phi**2,
                        (phi / M_pl)**6 * sp.sigmoid(phi / M_pl),
                    ])

                elif "quantum_classical_bridges" in fam_lower:
                    base += random.choice([
                        hbar**2 * sp.diff(phi, x)**2 / (2 * m**2),
                        m**4 * sp.tanh(phi / delta)**4,
                        Lambda**4 * sp.Heaviside(phi - delta),
                        hbar * sp.diff(phi, t)**2 / (2 * m**2),
                        m**4 * sp.sech(phi / delta)**6,
                        Lambda**4 * sp.exp(-sp.Abs(phi / delta)),
                        hbar**2 * sp.diff(phi, x)**2 * sp.diff(phi, t)**2,
                        m**4 * sp.cos(phi / delta)**4,
                        Lambda**4 * sp.Heaviside(delta - phi) * sp.tanh(phi / delta),
                        hbar * sp.log(1 + sp.diff(phi, x)**2),
                        m**4 * sp.Heaviside(phi) * phi**2,
                        Lambda**4 * sp.sigmoid(phi / delta),
                        hbar**2 * sp.diff(phi, t)**2 * sp.Heaviside(t),
                        m**4 * sp.exp(-phi**2 / delta**2) * sp.cos(k * x),
                        Lambda**4 * sp.log(1 + sp.Abs(phi / delta)),
                    ])

                elif "classical_physics" in fam_lower:
                    base += random.choice([
                        0.5 * m**2 * phi**2 + (lambda_q / 4) * phi**4,
                        G * m**2 * phi**2 / sp.sqrt(x**2 + y**2 + z**2 + delta**2),
                        m**4 * phi**4 / 4,
                        G * phi**4 / (4 * sp.sqrt(x**2 + y**2 + z**2)),
                        0.5 * m**2 * sp.diff(phi, t)**2,
                        Lambda**4 * (1 - sp.cos(phi / f_a)),
                        m**4 * sp.exp(-phi**2 / delta**2),
                        G * m**2 * phi**2 / (x**2 + y**2 + z**2 + delta**2),
                        0.5 * sp.diff(phi, x)**2 + 0.5 * sp.diff(phi, y)**2 + 0.5 * sp.diff(phi, z)**2,
                        m**4 * sp.sin(phi / delta)**2,
                        Lambda**4 * phi**2 / 2,
                        G * phi**6 / (6 * M_pl**2),
                        m**4 * sp.tanh(phi / delta)**2,
                        Lambda**4 * sp.cos(phi / f_a)**2,
                        G * m**2 * phi**2 * sp.exp(-sp.sqrt(x**2 + y**2 + z**2) / delta),
                    ])

                elif "cosmology_origins" in fam_lower:
                    base += random.choice([
                        Lambda**4 * (1 - sp.cos(phi / f_a))**2,
                        m**4 * sp.exp(-phi / M_pl),
                        Lambda**4 * sp.sech(phi / M_pl)**2,
                        m**4 * sp.tanh(phi / M_pl)**4,
                        Lambda**4 * sp.log(1 + sp.exp(-phi / M_pl)),
                        (phi / M_pl)**4 * sp.exp(-phi / M_pl),
                        Lambda**4 * sp.Heaviside(M_pl - phi),
                        m**4 * sp.cos(phi / f_a)**4,
                        Lambda**4 * sp.sin(phi / M_pl)**2,
                        (phi / M_pl)**6 * sp.sech(phi / M_pl)**2 / 6,
                        Lambda**4 * sp.exp(-2 * phi / M_pl),
                        m**4 * sp.log(1 + sp.exp(phi / M_pl)),
                        Lambda**4 * sp.besselj(0, phi / M_pl),
                        (phi / M_pl)**8 * sp.exp(-phi / M_pl) / 24,
                        Lambda**4 * sp.Heaviside(phi) * sp.tanh(phi / M_pl),
                    ])

                else:
                    # Famille inconnue ou nouvelle → exploration totalement sauvage
                    base += random.choice([
                        (phi / M_pl)**12 * Lambda**4 / 720,
                        delta**4 * sp.besselj(3, phi / delta),
                        hbar * sp.log(1 + sp.diff(phi, t)**2 + sp.diff(phi, x)**2),
                        Lambda**4 * sp.sech(phi / delta)**6,
                        m**4 * sp.Abs(phi / delta)**3.0,
                        delta**4 * sp.cos(k * x * t / delta),
                        hbar * sp.diff(phi, t)**4 / (24 * m**4),
                        Lambda**4 * sp.log(1 + sp.exp(sp.Abs(phi / M_pl))),
                        (phi / delta)**10 * sp.sin(phi / delta)**2 / 120,
                        delta**4 * sp.Heaviside(phi) * sp.Heaviside(delta - phi),
                        m**4 * sp.besselj(4, phi / delta),
                        hbar * sp.log(1 + sp.diff(phi, x)**4 + sp.diff(phi, t)**4),
                        Lambda**4 * sp.tanh(sp.Abs(phi / M_pl))**6,
                        delta**4 * sp.exp(-sp.Abs(phi / delta)**0.5),
                        (phi / M_pl)**14 * Lambda**4 / 5040,
                    ])

                print(f"[EXPLORATION LARGE] {random.randint(3, 8)} mutations appliquées pour {random_family}")

        return base, inspiration
        
        

        # === CANDIDAT VALIDE : ANALYSE ET CLASSEMENT ===
        if USE_LLM_DURING_SIMULATION:
            print(
                f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Consultation LLM en direct..."
            )
            analysis = LLMAgent.consult(
                str(raw_formula),
                f"{inspiration}\nRecherche d'une avancée théorique potentielle.",
            )
            # Gestion timeout/indisponible
            if "indisponible" in analysis.lower() or "timeout" in analysis.lower():
                self.energy += 20  # petit bonus malgré le timeout
            elif any(word in analysis.lower() for word in positive_keywords):
                self.energy += 70
            else:
                self.energy -= 4
        else:
            print(
                f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Stocké pour analyse LLM ultérieure"
            )
            analysis = "Analyse LLM différée — Mode batch activé (évaluation après simulation)."
            self.energy += 25  # bonus d'exploration en batch

        # Score basé sur propriétés intrinsèques
        score = 1000 - (comp * 10) + (500 if sym else 0) + (300 if dim_ok else 0)
        

        # Chemins de sauvegarde
        thematic_dir = f"discoveries_thematic/{main_cat_name}/{sub_cat}"
        os.makedirs(thematic_dir, exist_ok=True)
        thematic_file = (
            f"{thematic_dir}/discovery_p{pulse}_d{self.depth}_{self.id}.json"
        )

        flat_dir = "omega_agents_logs"
        os.makedirs(flat_dir, exist_ok=True)
        flat_file = f"{flat_dir}/discovery_p{pulse}_{self.id}.json"

        # Découverte enrichie
        discovery = {
            "pulse": pulse,
            "depth": self.depth,
            "agent_id": self.id,
            "formula": str(raw_formula),
            "latex": sp.latex(raw_formula),
            "inspiration": inspiration,
            "analysis": analysis,
            "score": score,
            "family": main_cat_name,
            "subfamily": sub_cat,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Sauvegarde dans les deux systèmes (thématique + plat)
        for file_path in [thematic_file, flat_file]:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(discovery, f, indent=4, ensure_ascii=False)

        # Ajout à la liste globale (pour le rapport final)
        global_discoveries.append(discovery)

        # ==============================================================================
        # ANTI-SATURATION : évite les millions de duplicatas
        # ==============================================================================
        seen_formulas = set()  # formules déjà vues (simplifiées)
        MAX_DUPLICATES = 1     # après 1 occurrence, pénalité massive
        formula_counter = defaultdict(int)

    def run_cycle(self, pulse):
        if self.energy <= 0:
            return None

        # Régénération passive
        self.energy += 2
        if self.energy > 350:
            self.energy = 350

        raw_formula, inspiration = self.generate_candidate()

        sym, comp = PhysicalValidator.check_symmetry(raw_formula)
        dim_ok, _ = PhysicalValidator.check_dimensional_consistency(raw_formula)

        # === MÉMOIRE CENTRALE : éviter de reproduire les formules déjà jugées ===
        MEMORY_FILE = "hydra_memory.json"
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memory = json.load(f)
        else:
            memory = {"formulas": {}}

        try:
            simplified = sp.simplify(raw_formula)
            key = str(simplified)
        except:
            key = str(raw_formula)

        if key in memory["formulas"]:
            entry = memory["formulas"][key]
            count = entry.get("count", 1)
            status = entry.get("status", "unknown")

            if status == "rejected" or "aberration" in status.lower():
                self.energy -= 150
                print(f"[MÉMOIRE] Formule déjà rejetée/aberrante — pénalité massive -150 pour {self.id}")
                return None  # on l'ignore complètement
            elif count > 10:
                self.energy -= 100
                print(f"[MÉMOIRE] Formule saturée ({count} fois) — pénalité -100")
            elif count > 5:
                self.energy -= 50
                print(f"[MÉMOIRE] Formule déjà vue {count} fois — pénalité -50")
        else:
            self.energy += 30
            print(f"[MÉMOIRE] Formule totalement nouvelle — bonus +30 énergie")

        # === REJET PRÉCOCE ===
        if not (sym and comp < 40 and dim_ok):
            reason = []
            if not sym: reason.append("asymétrie")
            if comp >= 40: reason.append("trop_complexe")
            if not dim_ok: reason.append("incohérence_dimensionnelle")

            reject_file = f"omega_agents_rejected/rejected_p{pulse}_{self.id}.txt"
            with open(reject_file, "w", encoding="utf-8") as f:
                f.write(f"Formule rejetée : {raw_formula}\n")
                f.write(f"Inspiration : {inspiration}\n")
                f.write(f"Raisons : {', '.join(reason)}\n")
                f.write(f"Complexité : {comp}\n")

            self.energy -= 2
            return None
        
        # === ANTI-SATURATION : pénalité si formule déjà vue trop souvent ===
        try:
            simplified = sp.simplify(raw_formula)
            key = str(simplified)
        except:
            key = str(raw_formula)

        formula_counter[key] += 1
        count = formula_counter[key]

        if count == 1:
            self.energy += 40
            print(f"[INNOVATION] Nouvelle formule — bonus +40 énergie pour {self.id}")
        elif count <= 5:
            self.energy += 10
            print(f"[BONNE IDÉE] Formule vue {count} fois — bonus +10")
        elif count <= MAX_DUPLICATES:
            self.energy -= 30
            print(f"[RÉPÉTITION] Formule courante ({count} fois) — pénalité -30")
        else:
            self.energy -= 100
            print(f"[SATURATION] Formule épuisée ({count} fois) — pénalité -100 pour {self.id}")

        # === CANDIDAT VALIDE : ANALYSE ET CLASSEMENT ===
        if USE_LLM_DURING_SIMULATION:
            print(f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Consultation LLM en direct...")
            analysis = LLMAgent.consult(
                str(raw_formula),
                f"{inspiration}\nRecherche d'une avancée théorique potentielle."
            )
            if "indisponible" in analysis.lower() or "timeout" in analysis.lower():
                self.energy += 20
            elif any(word in analysis.lower() for word in positive_keywords):
                self.energy += 70
            else:
                self.energy -= 4
        else:
            print(f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Stocké pour analyse LLM ultérieure")
            analysis = "Analyse LLM différée — Mode batch activé (évaluation après simulation)."
            self.energy += 25

        score = 1000 - (comp * 10) + (500 if sym else 0) + (300 if dim_ok else 0)

        main_cat_name, sub_cat = self.classify_discovery_advanced(inspiration, str(raw_formula))

        thematic_dir = f"discoveries_thematic/{main_cat_name}/{sub_cat}"
        os.makedirs(thematic_dir, exist_ok=True)
        thematic_file = f"{thematic_dir}/discovery_p{pulse}_d{self.depth}_{self.id}.json"

        flat_file = f"omega_agents_logs/discovery_p{pulse}_{self.id}.json"

        discovery = {
            'pulse': pulse,
            'depth': self.depth,
            'agent_id': self.id,
            'formula': str(raw_formula),
            'latex': sp.latex(raw_formula),
            'inspiration': inspiration,
            'analysis': analysis,
            'score': score,
            'family': main_cat_name,
            'subfamily': sub_cat,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        for file_path in [thematic_file, flat_file]:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(discovery, f, indent=4, ensure_ascii=False)

        global_discoveries.append(discovery)

        return discovery



# ==============================================================================
# SIMULATION
# ==============================================================================
def start_agentic_protocol():
    load_evolution_directives()
    print("=" * 80)
    print("DÉMARRAGE DU PROTOCOLE AGENTIC OMEGA")
    print(f"Modèle local : {MODEL_NAME} | Moteur math : SymPy")
    print(f"Population initiale : 3 agents | Pulses : {PULSES}")
    print("=" * 80)

    # === CHARGEMENT DYNAMIQUE DES THÈMES DISPONIBLES ===
    thematic_base = "discoveries_thematic"
    available_families = []
    available_subfamilies = []

    if os.path.exists(thematic_base):
        for family in os.listdir(thematic_base):
            family_path = os.path.join(thematic_base, family)
            if os.path.isdir(family_path):
                available_families.append(family)
                for sub in os.listdir(family_path):
                    sub_path = os.path.join(family_path, sub)
                    if os.path.isdir(sub_path):
                        available_subfamilies.append(f"{family}/{sub}")

    print(f"[THÉMATIQUE] {len(available_families)} familles et {len(available_subfamilies)} sous-familles détectées")
    print(f"Familles : {', '.join(available_families[:10])}{'...' if len(available_families) > 10 else ''}")

    # === CRÉATION DES DOSSIERS THÉMATIQUES BASÉS SUR DISCOVERY_CATEGORIES ===
    thematic_base = "discoveries_thematic"

    subcategories = {
        "Quantum_Laws": ["Scalar_Fields", "Axion_Like", "Higgs_Mechanisms", "Relaxion"],
        "Exotic_Matter_Discoveries": [
            "Fuzzy_Dark_Matter",
            "Ultra_Light_Scalars",
            "Solitons_BosonStars",
            "Axions_Composites",
        ],
        "Quantum_Gravity": [
            "Swampland",
            "Holographic",
            "Planck_Suppressed",
            "Loop_Quantum",
            "Emergent_Entanglement",
        ],
        "Gravity_Laws": [
            "Modified_Gravity",
            "Black_Hole_Thermo",
            "Superradiance",
            "No_Hair_Violations",
        ],
        "Wormholes_Theories": [
            "Traversable",
            "Entangled_ER=EPR",
            "Multiverse_Portals",
            "Quantum_Tunnels",
        ],
        "Multiverse_Theories": [
            "Inflationary_Bubbles",
            "Everett_Branches",
            "String_Landscape",
        ],
        "DNA_Cosmic_Links": [
            "Orch_OR",
            "Fractal_Patterns",
            "Epigenetic_Scalars",
            "Panspermia_Memory",
        ],
        "Cosmology_Origins": [
            "Early_Dark_Energy",
            "Primordial_BH",
            "Phase_Transitions",
        ],
        "Quantum_Classical_Bridges": [
            "Decoherence",
            "Macro_Quantum",
            "Objective_Collapse",
        ],
        "Classical_Physics": ["Effective_Field", "Symmetries", "Dimensional_Analysis"],
        "Mathematical_Laws": ["Fractal_Spacetimes", "Non_Linear", "Symmetry_Breaking"],
        "Meta_Evolution": [
            "Self_Improving",
            "Agentic_Convergence",
            "Theoretical_Evolution",
        ],
    }

    for cat_key, cat_name in DISCOVERY_CATEGORIES.items():
        subs = subcategories.get(cat_key, ["General"])
        for sub in subs:
            os.makedirs(f"{thematic_base}/{cat_name}/{sub}", exist_ok=True)
            
    # === CRÉATION DES DOSSIERS DE BASE ===
    os.makedirs("omega_agents_rejected", exist_ok=True)
    os.makedirs("omega_agents_logs", exist_ok=True)
    os.makedirs("omega_agents_reports", exist_ok=True)

    population = [AgenticHydra() for _ in range(3)]

    # Dossier pour les non classés
    os.makedirs(f"{thematic_base}/Uncategorized/General", exist_ok=True)

    final_discoveries = []

    for p in range(1, PULSES + 1):
        new_discoveries = []

        # === PHASE DE GÉNÉRATION DES CANDIDATS ===
        for agent in population[:]:  # copie pour éviter problèmes si on ajoute pendant l'itération
            result = agent.run_cycle(p)
            if result:
                new_discoveries.append(result)
        final_discoveries.extend(new_discoveries)

        # === LOGIQUE DE CLONAGE RECALIBRÉE : ÉLITISTE ET STABLE ===
        new_agents = []
        for agent in population:
            if agent.energy > 450:  # seuil haut : seuls les élites clonent
                if random.random() < 0.7:  # 70% de chance
                    new_agents.append(AgenticHydra(depth=agent.depth + 1, energy=120))
                    agent.energy -= 40
                    print(
                        f"\n[+] CLONAGE ÉLITE ! {agent.id} (énergie {agent.energy + 40} → {agent.energy}) → nouvelle tête profondeur {agent.depth + 1}"
                    )

        # Ajout des clones
        population.extend(new_agents)

        # Mort des agents faibles
        population = [agent for agent in population if agent.energy > 30]

        # === AFFICHAGE EN TEMPS RÉEL DES STATS (à chaque pulse) ===
        current_agents = len(population)
        total_clones = sum(1 for agent in population if agent.depth > 0)
        max_depth = max((agent.depth for agent in population), default=0)
        avg_energy = sum(agent.energy for agent in population) / current_agents if current_agents > 0 else 0
        total_discoveries = len(final_discoveries)

        print(f"Cycle {p}/{PULSES} | "
              f"Agents : {current_agents:4d} | "
              f"Clones : {total_clones:4d} | "
              f"Profondeur max : {max_depth:2d} | "
              f"Énergie moy. : {avg_energy:5.1f} | "
              f"Découvertes : {total_discoveries:5d}", end="\r")


    # Fin de la simulation
    save_final_report(final_discoveries)
print("\n")  # passe à la ligne après le dernier print end="\r"

# ==============================================================================
# RAPPORT FINAL AVEC CLUSTERING
# ==============================================================================
def save_final_report(discoveries):
    if not discoveries:
        print("\n\nAucune découverte n'a passé les filtres.")
        return

    # === CLUSTERING ===
    clusters = defaultdict(list)
    for d in discoveries:
        try:
            simplified = sp.simplify(sp.parse_expr(d["formula"]))
            key = str(simplified)
        except:
            key = d["formula"]
        clusters[key].append(d)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"omega_agents_reports/Rapport_Omega_{timestamp}.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# RAPPORT DE CONVERGENCE AGENTIC OMEGA\n\n")
        f.write(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(
            f"Modèle : {MODEL_NAME} | Découvertes validées : {len(discoveries)}\n\n"
        )

        f.write("## Top 10 candidats individuels\n\n")
        for i, d in enumerate(
            sorted(discoveries, key=lambda x: x["score"], reverse=True)[:10]
        ):
            f.write(
                f"### #{i+1} | Score : {d['score']:.0f} | Profondeur : {d['depth']}\n\n"
            )
            f.write(f"**Formule** : ${d['latex']}$\n\n")
            f.write(f"**Analyse du Critique Théorique** :\n\n{d['analysis']}\n\n")
            f.write("---\n\n")
            f.write(f"**Inspiration** : {d['inspiration']}\n\n")

        # === CLUSTERING SECTION ===
        if len(clusters) > 1:
            f.write("## Clusters de conjectures similaires détectés\n\n")
            for i, (key, cluster) in enumerate(clusters.items()):
                if len(cluster) > 1:
                    best = max(cluster, key=lambda x: x["score"])
                    f.write(
                        f"### Cluster #{i+1} — {len(cluster)} candidats similaires\n\n"
                    )
                    f.write(f"Forme dominante : ${best['latex']}$\n\n")
                    f.write(f"Meilleur score : {best['score']:.0f}\n")
                    f.write(f"Analyse représentative :\n\n{best['analysis']}\n\n")
                    f.write("---\n\n")

    print(f"\n\n[TERMINE] Rapport généré → {path}")
    print("L'hydre et son critique ont parlé.")


if __name__ == "__main__":
    start_agentic_protocol()
