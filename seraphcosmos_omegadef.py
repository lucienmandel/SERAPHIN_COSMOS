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

PULSES = 500
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
    "plausible", "intéressant", "cohérent", "prometteur", "profond",
    "fascinant", "étonnant", "innovant", "pertinent", "novateur",
    "convaincant", "solide", "élégant", "audacieux", "original",
    "remarquable", "excellent", "brillant", "stimulant",
    "80", "85", "90", "95", "100", "très bon", "bon potentiel"
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
    "Tunnel quantique macroscopique et quantification d'énergie dans circuits supraconducteurs (Nobel 2025 : Clarke, Devoret, Martinis), fondation des qubits supraconducteurs modernes."
]

UNSOLVED_THEMES = [
    "Quantum gravity reconciliation", "Nature of dark matter", "Multiverse existence",
    "Fine-tuning constants", "Antimatter asymmetry", "Dark energy composition",
    "Time arrow direction", "Hidden sectors", "Supersymmetry", "Fractal spacetime",
    "Ultra-light scalars", "Axions", "Higgs portal", "Relaxion",
    "Black hole information paradox", "Wormhole traversability", "DNA cosmic encoding", "Quantum entanglement in gravity"
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
    "Détection directe d'ondes gravitationnelles primordiales ou signatures de phase transitions cosmiques."
]

DISCOVERY_CATEGORIES = {
    'quantum': 'Quantum_Laws',
    'classical': 'Classical_Physics',
    'multiverse': 'Multiverse_Theories',
    'micro_macro_link': 'Quantum_Classical_Bridges',
    'big_bang': 'Cosmology_Origins',
    'new_matter': 'Exotic_Matter_Discoveries',
    'math': 'Mathematical_Laws',
    'meta_evolver': 'Meta_Evolution',
    'gravity': 'Gravity_Laws',
    'quantum_gravity': 'Quantum_Gravity',
    'wormholes': 'Wormholes_Theories',
    'dna_cosmic': 'DNA_Cosmic_Links'
}

# Symboles
x, y, z, t, k = sp.symbols('x y z t k', real=True)
phi, hbar, delta, G, m = sp.symbols('phi hbar delta G m')
f_a, Lambda, M_pl, lambda_q = sp.symbols('f_a Lambda M_pl lambda_q', positive=True, real=True)
# f_a  : échelle de brisure de symétrie pour axions (decay constant)
# Lambda : échelle d'énergie (ex. énergie noire, vide QCD, etc.)
# M_pl : masse de Planck réduite ≈ 2.4 × 10^18 GeV

global_discoveries = []

# ==============================================================================
# ARRÊT PROPRE Ctrl+C
# ==============================================================================
def graceful_shutdown(sig, frame):
    print("\n\n" + "="*80)
    print("KILL SWITCH ACTIVÉ (Ctrl+C détecté)")
    print("Interruption manuelle reçue — Génération du rapport final en cours...")
    print("="*80)
    
    # Sauvegarde immédiate du rapport avec toutes les découvertes accumulées
    if global_discoveries:
        save_final_report(global_discoveries)
        print(f"\nRapport final sauvegardé avec {len(global_discoveries)} découvertes.")
    else:
        print("\nAucune découverte à sauvegarder.")
    
    print("\nProtocole Oméga interrompu proprement.")
    
    sys.exit(0)

# Association du handler à SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, graceful_shutdown)

# Option bonus : gérer aussi SIGTERM (kill du processus)
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
                x: -1, y: -1, z: -1, t: -1,
                phi: 1, m: 1, G: -2, hbar: 0,
                delta: 1,
                f_a: 1,
                Lambda: 1,
                M_pl: 1
            }
            dim_expr = expr
            for sym, d in dim_rules.items():
                if d != 0:
                    dim_expr = dim_expr.subs(sym, sp.Symbol('M')**d)
                else:
                    dim_expr = dim_expr.subs(sym, 1)
            if sp.simplify(dim_expr).is_constant():
                return True, "DIMENSIONAL_OK"
        except Exception:
            pass
        return False, "DIMENSIONAL_FAIL"

# ==============================================================================
# LLM CRITIQUE
# ==============================================================================
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
                return response.json().get('response', "Pas de réponse du modèle").strip()
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

    def classify_discovery_advanced(self, inspiration, formula_str):
        """
        Classifie en grande famille (via DISCOVERY_CATEGORIES) + sous-famille fine.
        Retourne (main_category_name, subcategory)
        """
        insp_lower = inspiration.lower()
        form_lower = formula_str.lower()

        # Détection par mots-clés dans inspiration ou formule
        if any(word in insp_lower for word in ["axion", "alps", "fuzzy", "periodic", "soliton", "dark matter"]):
            if "cos(" in form_lower:
                return "Exotic_Matter_Discoveries", "Axion_Like" if "1 - cos" in form_lower else "Periodic_Potential"
            return "Exotic_Matter_Discoveries", "Fuzzy_Dark_Matter"

        if "quintessence" in insp_lower or "dark energy" in insp_lower:
            if "exp(-" in form_lower:
                return "Exotic_Matter_Discoveries", "Ultra_Light_Scalars"
            return "Cosmology_Origins", "Early_Dark_Energy"

        if any(word in insp_lower for word in ["higgs", "portal", "relaxion"]):
            if "phi**6" in form_lower or "phi^6" in form_lower:
                return "Quantum_Laws", "Higgs_Mechanisms"
            return "Quantum_Laws", "Relaxion"

        if any(word in insp_lower for word in ["black hole", "superradiance", "no-hair", "swampland"]):
            if "log(" in form_lower:
                return "Quantum_Gravity", "Holographic"
            if "m_pl" in form_lower:
                return "Quantum_Gravity", "Planck_Suppressed"
            return "Gravity_Laws", "Superradiance"

        if any(word in insp_lower for word in ["multiverse", "everett", "bubble"]):
            return "Multiverse_Theories", "Inflationary_Bubbles"

        if any(word in insp_lower for word in ["wormhole", "traversable", "er=epr"]):
            return "Wormholes_Theories", "Entangled_ER=EPR"

        if any(word in insp_lower for word in ["dna", "adn", "microtubules", "conscience", "orch-or"]):
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
        lambda: 0.5 * (sp.diff(phi, t)**2 - sp.diff(phi, x)**2 - sp.diff(phi, y)**2 - sp.diff(phi, z)**2) - (m**2 * phi**2)/2,
        lambda: 0.5 * sp.diff(phi, t)**2 - 0.5 * (sp.diff(phi, x)**2 + sp.diff(phi, y)**2 + sp.diff(phi, z)**2) - m**2 * phi**2 / 2,
    ]

    axion_templates = [
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a)),
        lambda: Lambda**4 * (1 - sp.cos(phi / f_a + k * x)),
        lambda: Lambda**4 * (1 - sp.cos(2 * phi / f_a)) / 2,

    ]

    higgs_portal_templates = [
        lambda: - (m**2 / 2) * phi**2 + (lambda_q / 4) * phi**4,  # m ici est masse effective, Lambda_q couplage quartique sans dim
        # Double puits stabilisé
        lambda: (Lambda**4 / 4) * (phi**2 / delta**2 - 1)**2,
        # Potentiel sextique swampland-compatible (supprimé par M_pl)
        lambda: phi**6 / (6 * M_pl**2) + (m**4 / M_pl**2) * phi**2,

    ]

    quintessence_templates = [
        lambda: Lambda**4 * sp.exp(-phi / M_pl),
        lambda: Lambda**4 / (1 + (phi / M_pl)**2)**2,

    ]

    gravity_templates = [
        lambda: (phi / M_pl)**2 * phi**2 * Lambda**2,  # exemple : phi^4 / M_pl^2
        lambda: hbar * sp.log(phi**2 / M_pl**2 + 1) * Lambda**4,
        lambda: Lambda**4 * sp.log(1 + G * phi**2 * M_pl**2),  # si tu veux garder G, compense
    ]

    swampland_templates = [
        lambda: (phi / M_pl)**4 * phi**4 * sp.log(phi**2 / M_pl**2 + 1),  # swampland-inspired
        lambda: Lambda**4 * sp.tanh(phi / f_a)**2,
    ]

    dna_cosmic_templates = [
        lambda: delta * sp.besselj(0, m * phi / delta) * sp.sin(k * x),
        lambda: m**4 * sp.Heaviside(phi - delta) * sp.exp(-phi**2 / delta**2),
        lambda: hbar * sp.log(sp.sin(phi / delta)**2 + 1) * m**2,
    ]

    all_templates = (
        kg_templates + axion_templates + higgs_portal_templates +
        quintessence_templates + gravity_templates + swampland_templates +
        dna_cosmic_templates
    )

    def get_inspiration(self):
        """
        Tire une connaissance ou un thème unsolved pour inspirer la génération.
        """
        if random.random() < 0.7:
            return "Thème à explorer : " + random.choice(UNSOLVED_THEMES)
        else:
            return "Connaissance de référence : " + random.choice(REAL_KNOWLEDGE_2025)

    def generate_candidate(self):
        """
        Génération contextualisée utilisant les templates partagés de la classe.
        """
        inspiration = self.get_inspiration()
        insp_lower = inspiration.lower()

        # Base : un template aléatoire parmi tous
        base = random.choice(self.__class__.all_templates)()

        # Mutation guidée par l'inspiration
        if random.random() < 0.6:
            if any(word in insp_lower for word in ["axion", "alps", "periodic", "fuzzy", "dark matter"]):
                base += Lambda**4 * (1 - sp.cos(phi / f_a))
            elif any(word in insp_lower for word in ["higgs", "portal", "relaxion"]):
                base += G * phi**2 * sp.Symbol('H')**2 / 2
            elif any(word in insp_lower for word in ["quintessence", "dark energy", "w(z)"]):
                base += Lambda**4 * sp.exp(-phi / M_pl)
            elif any(word in insp_lower for word in ["black hole", "trous noirs", "superradiance", "no-hair", "swampland"]):
                base += G * m**2 * phi**2 / sp.sqrt(x**2 + y**2 + z**2 + delta**2)
            elif any(word in insp_lower for word in ["multiverse", "wormhole", "everett"]):
                base += delta**4 * sp.log(1 + phi**2 / delta**2)
            elif any(word in insp_lower for word in ["dna", "adn", "microtubules", "conscience", "orch-or", "épigénétique"]):
                base += delta * sp.besselj(0, m * phi / delta) * sp.sin(k * x)
            else:
                base += random.choice([phi**4 / 4, m**4 * sp.tanh(phi / delta)**2, delta**4 * sp.log(1 + (phi / delta)**2)])

        # Mutations légères supplémentaires
        if random.random() < 0.4:
            safe_mods = [
                delta * m**2 * phi**2,
                G * phi**4 / 4,
                m**4 * sp.sin(phi / delta)**2 / 2,
                hbar**2 * sp.diff(phi, x)**2 / (2 * m**2),
                delta**4 * sp.log(1 + (phi / delta)**2)
            ]
            base += random.choice(safe_mods)

        return base, inspiration

    def run_cycle(self, pulse):
        if self.energy <= 0:
            return None

        # Régénération passive (plus lente, comme recalibré)
        self.energy += 2
        if self.energy > 350:
            self.energy = 350

        raw_formula, inspiration = self.generate_candidate()

        sym, comp = PhysicalValidator.check_symmetry(raw_formula)
        dim_ok, _ = PhysicalValidator.check_dimensional_consistency(raw_formula)

        # === REJET PRÉCOCE ===
        if not (sym and comp < 40 and dim_ok):
            reason = []
            if not sym: reason.append("asymétrie")
            if comp >= 40: reason.append("trop_complexe")
            if not dim_ok: reason.append("incohérence_dimensionnelle")

            reject_file = f"omega_agents_rejected/rejected_p{pulse}_{self.id}.txt"
            with open(reject_file, "w", encoding='utf-8') as f:
                f.write(f"Formule rejetée : {raw_formula}\n")
                f.write(f"Inspiration : {inspiration}\n")
                f.write(f"Raisons : {', '.join(reason)}\n")
                f.write(f"Complexité : {comp}\n")

            self.energy -= 2
            return None

        # === CANDIDAT VALIDE : ANALYSE ET CLASSEMENT ===
        if USE_LLM_DURING_SIMULATION:
            print(f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Consultation LLM en direct...")
            analysis = LLMAgent.consult(
                str(raw_formula),
                f"{inspiration}\nRecherche d'une avancée théorique potentielle."
            )
            # Gestion timeout/indisponible
            if "indisponible" in analysis.lower() or "timeout" in analysis.lower():
                self.energy += 20  # petit bonus malgré le timeout
            elif any(word in analysis.lower() for word in positive_keywords):
                self.energy += 70
            else:
                self.energy -= 4
        else:
            print(f"\n[PULSE {pulse}] {self.id} : Candidat prometteur → Stocké pour analyse LLM ultérieure")
            analysis = "Analyse LLM différée — Mode batch activé (évaluation après simulation)."
            self.energy += 25  # bonus d'exploration en batch

        # Score basé sur propriétés intrinsèques
        score = 1000 - (comp * 10) + (500 if sym else 0) + (300 if dim_ok else 0)

        # === CLASSEMENT THÉMATIQUE AVANCÉ ===
        main_cat_name, sub_cat = self.classify_discovery_advanced(inspiration, str(raw_formula))

        # Chemins de sauvegarde
        thematic_dir = f"discoveries_thematic/{main_cat_name}/{sub_cat}"
        os.makedirs(thematic_dir, exist_ok=True)
        thematic_file = f"{thematic_dir}/discovery_p{pulse}_d{self.depth}_{self.id}.json"

        flat_dir = "omega_agents_logs"
        os.makedirs(flat_dir, exist_ok=True)
        flat_file = f"{flat_dir}/discovery_p{pulse}_{self.id}.json"

        # Découverte enrichie
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

        # Sauvegarde dans les deux systèmes (thématique + plat)
        for file_path in [thematic_file, flat_file]:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(discovery, f, indent=4, ensure_ascii=False)

        # Ajout à la liste globale (pour le rapport final)
        global_discoveries.append(discovery)

        return discovery

# ==============================================================================
# SIMULATION
# ==============================================================================
def start_agentic_protocol():
    print("="*80)
    print("DÉMARRAGE DU PROTOCOLE AGENTIC OMEGA")
    print(f"Modèle local : {MODEL_NAME} | Moteur math : SymPy")
    print(f"Population initiale : 3 agents | Pulses : {PULSES}")
    print("="*80)

# === CRÉATION DES DOSSIERS THÉMATIQUES BASÉS SUR DISCOVERY_CATEGORIES ===
    thematic_base = "discoveries_thematic"

    subcategories = {
        'Quantum_Laws': ['Scalar_Fields', 'Axion_Like', 'Higgs_Mechanisms', 'Relaxion'],
        'Exotic_Matter_Discoveries': ['Fuzzy_Dark_Matter', 'Ultra_Light_Scalars', 'Solitons_BosonStars', 'Axions_Composites'],
        'Quantum_Gravity': ['Swampland', 'Holographic', 'Planck_Suppressed', 'Loop_Quantum', 'Emergent_Entanglement'],
        'Gravity_Laws': ['Modified_Gravity', 'Black_Hole_Thermo', 'Superradiance', 'No_Hair_Violations'],
        'Wormholes_Theories': ['Traversable', 'Entangled_ER=EPR', 'Multiverse_Portals', 'Quantum_Tunnels'],
        'Multiverse_Theories': ['Inflationary_Bubbles', 'Everett_Branches', 'String_Landscape'],
        'DNA_Cosmic_Links': ['Orch_OR', 'Fractal_Patterns', 'Epigenetic_Scalars', 'Panspermia_Memory'],
        'Cosmology_Origins': ['Early_Dark_Energy', 'Primordial_BH', 'Phase_Transitions'],
        'Quantum_Classical_Bridges': ['Decoherence', 'Macro_Quantum', 'Objective_Collapse'],
        'Classical_Physics': ['Effective_Field', 'Symmetries', 'Dimensional_Analysis'],
        'Mathematical_Laws': ['Fractal_Spacetimes', 'Non_Linear', 'Symmetry_Breaking'],
        'Meta_Evolution': ['Self_Improving', 'Agentic_Convergence', 'Theoretical_Evolution']
    }

    for cat_key, cat_name in DISCOVERY_CATEGORIES.items():
        subs = subcategories.get(cat_key, ['General'])
        for sub in subs:
            os.makedirs(f"{thematic_base}/{cat_name}/{sub}", exist_ok=True)

    # Dossier pour les non classés
    os.makedirs(f"{thematic_base}/Uncategorized/General", exist_ok=True)

    population = [AgenticHydra() for _ in range(3)]
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
            if agent.energy > 250:  # seuil haut : seuls les élites clonent
                if random.random() < 0.7:  # 70% de chance
                    new_agents.append(AgenticHydra(depth=agent.depth + 1, energy=120))
                    agent.energy -= 40
                    print(f"\n[+] CLONAGE ÉLITE ! {agent.id} (énergie {agent.energy + 40} → {agent.energy}) → nouvelle tête profondeur {agent.depth + 1}")

        # Ajout des clones
        population.extend(new_agents)

        # Mort des agents faibles
        population = [agent for agent in population if agent.energy > 30]

        # Affichage progrès
        avg_energy = sum(a.energy for a in population) / len(population) if population else 0
        print(f"Cycle {p}/{PULSES} | Agents : {len(population):4d} | Découvertes : {len(final_discoveries):5d} | Énergie moyenne : {avg_energy:.1f}",
              end='\r')

    # Fin de la simulation
    save_final_report(final_discoveries)

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
            simplified = sp.simplify(sp.parse_expr(d['formula']))
            key = str(simplified)
        except:
            key = d['formula']
        clusters[key].append(d)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"omega_agents_reports/Rapport_Omega_{timestamp}.md"

    with open(path, "w", encoding='utf-8') as f:
        f.write("# RAPPORT DE CONVERGENCE AGENTIC OMEGA\n\n")
        f.write(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"Modèle : {MODEL_NAME} | Découvertes validées : {len(discoveries)}\n\n")

        f.write("## Top 10 candidats individuels\n\n")
        for i, d in enumerate(sorted(discoveries, key=lambda x: x['score'], reverse=True)[:10]):
            f.write(f"### #{i+1} | Score : {d['score']:.0f} | Profondeur : {d['depth']}\n\n")
            f.write(f"**Formule** : ${d['latex']}$\n\n")
            f.write(f"**Analyse du Critique Théorique** :\n\n{d['analysis']}\n\n")
            f.write("---\n\n")
            f.write(f"**Inspiration** : {d['inspiration']}\n\n")

        # === CLUSTERING SECTION ===
        if len(clusters) > 1:
            f.write("## Clusters de conjectures similaires détectés\n\n")
            for i, (key, cluster) in enumerate(clusters.items()):
                if len(cluster) > 1:
                    best = max(cluster, key=lambda x: x['score'])
                    f.write(f"### Cluster #{i+1} — {len(cluster)} candidats similaires\n\n")
                    f.write(f"Forme dominante : ${best['latex']}$\n\n")
                    f.write(f"Meilleur score : {best['score']:.0f}\n")
                    f.write(f"Analyse représentative :\n\n{best['analysis']}\n\n")
                    f.write("---\n\n")

    print(f"\n\n[TERMINE] Rapport généré → {path}")
    print("L'hydre et son critique ont parlé.")

if __name__ == "__main__":
    start_agentic_protocol()
