import os
import random
import json
import sympy as sp
from constants_config import DIM_MAP, ALL_SYMBOLS, THEME_CONSTANTS_MAP, DISCOVERY_CATEGORIES, SUBCATEGORIES
from knowledge_base import REAL_KNOWLEDGE_2025, UNSOLVED_THEMES
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import logging
import sys
sys.dont_write_bytecode = True


# Configuration du logging pro
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - [OMEGA HYDRA] - %(message)s')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PULSES = 140
CLONE_THRESHOLD = 240         # Seuil énergétique pour cloner
CLONE_PROBA_BASE = 0.45        # Probabilité de clonage
CLONE_ENERGY_SHARE = 0.55      # Part d'énergie donnée au clone
MEMORY_FILE = "hydra_memory.json"
DIRECTIVES_FILE = "hydra_evolution_directives.json"
EXPERT_POOL_FILE = "expert_pool.json"
HYDRA_MEMORY = {
    "formulas": {},  # Contiendra tes 45 000 formules existantes
    "stats": {
        "concept_scores": {},    # Pour le choix des Niches
        "term_patterns": {},     # Pour l'analyse de ce qui "paye"
        "category_scores": {},   # Pour l'évolution globale
        "bad_patterns": {}       # Pour apprendre des échecs
    }
}

COMPLEX_OPS = [
    "sp.besselj(0, {x})",           # Vibrations et modes propres
    "sp.assoc_legendre(1, 1, {x})",  # Symétrie sphérique
    "sp.LambertW({x})",              # Équilibres transcendants
    "sp.erf({x})",                   # Probabilités et dissipation
    "sp.gamma({x})",                 # Théorie des cordes / Régularisation
    "sp.airyai({x})",                # Optique physique
    "sp.zeta({x})",                  # Physique statistique
    "sp.grad_sq({x})"                # Énergie cinétique du champ
]


# ==============================================================================
# UTILITAIRES I/O
# ==============================================================================
def load_expert_pool():
    if os.path.exists(EXPERT_POOL_FILE):
        with open(EXPERT_POOL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

POOL_DATA = load_expert_pool()

def load_json_safe(path, default=None):
    if default is None: default = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Erreur chargement {path}: {e}")
            return default
    return default

def save_json_safe(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Erreur sauvegarde {path}: {e}")

def load_evolution_directives():
    return load_json_safe(DIRECTIVES_FILE, {
        "priority_domains": [],
        "complexity_bias": 0
    })

# ==============================================================================
# MÉMOIRE ÉVOLUTIVE PERSISTANTE (HYDRA MEMORY)
# ==============================================================================
def load_hydra_memory():
    global HYDRA_MEMORY
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # On récupère les stats si elles existent, sinon on initialise
                # C'est ici que le KeyError est évité
                stats = data.get("stats", {})
                
                HYDRA_MEMORY["concept_scores"].update(stats.get("concept_scores", {}))
                HYDRA_MEMORY["category_scores"].update(stats.get("category_scores", {}))
                HYDRA_MEMORY["term_patterns"].update(stats.get("term_patterns", {}))
                HYDRA_MEMORY["bad_patterns"].update(stats.get("bad_patterns", {}))
                
                logging.info(f"🧠 Mémoire chargée avec succès ({len(data.get('formulas', {}))} formules connues)")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la mémoire : {e}")
            
def weighted_choice_from_dict(score_dict):
    if not score_dict:
        return None
    total = sum(max(v, 0.1) for v in score_dict.values())
    r = random.uniform(0, total)
    acc = 0.0
    for k, v in score_dict.items():
        acc += max(v, 0.1)
        if acc >= r:
            return k
    return random.choice(list(score_dict.keys()))

def update_hydra_memory(discovery, reward):
    global HYDRA_MEMORY
    
    # 1. Sécurité : Initialisation des clés si elles manquent
    if "formulas" not in HYDRA_MEMORY:
        HYDRA_MEMORY["formulas"] = {}
    if "stats" not in HYDRA_MEMORY:
        HYDRA_MEMORY["stats"] = {
            "concept_scores": {},
            "category_scores": {},
            "term_patterns": {},
            "bad_patterns": {}
        }

    formula = discovery.get("formula", "")
    inspiration = discovery.get("inspiration", "Unknown")
    
    # 2. Mise à jour des stats (On range bien dans ["stats"])
    # Utilisation de .get() pour éviter le KeyError sur les dictionnaires internes
    HYDRA_MEMORY["stats"]["concept_scores"][inspiration] = HYDRA_MEMORY["stats"]["concept_scores"].get(inspiration, 0) + reward
    
    patterns = ["sin", "cos", "exp", "log", "sqrt", "tanh", "Abs", "M_p", "G", "Lambda"]
    for p in patterns:
        if p in formula:
            current_val = HYDRA_MEMORY["stats"]["term_patterns"].get(p, 0)
            HYDRA_MEMORY["stats"]["term_patterns"][p] = current_val + 1

    # 3. Enregistrement de la formule à la RACINE (Comme ton exemple)
    HYDRA_MEMORY["formulas"][formula] = {
        "formula": formula,
        "inspiration": inspiration,
        "depth": discovery.get("depth", 0),
        "agent_id": discovery.get("agent_id", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "score": reward,
        "score_breakdown": discovery.get("score_breakdown", {}), # On garde le détail
        "prefilter_status": "excellente" if reward > 9000 else "intéressante",
        "learning_potential": "strong_positive" if reward > 9000 else "mixed"
    }


# ==============================================================================
# CLASSE MAÎTRESSE : AGENTIC HYDRA (TYPE III)
# ==============================================================================
class AgenticHydra:
    """
    Agent autonome de découverte mathématique/physique.
    Utilise une approche évolutive multi-couches (5-15 mutations) pour générer
    des conjectures hybrides (Quantum Gravity, Bio-Physics, Cosmology).
    """

    # Mémoire collective de l'espèce
    thematic_base = "discoveries_thematic"
    available_families = []
    available_subfamilies = []
    supervisor_templates = []
    global_registry = set()

    @classmethod
    def load_collective_memory(cls):
        """Initialisation de la connaissance globale de l'Hydre."""
        # 1. Cartographie des fichiers existants
        cls.available_families = []
        cls.available_subfamilies = []
        
        if os.path.exists(cls.thematic_base):
            for family in os.listdir(cls.thematic_base):
                family_path = os.path.join(cls.thematic_base, family)
                if os.path.isdir(family_path):
                    cls.available_families.append(family)
                    for sub in os.listdir(family_path):
                        cls.available_subfamilies.append(f"{family}/{sub}")

        # 2. Chargement des templates de base (Superviseur)
        # On assure un pool minimal pour l'hybridation
        phi = ALL_SYMBOLS.get('phi', sp.Symbol('phi'))
        M_p = ALL_SYMBOLS.get('M_p', sp.Symbol('M_p'))
        cls.supervisor_templates = [
            lambda: sp.exp(-phi**2),
            lambda: sp.tanh(phi),
            lambda: sp.log(1 + phi**2),
            lambda: sp.sin(phi)**2
        ]
        
        logging.info(f"Mémoire collective chargée : {len(cls.available_families)} familles, {len(cls.available_subfamilies)} sous-thèmes.")

    def __init__(self, depth=0, energy=100, last_score=0):
        self.depth = depth
        self.energy = energy
        self.last_score = last_score 
        self.directives = ""
        self.agent_id = f"AGENT_{random.randint(10000, 99999)}"

    def choose_niche_inspiration(self):
        """Identifie et choisit un thème sous-exploré par l'élite."""
        global HYDRA_MEMORY
        
        # 1. Lister tous les thèmes possibles (tes constantes existantes)
        all_possible_themes = REAL_KNOWLEDGE_2025 + UNSOLVED_THEMES
        
        # 2. Compter les thèmes déjà présents dans les scores > 9000
        theme_usage = {}
        for data in HYDRA_MEMORY["formulas"].values():
            if data.get("score", 0) >= 9000:
                theme = data.get("inspiration")
                theme_usage[theme] = theme_usage.get(theme, 0) + 1
                
        # 3. Trouver les niches (thèmes avec 0 ou très peu d'entrées)
        niches = [t for t in all_possible_themes if theme_usage.get(t, 0) == 0]
        
        # 4. Si une niche existe, on a 70% de chance de s'y installer
        if niches and random.random() < 0.70:
            niche_choice = random.choice(niches)
            return niche_choice
            
        # Sinon, retour à un choix aléatoire classique
        return random.choice(all_possible_themes)
            
    def get_smart_inspiration(self):
        r = random.random()

        if r < 0.4 and UNSOLVED_THEMES:
            return "Défi fondamental : " + random.choice(UNSOLVED_THEMES)

        elif r < 0.7 and REAL_KNOWLEDGE_2025:
            return "Hybride connaissance : " + random.choice(REAL_KNOWLEDGE_2025)

        else:
            if POOL_DATA:
                # Au lieu de chercher une clé précise qui peut varier
                stats = HYDRA_MEMORY.get("stats", {})

                # On essaie de récupérer les scores de catégories, sinon on prend un dico vide
                cat_dict = stats.get("category_scores", {})

                if not cat_dict:
                    # Si on n'a pas encore de stats, on choisit une inspiration au hasard 
                    # dans ta liste de thèmes pour ne pas bloquer l'agent
                    inspiration = random.choice(REAL_KNOWLEDGE_2025 + UNSOLVED_THEMES)
                    return inspiration
                else:
                    # Si on a des stats, on utilise ton système de choix pondéré
                    return weighted_choice_from_dict(cat_dict)

    def get_random_term_from_category(self, category_key):
        """Version OMNISCIENTE : Fusionne le JSON Expert et la logique SymPy."""
        # 1. Récupération des Symboles
        phi = ALL_SYMBOLS.get('phi', sp.Symbol('phi'))
        M_pl = ALL_SYMBOLS.get('M_p', sp.Symbol('M_p'))
        delta = ALL_SYMBOLS.get('delta', sp.Symbol('delta'))
        Lambda = ALL_SYMBOLS.get('Lambda', sp.Symbol('Lambda'))
        m = ALL_SYMBOLS.get('m', sp.Symbol('m'))
        hbar = ALL_SYMBOLS.get('hbar', sp.Symbol('hbar'))
        G = ALL_SYMBOLS.get('G', sp.Symbol('G'))
        t = ALL_SYMBOLS.get('t', sp.Symbol('t'))
        
        # 2. On récupère un concept d'expert lié à la catégorie (via POOL_DATA chargé du JSON)
        # Si la catégorie est "Gravity_Laws", on cherche dans "Extreme_Physics" ou "Non_Euclidean_Geometry"
        pool_category = "Extreme_Physics"
        if "Biology" in category_key or "DNA" in category_key: pool_category = "Biology_and_Systems"
        elif "Quantum" in category_key: pool_category = "Macro_Quantum_Phenomena"
        elif "Math" in category_key: pool_category = "Advanced_Mathematics"
        
        # Extraction du concept littéral (ex: "Calabi-Yau Folding")
        concept = random.choice(POOL_DATA.get(pool_category, ["Fundamental Field"]))
        
        # 3. MOTEUR DE TRADUCTION (Du concept vers l'équation)
        # C'est ici que la puissance augmente : on crée des structures basées sur le NOM du concept
        
        # --- STRUCTURES GÉOMÉTRIQUES (Folding, Curvature, Topology) ---
        if any(w in concept for w in ["Folding", "Curvature", "Topology", "Metric"]):
            return random.choice([
                sp.sin(phi/M_pl)**2 * sp.exp(-phi/delta), 
                G * m / (sp.Abs(phi) + delta),
                sp.log(1 + sp.Abs(sp.diff(phi, t))**2) if hasattr(phi, 'diff') else sp.log(1+phi**2)
            ])

        # --- STRUCTURES ONDULATOIRES (Resonance, Vibration, Frequency) ---
        elif any(w in concept for w in ["Resonance", "Vibration", "Wave", "Schumann"]):
            k = sp.Symbol('k')
            return sp.sin(k * phi) * sp.exp(-t/delta) * (hbar/m)

        # --- STRUCTURES D'INFORMATION (Holographic, Entropy, Complexity) ---
        elif any(w in concept for w in ["Holographic", "Entropy", "Information", "Bekenstein"]):
            return random.choice([
                2 * sp.pi * sp.Abs(phi/M_pl)**2, # Area law
                -phi * sp.log(sp.Abs(phi) + 1e-9), # Shannon-like
                delta**2 / (phi**2 + M_pl**2)
            ])

        # --- STRUCTURES BIOLOGIQUES (Morphogenesis, Phyllotaxis, Population) ---
        elif any(w in concept for w in ["Turing", "Morphogenesis", "Phyllotaxis", "Organization"]):
            phi_gold = 1.6180339
            return delta * sp.besselj(0, phi/phi_gold) * sp.cos(phi)

        # 4. FALLBACK : Si le concept est trop exotique, on crée une hybridation pure
        return (phi/M_pl)**random.randint(1, 4) * Lambda**2
    
    def generate_candidate(self):
        inspiration = self.get_smart_inspiration()
        insp_lower = inspiration.lower()

        # 1. INITIALISATION DES SYMBOLES LOCAUX (Pour éviter les NameError)
        # On mappe les variables du snippet vers les symboles SymPy réels
        phi = ALL_SYMBOLS.get('phi', sp.Symbol('phi'))
        x = ALL_SYMBOLS.get('x', sp.Symbol('x'))
        y = ALL_SYMBOLS.get('y', sp.Symbol('y'))
        z = ALL_SYMBOLS.get('z', sp.Symbol('z'))
        t = ALL_SYMBOLS.get('t', sp.Symbol('t'))
        
        # Constantes physiques
        M_pl = ALL_SYMBOLS.get('M_p', sp.Symbol('M_p')) # Planck Mass
        Lambda = ALL_SYMBOLS.get('Lambda', sp.Symbol('Lambda')) # Cosmo constant
        f_a = ALL_SYMBOLS.get('f_a', sp.Symbol('f_a')) # Axion decay constant
        G = ALL_SYMBOLS.get('G', sp.Symbol('G'))
        hbar = ALL_SYMBOLS.get('hbar', sp.Symbol('hbar'))
        c = ALL_SYMBOLS.get('c', sp.Symbol('c'))
        k_B = ALL_SYMBOLS.get('k_B', sp.Symbol('k_B'))
        
        # Paramètres divers
        delta = ALL_SYMBOLS.get('l_p', sp.Symbol('delta')) # Souvent Planck length ou scale
        m = ALL_SYMBOLS.get('m', sp.Symbol('m'))
        k = ALL_SYMBOLS.get('k', sp.Symbol('k'))
        omega = ALL_SYMBOLS.get('omega', sp.Symbol('omega'))
        lambda_q = sp.Symbol('lambda_q') # Quartic coupling

        num_mutations = random.randint(5, 15)        
        base = 0
        
        for _ in range(num_mutations):
            term = 0
            
            # --- LOGIQUE DE SÉLECTION BASÉE SUR LE SAVOIR ---
            if any(word in insp_lower for word in ["scalar", "fuzzy", "dark matter"]):
                # L'agent "sait" qu'il doit utiliser des champs scalaires
                term = random.choice([Lambda**4 * sp.exp(-phi/M_pl), m**2 * phi**2])
                
            elif any(word in insp_lower for word in ["dna", "biological", "brain"]):
                # L'agent s'oriente vers la biologie quantique
                term = random.choice([delta * sp.besselj(0, phi/delta), hbar * sp.log(phi)])
            
            # === 1. SÉLECTION DU TERME (TES TEMPLATES DÉTAILLÉS) ===
            if random.random() < 0.20:
                cat_aleatoire = random.choice(list(SUBCATEGORIES.keys()))
                # L'agent pioche un terme "sauvage" hors de sa zone de confort
                term = self.get_random_term_from_category(cat_aleatoire)
                
                # --- BRANCHE A : MUTATION GUIDÉE PAR L'INSPIRATION ---
                
                # AXIONS / DARK MATTER
                if any(word in insp_lower for word in ["axion", "alps", "fuzzy", "dark matter"]):
                    term = random.choice([
                        Lambda**4 * (1 - sp.cos(phi / f_a)),
                        Lambda**4 * (1 - sp.cos(2 * phi / f_a)) / 2,
                        Lambda**4 * sp.sin(phi / f_a)**2,
                        Lambda**4 * (1 - sp.cos(3 * phi / f_a)) / 3,
                        m**2 * phi**2 * sp.cos(omega * t)
                    ])

                # DARK ENERGY / QUINTESSENCE
                elif any(word in insp_lower for word in ["quintessence", "dark energy", "w(z)"]):
                    term = random.choice([
                        Lambda**4 * sp.exp(-phi / M_pl),
                        Lambda**4 / (1 + (phi / M_pl)**2)**2,
                        Lambda**4 * sp.sech(phi / M_pl)**2,
                        Lambda**4 * sp.exp(-2 * phi / M_pl),
                        (phi / M_pl)**2 * Lambda**4
                    ])

                # TROUS NOIRS / HOLOGRAPHIE
                elif any(word in insp_lower for word in ["black hole", "swampland", "holograph"]):
                    term = random.choice([
                        Lambda**4 * sp.log(1 + (phi / M_pl)**2),
                        (phi / M_pl)**4 * Lambda**4,
                        Lambda**4 * sp.tanh(phi / M_pl)**2,
                        (phi / M_pl)**8 * sp.log(1 + phi**2 / M_pl**2) / M_pl**4,
                        G * phi**2 / (x**2 + y**2 + z**2 + 1e-9)**0.5
                    ])

                # WORMHOLES / MULTIVERS
                elif any(word in insp_lower for word in ["wormhole", "multiverse", "everett"]):
                    term = random.choice([
                        delta**4 * sp.log(1 + phi**2 / delta**2),
                        Lambda**4 * sp.exp(-(phi / delta)**2),
                        delta**4 * sp.Heaviside(delta - phi) * sp.log(1 + phi**2 / delta**2),
                        (phi / M_pl)**4 * sp.tanh(phi / delta)**4,
                        sp.exp(-sp.Abs(t)) * sp.cos(phi/delta)
                    ])

                # BIOLOGIE QUANTIQUE (DNA / CONSCIENCE)
                elif any(word in insp_lower for word in ["dna", "conscience", "orch-or", "microtubules"]):
                    term = random.choice([
                        delta * sp.besselj(0, phi / delta) * sp.sin(k * x),
                        hbar * sp.log(sp.sin(phi / delta)**2 + 1) * m**2,
                        delta**4 * sp.log(1 + sp.Abs(phi / delta)**1.618),
                        m**4 * sp.besselj(1, phi / delta)**2,
                        sp.exp(-phi/delta) * sp.sin(x/delta)**2
                    ])

                # DÉFAUT : CRÉATIVITÉ LIBRE
                else:
                    term = random.choice([
                        (phi / M_pl)**6 * Lambda**4 / 6,
                        m**4 * sp.tanh(phi / delta)**4,
                        delta**4 * sp.log(1 + (phi / delta)**3),
                        Lambda**4 * sp.sech(phi / f_a)**4,
                        G * phi**4 / M_pl**2,
                        hbar**2 * sp.diff(phi, t)**2 / (2 * m**2) if hasattr(phi, 'diff') else m**2*phi**2,
                        (phi / M_pl)**10 * Lambda**4 / 120
                    ])
                    
            else:
                # --- BRANCHE B : EXPLORATION SAUVAGE (30%) ---
                term = random.choice([
                    delta * m**2 * phi**2,
                    m**4 * sp.sin(phi / delta)**4 / 4,
                    hbar**2 * sp.diff(phi, x)**2 / (2 * m**2) if hasattr(phi, 'diff') else m**2*phi**2,
                    delta**4 * sp.log(1 + sp.Abs(phi / delta)**2),
                    (phi / M_pl)**8 * Lambda**4 / 24,
                    G * m**2 * phi**2 / sp.sqrt(x**2 + y**2 + z**2 + delta**2),
                    Lambda**4 * sp.Heaviside(phi - delta)
                ])

            # === 2. LE MOTEUR DE FUSION (TRANSVERSALITÉ) ===
            # On applique ici la logique de couplage sur le 'term' qu'on vient de générer
            if term != 0:
                if base == 0:
                    base = term
                else:
                    fusion_roll = random.random()
                    if fusion_roll < 0.6:
                        base += term  # Superposition (Standard)
                    elif fusion_roll < 0.85:
                        base *= term  # Couplage (C'est ici que la transversalité opère !)
                    elif fusion_roll < 0.95:
                        # Composition (Modulation par l'échelle de Planck)
                        base = base * sp.exp(-sp.Abs(term) / (ALL_SYMBOLS.get('M_p', sp.Symbol('M_p'))))
                    else:
                        # Division (Potentiel inverse / Points critiques)
                        base += 1 / (sp.Abs(term) + 1e-9)
                        
        # === 3. HYBRIDATION FINALE ===
        if random.random() < 0.25 and self.supervisor_templates:
            try:
                extra = random.choice(self.supervisor_templates)()
                base += 0.5 * extra
                # logging.debug(f"[{self.id}] Hybridation appliquée.")
            except:
                pass

        # === 4. MUTATIONS TEMPORELLES (Le Temps est prioritaire) ===
        if random.random() < 0.4:
            # logging.debug(f"[{self.id}] Injection de dynamique temporelle.")
            try:
                time_term = random.choice([
                    hbar * sp.diff(phi, t)**2 / (2 * m**2),
                    Lambda**4 * sp.exp(-t / delta),
                    delta**4 * sp.log(1 + t**2 / delta**2),
                    m**4 * sp.Heaviside(t) * sp.exp(-t / delta),
                    (phi / M_pl)**4 * sp.sin(t / delta)**2,
                    hbar * sp.log(1 + sp.diff(phi, t)**2),
                    Lambda**4 * sp.Heaviside(t) * sp.log(1 + t / delta),
                    delta**4 * sp.tanh(t / delta)**2,
                    hbar * sp.diff(phi, t)**4 / (24 * m**4),
                    Lambda**4 * sp.log(sp.exp(-t / delta) + 1),
                    m**4 * sp.cos(t / delta)**4,
                    delta**4 * sp.Heaviside(-t) * sp.exp(t / delta),
                    (phi / M_pl)**6 * sp.sin(t / M_pl)**2 / 6,
                    hbar * sp.log(1 + sp.diff(phi, t)**4),
                    Lambda**4 * sp.sech(t / delta)**2
                ])
                base += time_term
            except:
                pass # Ignorer si la différenciation échoue (ex: phi n'est pas une fonction)

        # === 5. EXPLORATION FORCÉE DE NOUVELLES FAMILLES (30%) ===
        if random.random() < 0.3 and self.__class__.available_families:
            random_family = random.choice(self.__class__.available_families)
            fam_lower = random_family.lower()
            
            # Injection spécifique basée sur la famille forcée
            fam_term = 0
            if "wormhole" in fam_lower:
                fam_term = delta**4 * sp.log(1 + phi**2 / delta**2)
            elif "dna" in fam_lower:
                fam_term = delta**4 * sp.sin(phi / delta)**6
            elif "multiverse" in fam_lower:
                fam_term = Lambda**4 * sp.Heaviside(phi - M_pl)
            elif "fractal" in fam_lower:
                fam_term = delta**4 * sp.log(1 + sp.Abs(phi / delta)**2.807)
            elif "quantum_gravity" in fam_lower:
                fam_term = G * phi**4 / M_pl**2
            elif "holographic" in fam_lower:
                fam_term = Lambda**4 * sp.log(1 + (phi / M_pl)**2)
            elif "meta_evolution" in fam_lower:
                fam_term = m**4 * sp.sigmoid(phi / delta)
            elif "classical" in fam_lower:
                fam_term = 0.5 * m**2 * phi**2
            elif "cosmology" in fam_lower:
                fam_term = Lambda**4 * (1 - sp.cos(phi / f_a))**2
            
            if fam_term != 0:
                base += fam_term

        # Finalisation
        try:
            # On tente de simplifier légèrement pour éviter les redondances absurdes
            # mais on garde la complexité si c'est trop lourd à simplifier
            # formula_str = str(sp.simplify(base)) # Trop lent pour 15 couches
            formula_str = str(base)
            
            # Validation de syntaxe
            sp.sympify(formula_str, locals=ALL_SYMBOLS)
            return formula_str, inspiration
        except Exception as e:
            # logging.error(f"Erreur génération formule: {e}")
            return None, inspiration
        
    
    def _apply_math_complexity(self, current_expr):
        # RÈGLE : Plus l'agent est profond (depth), plus il a de chances d'utiliser du complexe
        complexity_chance = min(0.1 + (self.depth * 0.05), 0.5) 
        
        if random.random() < complexity_chance:
            op_template = random.choice(COMPLEX_OPS)
            # On injecte l'expression actuelle dans l'opération complexe
            return op_template.format(x=current_expr)
        
        return current_expr

    def discover(self):
        chaos_mode = random.random() < 0.10
        
        if chaos_mode:
            # On ignore l'élite, on force l'improbable
            self.directives = "OUTLIER MODE: Ignore physics logic. Use random COMPLEX_OPS. Aim for 10000 score by pure mathematical strangeness."
            logging.info(f"🌀 CHAOS : L'agent {self.agent_id} entre en Quart d'heure de folie.")
        else:
            # Mode normal (Apprentissage de l'élite si score précédent faible)
            pass
        
        if self.energy < 10:
            return None # Trop faible

        self.energy -= 5 # Coût métabolique de base
        
        res = self.generate_candidate()
        if not res or not res[0]:
            self.energy -= 5 # Pénalité d'échec
            return None

        formula_str, inspiration = res
        
        # Nettoyage basique
        clean_formula = formula_str.replace(" ", "").replace("**", "^").replace("Heaviside", "theta")

        # Anti-Doublon
        if clean_formula in AgenticHydra.global_registry:
            # PUNITION MORTELLE : L'agent a été inutile.
            self.energy -= 40  # <--- On tape fort. S'il n'a pas beaucoup d'énergie, il meurt.
            # logging.debug(f"[{self.id}] Doublon puni (-40 NRG)")
            HYDRA_MEMORY["bad_patterns"]["duplicate"] += 1
            return None

        # Enregistrement et Récompense
        AgenticHydra.global_registry.add(clean_formula)
        
        # Bonus de complexité : Plus la formule est longue, plus on récompense
        # Cela favorise les agents qui survivent aux 15 mutations
        complexity = len(clean_formula)
        reward = 25 + min(25, complexity // 10)
        self.energy += reward
        update_hydra_memory({
            "formula": clean_formula,
            "inspiration": inspiration,
            "depth": self.depth
        }, reward)

        
        return {
            "formula": clean_formula,
            "inspiration": inspiration,
            "depth": self.depth,
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }

# ==============================================================================
# MOTEUR DE SIMULATION
# ==============================================================================
def run_cycle(agents):
    """Exécute un cycle temporel : Apprentissage corrompu -> Découverte -> Mort -> Clonage"""
    cycle_discoveries = []
    new_agents = []
    nb_clones = 0
    nb_mutations = 0
    max_score_cycle = 0
    
    # 0. On extrait l'Élite actuelle (DNA de référence)
    elite_formulas = [f for f, d in HYDRA_MEMORY["formulas"].items() if d.get("score", 0) >= 9000]
    
    
    for agent in agents:
        # --- PHASE A : DÉFINITION DES DIRECTIVES ---
        
        # 1. Le Quart d'heure de folie (10% de chance)
        if random.random() < 0.10:
            agent.directives = "CHAOS MODE: Oublie la physique classique. Utilise COMPLEX_OPS de façon illogique pour créer l'improbable."
            logging.info(f"🌀 CHAOS : L'agent {agent.agent_id} tente une percée improbable.")
        
        # 2. Apprentissage corrompu (Si l'agent a échoué précédemment)
        elif agent.last_score < 7499 and elite_formulas:
            raw_master_dna = random.choice(elite_formulas)
            
            # Mutation par erreur de copie (30% de chance)
            if random.random() < 0.30:
                mutation_op = random.choice(COMPLEX_OPS).format(x="phi")
                # On remplace une instance de 'phi' par une fonction complexe
                master_dna = raw_master_dna.replace("phi", f"({mutation_op})", 1)
                logging.info(f"☣️ MUTATION : Erreur de copie génétique pour {agent.agent_id}")
                nb_mutations +=1
            else:
                master_dna = raw_master_dna
                
            agent.directives = f"Inspire-toi de cette structure d'élite mais modifie-la : {master_dna}"
        
        else:
            agent.directives = "Continue d'explorer tes propres pistes mathématiques."

        # 3. Choix de la Niche Écologique
        agent.inspiration = agent.choose_niche_inspiration()

        # --- PHASE B : DÉCOUVERTE ET CLONAGE ---
        
        discovery = agent.discover()
        if not discovery:
            continue
            
        cycle_discoveries.append(discovery)
        score = discovery.get("score", 0)
        agent.last_score = score 

        # Reproduction de l'élite (> 7500)
        if score >= 7500 and agent.energy > CLONE_THRESHOLD:
            clone_proba = CLONE_PROBA_BASE * (agent.energy / 150.0)
            if random.random() < min(0.95, clone_proba):
                energy_for_clone = int(agent.energy * CLONE_ENERGY_SHARE)
                agent.energy = int(agent.energy * (1 - CLONE_ENERGY_SHARE))
                
                clone = agent.clone()
                nb_clones +=1
                clone.energy = energy_for_clone
                clone.depth = agent.depth + 1
                new_agents.append(clone)
                logging.info(f"🧬 ELITE CLONAGE : {agent.agent_id} (Score {score}) -> Depth {clone.depth}")

    # --- PHASE C : SURVIE ET FUSION ---
    survivors = [a for a in agents if a.energy > 0]
    survivors.extend(new_agents)
    # Calcul des stats globales
    total_agents = len(survivors)
    avg_depth = sum(a.depth for a in survivors) / total_agents if total_agents > 0 else 0

    # Affichage dynamique sur une seule ligne (\r permet de revenir au début de la ligne)
    sys.stdout.write(
        f"\r[HYDRA STATUS] Agents: {total_agents} | Clones: {nb_clones} | Mutations: {nb_mutations} | "
        f"Max Score: {max_score_cycle:.0f} | Avg Depth: {avg_depth:.1f} | Scan: {len(HYDRA_MEMORY['formulas'])}"
    )
    sys.stdout.flush()
    return survivors, cycle_discoveries

def start_agentic_protocol():
    """Point d'entrée principal"""
    print(f"\n🌌 INITIALISATION DU PROTOCOLE HYDRA (TYPE III)")
    print("="*60)
    
    # Chargement mémoire
    AgenticHydra.load_collective_memory()
    
    # Genèse
    agents = [AgenticHydra(energy=120) for _ in range(5)]
    all_discoveries = []

    try:
        for p in range(1, PULSES + 1):
            agents, cycle_disc = run_cycle(agents)
            all_discoveries.extend(cycle_disc)
            
            # Stats en temps réel
            current_agents = len(agents)
            max_depth = max((a.depth for a in agents), default=0)
            avg_energy = sum(a.energy for a in agents) / current_agents if current_agents else 0
            
            print(f"Pulse {p:03d}/{PULSES} | Agents: {current_agents:3d} | MaxGen: {max_depth:2d} | AvgNRG: {avg_energy:5.1f} | Discoveries: {len(all_discoveries):4d}", end='\r')

            # Sécurité anti-extinction
            if current_agents == 0:
                print("\n[!] EXTINCTION DETECTÉE - Réinjection d'urgence.")
                agents = [AgenticHydra(energy=100) for _ in range(3)]

            # Sécurité surpopulation
            if current_agents > 500:
                agents = sorted(agents, key=lambda x: x.energy, reverse=True)[:400]

    except KeyboardInterrupt:
        print("\n\n[!] Interruption manuelle du protocole.")
    
    print("\n" + "="*60)
    print(f"RAPPORT FINAL : {len(all_discoveries)} découvertes générées.")
    
    # ==============================================================================
    # SAUVEGARDE FINALE & RÉCAPITULATIF STRATÉGIQUE
    # ==============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Préparation du dictionnaire de formules (pour hydra_memory / prefilter)
    formatted_formulas = {
        f["formula"]: f for f in all_discoveries if "formula" in f
    }
    
    # 2. Préparation du RÉCAPITULATIF (Scores et Tendances)
    # On convertit les defaultdict en dict simples
    recap_global = {
        # On va chercher dans le sous-dictionnaire 'stats'
        "concept_scores": dict(HYDRA_MEMORY.get("stats", {}).get("concept_scores", {})),
        "category_scores": dict(HYDRA_MEMORY.get("stats", {}).get("category_scores", {})),
        "term_patterns": dict(HYDRA_MEMORY["term_patterns"]),
        "bad_patterns": dict(HYDRA_MEMORY["bad_patterns"]),
        "trends": {
            "dominant_category": max(HYDRA_MEMORY["category_scores"], key=HYDRA_MEMORY["category_scores"].get) if HYDRA_MEMORY["category_scores"] else None,
            "top_concept": max(HYDRA_MEMORY["concept_scores"], key=HYDRA_MEMORY["concept_scores"].get) if HYDRA_MEMORY["concept_scores"] else None,
            "most_used_function": max(HYDRA_MEMORY["term_patterns"], key=HYDRA_MEMORY["term_patterns"].get) if HYDRA_MEMORY["term_patterns"] else None,
            "total_discoveries_this_session": len(all_discoveries)
        }
    }

    # 3. SAUVEGARDE DU CERVEAU (hydra_memory.json)
    # On garde les formules ICI pour que le préfiltre puisse bosser
    save_json_safe(
        MEMORY_FILE, 
        {
            "formulas": formatted_formulas, 
            "stats": recap_global 
        }
    )

    
    # 4. SAUVEGARDE DE L'ARCHIVE (Le récapitulatif pur que tu as demandé)
    filename_archive = f"hydra_archive_stats_{timestamp}.json"
    save_json_safe(filename_archive, recap_global)
    
    print(f"✅ Mémoire synchronisée pour Préfiltre : {MEMORY_FILE}")
    print(f"📊 Récapitulatif stratégique archivé : {filename_archive}")
    print(f"🔥 Tendance : {recap_global['trends']['dominant_category']} domine avec {recap_global['trends']['top_concept']}")
    print("=" * 60)
if __name__ == "__main__":
    start_agentic_protocol()
