#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IRIS-Station — Phases 1 à 5.

Phase 1 (squelette) : fenêtre Qt, base SQLite, gestion de projets, vérification
du modèle spaCy au démarrage (dépendance seulement).

Phase 2 (module ACH) : matrice des hypothèses concurrentes, score d'incohérence
pondéré, classement, diagnosticité, analyse de sensibilité.

Phase 3 (moteur bayésien) : vraisemblances P(E|H)/P(E|¬H), probabilités a
posteriori en log-espace, facteur de Bayes par cellule, table et diagramme.

Phase 4 (calibreur de prédictions) : onglets Matrice ACH / Prédictions / Graphe,
suivi et résolution de prédictions, score de Brier (global, par catégorie, par
période), décomposition de Murphy et diagramme de fiabilité (matplotlib).

Phase 5 (graphe de connaissances) : extraction d'entités et de relations par
spaCy + règles (zéro LLM), graphe rendu en matplotlib (disposition networkx),
plus court chemin en CTE récursive SQLite, communautés (Louvain) et centralité
d'intermédiarité (Brandes) en Python pur.

Phase 6 (NLP avancé) : résumé extractif par TextRank (PageRank sur similarité
cosinus de phrases), mots-clés par TF-IDF, patterns de relations externalisés
dans patterns.json (éditable), relations de citation/attribution (DOCUMENTED_BY,
source anonyme) et score de fiabilité des sources. Tout est déterministe.

Phase 7 (interface consolidée) : onglet « Vue d'ensemble » (dashboard 2×2 avec
cartes navigables), exports (rapport Markdown, matrice ACH et prédictions en CSV,
graphe en GraphML), restauration du dernier projet via QSettings, sauvegarde de
copie horodatée et barre de statut permanente.

Phase 8 (analyse assistée + Yggdrasil) : export Markdown enrichi d'un front matter
YAML compatible Yggdrasil (identifiants positionnels, historique de probabilité
par hypothèse, score de tension par nœud de bifurcation), analyse assistée par un
LLM OpenAI-compatible (FantasyAI Cloud) avec interface de révision humaine
obligatoire avant insertion atomique, et détection de doublons (preuves par
similarité d'embeddings, entités par normalisation). Migration de schéma non
destructive (horizons, nœuds de bifurcation, colonnes Yggdrasil sur hypotheses).

Le LLM reste périphérique : il propose, l'utilisateur valide, et tous les calculs
(ACH, bayésien, scoring, graphe, résumés) demeurent locaux et déterministes. La
clé API n'est stockée que dans QSettings, jamais en base.

Lançable tel quel :  python main.py
Empaquetable :       PyInstaller (chemins relatifs à sys._MEIPASS / dossier exe).
"""

from __future__ import annotations

import math
import os
import csv
import json
import re
import sqlite3
import statistics
import sys
import unicodedata
from collections import defaultdict, deque
from datetime import date, datetime
from functools import partial
from html import escape
from pathlib import Path

# --- Import PySide6 avec message d'aide si absent ----------------------------
try:
    from PySide6.QtCore import (
        QByteArray,
        QDate,
        QObject,
        QRectF,
        QSettings,
        QSize,
        QStandardPaths,
        Qt,
        QThread,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QDesktopServices,
        QFont,
        QKeySequence,
        QPainter,
        QPen,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDateEdit,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    from PySide6.QtNetwork import (
        QNetworkAccessManager,
        QNetworkReply,
        QNetworkRequest,
    )
except ImportError:  # pragma: no cover - dépendance manquante
    _msg = (
        "PySide6 est requis pour lancer IRIS-Station.\n\n"
        "Installez-le avec :\n    pip install PySide6\n\n"
        "Si vous avez créé un exécutable, rebuildez APRÈS avoir installé "
        "PySide6 dans l'environnement de build."
    )
    # sys.stderr peut valoir None dans un exécutable « windowed » (sans console) :
    # on écrit prudemment et, sous Windows, on affiche une boîte de dialogue native.
    if sys.stderr is not None:
        try:
            sys.stderr.write(_msg + "\n")
        except Exception:  # noqa: BLE001
            pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0, _msg, "IRIS-Station — dépendance manquante", 0x10
            )
        except Exception:  # noqa: BLE001
            pass
    sys.exit(1)

# --- matplotlib (diagramme de fiabilité) — import tolérant -------------------
MATPLOTLIB_AVAILABLE = True
try:
    os.environ.setdefault("QT_API", "PySide6")
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # noqa: BLE001 - matplotlib absent ou backend indisponible
    MATPLOTLIB_AVAILABLE = False

# --- networkx (uniquement pour la DISPOSITION du graphe) — import tolérant ----
# networkx ne sert qu'au calcul des positions (spring/kamada-kawai). Les
# algorithmes de graphe (plus court chemin, communautés, intermédiarité) sont
# implémentés en SQL/Python pur, sans networkx.
NETWORKX_AVAILABLE = True
try:
    import networkx as nx
except Exception:  # noqa: BLE001 - networkx absent
    NETWORKX_AVAILABLE = False

# --- PyYAML (export Yggdrasil) — import tolérant ------------------------------
YAML_AVAILABLE = True
try:
    import yaml
except Exception:  # noqa: BLE001 - pyyaml absent
    YAML_AVAILABLE = False


# =============================================================================
#  Constantes & utilitaires de chemins (compatibles PyInstaller)
# =============================================================================

APP_NAME = "IRIS-Station"
ORG_NAME = "IRIS"
DB_FILENAME = "iris_station.db"
SPACY_MODEL = "fr_core_news_sm"

# Échelle de cohérence ACH (code → valeur numérique).
CODE_TO_SCORE: dict[str, int] = {"CC": 1, "C": 2, "N": 3, "I": 4, "II": 5}
SCORE_TO_CODE: dict[int, str] = {v: k for k, v in CODE_TO_SCORE.items()}
COMBO_ITEMS: list[str] = ["", "CC", "C", "N", "I", "II"]  # "" = non évalué
CODE_TOOLTIP = (
    "Cohérence preuve/hypothèse :\n"
    "  CC = très cohérent (1)\n"
    "  C  = cohérent (2)\n"
    "  N  = neutre (3)\n"
    "  I  = incohérent (4)\n"
    "  II = très incohérent (5)\n"
    "  (vide) = non évalué"
)
GREEN = QColor("#9be79b")

# Échelle 1-5 des vraisemblances bayésiennes → probabilités.
LIKELIHOOD_OPTIONS: list[tuple[str, float | None]] = [
    ("— non défini", None),
    ("5 — très probable (0,90)", 0.90),
    ("4 — probable (0,70)", 0.70),
    ("3 — neutre (0,50)", 0.50),
    ("2 — improbable (0,30)", 0.30),
    ("1 — très improbable (0,10)", 0.10),
]

BF_GREEN = "#1b7f3b"
BF_RED = "#b03030"
BF_GREY = "#888888"

# Catégories proposées par défaut pour les prédictions (liste personnalisable).
DEFAULT_CATEGORIES = [
    "Politique",
    "Économie",
    "Technologie",
    "Société",
    "Science",
    "Santé",
    "Autre",
]


def resource_path(relative: str | Path) -> Path:
    """Renvoie le chemin absolu d'une ressource embarquée (compatible PyInstaller)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent
    return base / relative


def app_data_dir() -> Path:
    """Renvoie (et crée) le dossier de données utilisateur de l'application."""
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    base = Path(location) if location else Path.home() / f".{APP_NAME.lower()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def parse_iso_date(value: str | None) -> date | None:
    """Convertit une chaîne ISO (éventuellement datetime) en date, ou None."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


# =============================================================================
#  Algorithmes ACH (fonctions pures)
# =============================================================================

def compute_incoherence(
    hyp_ids: list[int],
    evidence: list[tuple[int, int]],
    scores: dict[tuple[int, int], int],
) -> dict[int, float | None]:
    """Score d'incohérence pondéré de chaque hypothèse (None si non évaluée)."""
    result: dict[int, float | None] = {}
    for h in hyp_ids:
        total = 0.0
        count = 0
        for ev_id, cred in evidence:
            s = scores.get((ev_id, h))
            if s is not None:
                total += s * (cred or 1)
                count += 1
        result[h] = (total / count) if count else None
    return result


def compute_ranking(
    hyp_ids: list[int], incoherence: dict[int, float | None]
) -> list[int]:
    """Classe les hypothèses par incohérence croissante (départage par id)."""
    scored = [(h, incoherence[h]) for h in hyp_ids if incoherence.get(h) is not None]
    scored.sort(key=lambda x: (x[1], x[0]))  # type: ignore[arg-type]
    return [h for h, _ in scored]


def compute_diagnosticity(
    hyp_ids: list[int],
    evidence: list[tuple[int, int]],
    scores: dict[tuple[int, int], int],
) -> dict[int, float | None]:
    """Diagnosticité de chaque preuve = écart-type des valeurs pondérées."""
    result: dict[int, float | None] = {}
    for ev_id, cred in evidence:
        vals = [
            s * (cred or 1)
            for h in hyp_ids
            if (s := scores.get((ev_id, h))) is not None
        ]
        if len(vals) >= 2:
            result[ev_id] = statistics.pstdev(vals)
        elif len(vals) == 1:
            result[ev_id] = 0.0
        else:
            result[ev_id] = None
    return result


def compute_sensitivity(
    hyp_ids: list[int],
    evidence: list[tuple[int, int]],
    scores: dict[tuple[int, int], int],
) -> tuple[list[int], list[tuple[int, bool, list[int]]]]:
    """Effet du retrait de chaque preuve sur le classement ACH."""
    base_rank = compute_ranking(hyp_ids, compute_incoherence(hyp_ids, evidence, scores))
    out: list[tuple[int, bool, list[int]]] = []
    for ev_id, _ in evidence:
        reduced = [(e, c) for (e, c) in evidence if e != ev_id]
        rank = compute_ranking(hyp_ids, compute_incoherence(hyp_ids, reduced, scores))
        out.append((ev_id, rank != base_rank, rank))
    return base_rank, out


# =============================================================================
#  Moteur bayésien (fonction pure, log-espace)
# =============================================================================

def compute_bayesian(
    hyp_ids: list[int],
    priors: dict[int, float | None],
    likelihoods: dict[tuple[int, int], tuple[float, float]],
) -> dict[int, dict[str, float | int]]:
    """Probabilités a posteriori par le théorème de Bayes (log-odds stables)."""
    eps = 1e-9
    out: dict[int, dict[str, float | int]] = {}
    for h in hyp_ids:
        prior = priors.get(h)
        if prior is None:
            prior = 0.5
        prior = min(max(prior, 1e-6), 1 - 1e-6)

        log_odds = math.log(prior) - math.log(1 - prior)
        cum_log_lr = 0.0
        n = 0
        for (ev_id, hyp_id), (p_h, p_not_h) in likelihoods.items():
            if hyp_id != h:
                continue
            log_lr = math.log(max(p_h, eps)) - math.log(max(p_not_h, eps))
            log_odds += log_lr
            cum_log_lr += log_lr
            n += 1

        out[h] = {
            "prior": prior,
            "cumulative_lr": math.exp(cum_log_lr),
            "posterior": 1.0 / (1.0 + math.exp(-log_odds)),
            "n": n,
        }
    return out


# =============================================================================
#  Scoring des prédictions (Brier, Murphy, fiabilité) — fonctions pures
# =============================================================================

def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Score de Brier = moyenne des (p − o)². 0 = parfait, 0.25 = hasard.

    Args:
        pairs: liste de couples (probabilité annoncée, issue 0/1).

    Returns:
        Le score, ou None si aucune prédiction résolue.
    """
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def murphy_decomposition(
    pairs: list[tuple[float, int]], n_bins: int = 10
) -> dict[str, float] | None:
    """Décomposition de Murphy : Brier ≈ REL − RES + UNC (binning par déciles).

    REL (fiabilité/calibration) = Σ n_k (p̄_k − ō_k)² / N
    RES (résolution/discrimination) = Σ n_k (ō_k − ō)² / N
    UNC (incertitude du domaine) = ō (1 − ō)

    L'identité est exacte lorsque la probabilité est constante par bin ; avec un
    découpage en déciles, REL − RES + UNC approche le Brier (terme de dispersion
    intra-bin résiduel). Le Brier renvoyé est toujours la valeur directe exacte.
    """
    if not pairs:
        return None
    n = len(pairs)
    o_bar = sum(o for _, o in pairs) / n
    bins: dict[int, list[tuple[float, int]]] = {}
    for p, o in pairs:
        k = min(int(p * n_bins), n_bins - 1)
        bins.setdefault(k, []).append((p, o))
    rel = 0.0
    res = 0.0
    for items in bins.values():
        nk = len(items)
        p_bar_k = sum(p for p, _ in items) / nk
        o_bar_k = sum(o for _, o in items) / nk
        rel += nk * (p_bar_k - o_bar_k) ** 2
        res += nk * (o_bar_k - o_bar) ** 2
    return {
        "brier": brier_score(pairs),  # type: ignore[dict-item]
        "rel": rel / n,
        "res": res / n,
        "unc": o_bar * (1 - o_bar),
    }


def reliability_bins(
    pairs: list[tuple[float, int]], n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Données du diagramme de fiabilité, regroupées par déciles de probabilité.

    Returns:
        Liste (déciles non vides) de {'bin', 'mean_p', 'obs', 'n'}.
    """
    bins: dict[int, list[tuple[float, int]]] = {}
    for p, o in pairs:
        k = min(int(p * n_bins), n_bins - 1)
        bins.setdefault(k, []).append((p, o))
    out: list[dict[str, float | int]] = []
    for k in range(n_bins):
        items = bins.get(k, [])
        if items:
            nk = len(items)
            out.append(
                {
                    "bin": k,
                    "mean_p": sum(p for p, _ in items) / nk,
                    "obs": sum(o for _, o in items) / nk,
                    "n": nk,
                }
            )
    return out


# =============================================================================
#  Pipeline NLP local (spaCy) — extraction d'entités et de relations, zéro LLM
# =============================================================================

# Mapping des étiquettes NER de spaCy (fr_core_news_sm : PER/LOC/ORG/MISC) vers
# les types internes. GPE (modèles anglais) est replié sur LOC.
NER_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "LOC",
    "MISC": "MISC",
}
ENTITY_TYPES = ["PERSON", "ORG", "LOC", "MISC", "DATE"]

MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre",
]

# Déclencheurs lexicaux par défaut des relations dirigées. Sérialisés tels quels
# dans patterns.json : chaque clé est un type de relation, chaque valeur une liste
# de patterns spaCy Matcher (eux-mêmes des listes de dictionnaires de tokens).
DEFAULT_PATTERNS: dict[str, list[list[dict]]] = {
    "AFFIRMS": [
        [{"LEMMA": {"IN": ["affirmer", "déclarer", "annoncer", "soutenir",
                           "prétendre", "assurer"]}}],
    ],
    "CONTRADICTS": [
        [{"LEMMA": {"IN": ["contredire", "nier", "démentir", "réfuter",
                           "contester"]}}],
    ],
    "CREATED": [
        [{"LEMMA": {"IN": ["créer", "fonder", "lancer", "développer",
                           "concevoir", "bâtir"]}}],
    ],
    "AFFILIATED_WITH": [
        [{"LEMMA": "membre"}, {"LOWER": "de"}],
        [{"LEMMA": "travailler"}, {"LOWER": "pour"}],
        [{"LEMMA": "appartenir"}, {"LOWER": "à"}],
        [{"LEMMA": "diriger"}],
    ],
    # Citation et attribution (Phase 6).
    "DOCUMENTED_BY": [
        [{"LEMMA": "document"}, {"LOWER": "consulté"}, {"LOWER": "par"}],
        [{"LEMMA": "rapport"}, {"LOWER": "de"}],
        [{"LEMMA": "note"}, {"LOWER": "de"}],
    ],
    # Source nommée : « selon X », « d'après X » (X devient source).
    "SOURCE": [
        [{"LOWER": "selon"}],
        [{"LOWER": {"IN": ["d'après", "d’après"]}}],
        [{"LOWER": {"IN": ["d'", "d’"]}}, {"LOWER": "après"}],
    ],
    # Source anonyme : « selon une source proche de X » → X marquée anonyme.
    "ANON_SOURCE": [
        [{"LOWER": "selon"}, {"LOWER": "une"}, {"LOWER": "source"}],
        [{"LOWER": "source"}, {"LOWER": "proche"}, {"LOWER": "de"}],
        [{"LOWER": "source"}, {"LOWER": "anonyme"}],
    ],
}

# Types de relations dirigées « déclencheur → (sujet à gauche, objet à droite) ».
DIRECTED_RELATIONS = {"AFFIRMS", "CONTRADICTS", "CREATED", "AFFILIATED_WITH",
                      "DOCUMENTED_BY"}
# Relations marquant une entité comme source (pour le score de fiabilité).
SOURCE_RELATIONS = {"SOURCE", "AFFIRMS"}


def map_ner_label(label: str) -> str | None:
    """Convertit une étiquette NER spaCy en type interne (ou None si ignorée)."""
    return NER_LABEL_MAP.get(label)


# Indices lexicaux pour reclasser une entité en ORG. Corrige à la fois le
# sur-étiquetage PERSON de fr_core_news_sm sur les organisations françaises et
# les sorties LLM qui étiquettent mal médias/institutions.
_ORG_TOKENS = {
    "parti", "etat", "fed", "fbi", "cia", "nsa", "onu", "otan", "ue", "banque",
    "cour", "assemblee", "senat", "ministere", "agence", "universite", "ecole",
    "conseil", "commission", "comite", "federation", "syndicat", "institut",
    "fondation", "association", "ong", "societe", "groupe", "entreprise",
    "compagnie", "journal", "gazette", "presse", "media", "medias", "tribunal",
    "parquet", "police", "gendarmerie", "armee", "gouvernement", "republique",
    "times", "post", "news", "project", "inc", "corp", "ltd", "sa", "sas",
    "gmbh", "llc", "group", "press", "tv", "radio", "agency", "bank",
    "department", "departement", "bureau", "office", "council", "committee",
    "party", "university", "institute", "foundation", "union",
}
# Médias / agences de presse connus (comparaison normalisée).
_MEDIA_ORGS = {
    "afp", "cnn", "bbc", "reuters", "ap", "npr", "pbs", "msnbc", "fox news",
    "le monde", "los angeles times", "new york times", "the new york times",
    "washington post", "cnews", "bfmtv", "france info", "liberation",
    "le figaro", "mediapart", "the guardian", "politico", "axios",
}
_KNOWN_ORGS = _MEDIA_ORGS | {
    "fed", "fbi", "cia", "nsa", "onu", "otan", "ue", "the representation project",
    "parti democrate", "parti republicain", "parti socialiste", "fmi", "ocde",
}


def refine_entity_type(name: str, base_type: str | None) -> str:
    """Affine le type d'une entité : reclasse en ORG les institutions et médias
    mal étiquetés PERSON/MISC. Laisse intacts LOC, DATE et ORG."""
    base = base_type or "MISC"
    if base in ("LOC", "DATE", "ORG"):
        return base
    raw = (name or "").strip()
    norm = normalize_entity_name(raw)
    if not norm:
        return base
    if norm in _KNOWN_ORGS:
        return "ORG"
    tokens = set(re.split(r"[\s\-’']+", norm))
    if tokens & _ORG_TOKENS:
        return "ORG"
    # Sigle en capitales (CNN, AFP, FBI, ONU) — rarement un nom de personne.
    if re.fullmatch(r"[A-ZÀ-Ý][A-ZÀ-Ý.&]{1,6}", raw):
        return "ORG"
    return base


def refine_relation_type(
    source_name: str, target_name: str, base_type: str
) -> str:
    """Affine le type d'une relation : un média qui « affirme » à propos d'une
    entité la *rapporte* (SOURCE) plutôt qu'il ne l'asserte (AFFIRMS)."""
    if base_type == "AFFIRMS" and normalize_entity_name(source_name) in _MEDIA_ORGS:
        return "SOURCE"
    return base_type


def patterns_path() -> Path:
    """Chemin du fichier de patterns éditable par l'utilisateur."""
    return app_data_dir() / "patterns.json"


def load_patterns() -> dict[str, list[list[dict]]]:
    """Charge patterns.json (créé depuis les valeurs par défaut s'il est absent).

    En cas de fichier illisible, on retombe sur les patterns par défaut sans
    bloquer l'application.
    """
    path = patterns_path()
    if not path.exists():
        try:
            path.write_text(
                json.dumps(DEFAULT_PATTERNS, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return dict(DEFAULT_PATTERNS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
    except (OSError, ValueError):
        pass
    return dict(DEFAULT_PATTERNS)


def build_matcher(nlp, patterns: dict[str, list[list[dict]]]):  # noqa: ANN001
    """Construit le Matcher des relations à partir d'un dict de patterns.

    Chaque règle malformée (éditée par l'utilisateur) est ignorée sans faire
    échouer l'ensemble.
    """
    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    for rel_type, rule_patterns in patterns.items():
        try:
            matcher.add(rel_type, rule_patterns)
        except Exception:  # noqa: BLE001 - pattern utilisateur invalide
            continue
    return matcher


def build_date_matcher(nlp):  # noqa: ANN001 - dépend de spaCy
    """Construit un Matcher conservateur pour les dates (absentes du modèle fr)."""
    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    matcher.add(
        "DATE",
        [
            [{"IS_DIGIT": True}, {"LOWER": {"IN": MONTHS_FR}}, {"IS_DIGIT": True, "OP": "?"}],
            [{"LOWER": {"IN": MONTHS_FR}}, {"IS_DIGIT": True}],
            [{"TEXT": {"REGEX": r"^(19|20)\d{2}$"}}],
        ],
    )
    return matcher


def analyze_text(
    nlp, matcher, date_matcher, text: str  # noqa: ANN001
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, str, str, str]],
    list[tuple[str, str, dict]],
]:
    """Extrait entités, relations et métadonnées d'un texte (spaCy + règles).

    Returns:
        (entities, relations, entity_meta) où
        - entities = [(nom, type)] dédupliqué,
        - relations = [(sujet, type_sujet, type_relation, objet, type_objet)],
        - entity_meta = [(nom, type, métadonnées)] (ex. source anonyme).
    """
    from spacy.util import filter_spans

    doc = nlp(text)

    ents: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_entity(name: str, etype: str) -> None:
        name = name.strip()
        if len(name) < 2:
            return
        key = (name.lower(), etype)
        if key not in seen:
            seen.add(key)
            ents.append((name, etype))

    for ent in doc.ents:
        mapped = map_ner_label(ent.label_)
        if mapped:
            add_entity(ent.text, refine_entity_type(ent.text, mapped))

    date_spans = filter_spans([doc[s:e] for _, s, e in date_matcher(doc)])
    for span in date_spans:
        add_entity(span.text, "DATE")

    relations: list[tuple[str, str, str, str, str]] = []
    entity_meta: list[tuple[str, str, dict]] = []

    def ent_type(span) -> str | None:  # noqa: ANN001
        mapped = map_ner_label(span.label_)
        return refine_entity_type(span.text, mapped) if mapped else None

    for match_id, start, end in matcher(doc):
        rel_type = nlp.vocab.strings[match_id]
        sent = doc[start].sent
        sent_ents = [
            e for e in doc.ents
            if e.start >= sent.start and e.end <= sent.end and ent_type(e)
        ]
        right = [e for e in sent_ents if e.start >= end]

        if rel_type == "ANON_SOURCE":
            # Marque l'entité à droite comme source anonyme (aucune relation créée).
            if right:
                src = min(right, key=lambda e: e.start)
                entity_meta.append(
                    (src.text.strip(), ent_type(src), {"anonymous_source": True})
                )
            continue

        if rel_type == "SOURCE":
            if not right:
                continue
            src = min(right, key=lambda e: e.start)
            for other in sent_ents:
                if other is not src:
                    relations.append(
                        (src.text.strip(), ent_type(src), "SOURCE",
                         other.text.strip(), ent_type(other))
                    )
            continue

        # Relations dirigées : sujet le plus proche à gauche, objet à droite.
        left = [e for e in sent_ents if e.end <= start]
        if left and right:
            subj = max(left, key=lambda e: e.end)
            obj = min(right, key=lambda e: e.start)
            relations.append(
                (subj.text.strip(), ent_type(subj), rel_type,
                 obj.text.strip(), ent_type(obj))
            )

    return ents, relations, entity_meta


# =============================================================================
#  Résumé extractif (TextRank) et mots-clés (TF-IDF) — Python pur, zéro LLM
# =============================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs (0 si l'un est nul)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pagerank(
    sim: list[list[float]], damping: float = 0.85, tol: float = 1e-4,
    max_iter: int = 100,
) -> list[float]:
    """PageRank par itérations de la puissance sur une matrice de similarité."""
    n = len(sim)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    out = [sum(row) for row in sim]
    scores = [1.0 / n] * n
    for _ in range(max_iter):
        new = [(1 - damping) / n] * n
        for i in range(n):
            if out[i] == 0:  # nœud isolé : redistribution uniforme
                share = damping * scores[i] / n
                for j in range(n):
                    new[j] += share
            else:
                row = sim[i]
                base = damping * scores[i] / out[i]
                for j in range(n):
                    if row[j] > 0:
                        new[j] += base * row[j]
        diff = sum(abs(new[k] - scores[k]) for k in range(n))
        scores = new
        if diff < tol:
            break
    return scores


def textrank_select(vectors: list[list[float] | None], k: int) -> list[int]:
    """Sélectionne les k phrases de plus haut score TextRank (ordre original).

    Renvoie [] si aucune similarité exploitable (vecteurs tous nuls) — le repli
    lexical est géré par l'appelant.
    """
    n = len(vectors)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    sim = [[0.0] * n for _ in range(n)]
    any_edge = False
    for i in range(n):
        vi = vectors[i]
        if not vi:
            continue
        for j in range(i + 1, n):
            vj = vectors[j]
            if not vj:
                continue
            s = max(cosine_similarity(vi, vj), 0.0)
            if s > 0:
                any_edge = True
            sim[i][j] = s
            sim[j][i] = s
    if not any_edge:
        return []
    scores = pagerank(sim)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(order)


def textrank_select_lexical(token_sets: list[set[str]], k: int) -> list[int]:
    """Repli lexical de TextRank : similarité = chevauchement de lemmes (Jaccard)."""
    n = len(token_sets)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if a and b:
                inter = len(a & b)
                union = len(a | b)
                s = inter / union if union else 0.0
                sim[i][j] = s
                sim[j][i] = s
    scores = pagerank(sim)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(order)


def compute_tfidf(corpus: list[list[str]]) -> list[dict[str, float]]:
    """TF-IDF par document (IDF lissé pour rester robuste sur les petits corpus).

    TF = fréquence du terme / total des termes ;
    IDF = ln((1 + N) / (1 + df)) + 1 (variante lissée de ln(N / df)).
    """
    n = len(corpus)
    df: dict[str, int] = defaultdict(int)
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    out: list[dict[str, float]] = []
    for doc in corpus:
        total = len(doc) or 1
        tf: dict[str, int] = defaultdict(int)
        for t in doc:
            tf[t] += 1
        scores = {
            t: (c / total) * (math.log((1 + n) / (1 + df[t])) + 1.0)
            for t, c in tf.items()
        }
        out.append(scores)
    return out


def top_keywords(scores: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    """Renvoie les n termes de plus haut score TF-IDF."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def content_lemmas(doc) -> list[str]:  # noqa: ANN001 - Doc spaCy
    """Lemmes de contenu (alpha, hors stopwords, longueur > 2), en minuscules."""
    return [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop and len(t.text) > 2
    ]


def _span_vector(span, doc):  # noqa: ANN001 - Span/Doc spaCy
    """Vecteur d'une phrase : moyenne des embeddings de tokens (96 dim, fr_sm).

    Utilise span.vector si non nul, sinon la moyenne des lignes de doc.tensor
    (sortie du tok2vec), sinon None.
    """
    import numpy as np

    vec = span.vector
    if vec is not None and getattr(vec, "size", 0) and np.any(vec):
        return vec
    tensor = getattr(doc, "tensor", None)
    if tensor is not None and getattr(tensor, "size", 0) and span.end > span.start:
        return tensor[span.start:span.end].mean(axis=0)
    return None


def summarize_doc(doc, k_max: int = 5) -> tuple[list[str], list[int]]:  # noqa: ANN001
    """Résumé extractif TextRank d'un Doc spaCy.

    Returns:
        (phrases, indices_sélectionnés) — K = min(5, N/3), ordre d'origine.
    """
    sents = list(doc.sents)
    sentences = [s.text.strip() for s in sents]
    n = len(sentences)
    if n == 0:
        return [], []
    k = min(k_max, max(1, n // 3))
    if n <= k:
        return sentences, list(range(n))

    vectors: list[list[float] | None] = []
    for s in sents:
        v = _span_vector(s, doc)
        vectors.append([float(x) for x in v] if v is not None else None)

    selected = textrank_select(vectors, k)
    if not selected:  # vecteurs inexploitables → repli lexical (Jaccard de lemmes)
        token_sets = [
            {t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop}
            for s in sents
        ]
        selected = textrank_select_lexical(token_sets, k)
    return sentences, selected


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÝÉÈÊ«\"0-9])")


def split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases (pur Python, sans spaCy)."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []
    parts = _SENT_SPLIT_RE.split(normalized)
    return [p.strip() for p in parts if p.strip()]


def summarize_text_plain(text: str, k_max: int = 5) -> tuple[list[str], list[int]]:
    """Résumé extractif TextRank sans spaCy : phrases vectorisées en TF-IDF.

    Repli utilisé quand spaCy est absent ou incompatible. Renvoie
    (phrases, indices_sélectionnés) comme summarize_doc.
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return [], []
    k = min(k_max, max(1, n // 3))
    if n <= k:
        return sentences, list(range(n))

    tokens = [
        [w for w in re.findall(r"[\wàâäéèêëïîôöùûüç]+", s.lower()) if len(w) > 2]
        for s in sentences
    ]
    df: dict[str, int] = defaultdict(int)
    for toks in tokens:
        for w in set(toks):
            df[w] += 1
    vocab = {w: i for i, w in enumerate(df)}
    vectors: list[list[float] | None] = []
    for toks in tokens:
        total = len(toks) or 1
        vec = [0.0] * len(vocab)
        counts: dict[str, int] = defaultdict(int)
        for w in toks:
            counts[w] += 1
        for w, c in counts.items():
            vec[vocab[w]] = (c / total) * (math.log((1 + n) / (1 + df[w])) + 1.0)
        vectors.append(vec if any(vec) else None)

    selected = textrank_select(vectors, k)
    if not selected:
        selected = textrank_select_lexical([set(t) for t in tokens], k)
    if not selected:  # ultime repli : les phrases les plus « riches »
        selected = sorted(range(n), key=lambda i: len(tokens[i]), reverse=True)[:k]
        selected.sort()
    return sentences, selected


# =============================================================================
#  Helpers Phase 8 — normalisation, JSON, tokens, modèle spaCy partagé
# =============================================================================

_SHARED_NLP = None
_SHARED_NLP_FAILED = False


def _get_shared_nlp():  # noqa: ANN001 - renvoie un nlp spaCy ou None
    """Charge le modèle spaCy une seule fois (cache) ; None si indisponible."""
    global _SHARED_NLP, _SHARED_NLP_FAILED
    if _SHARED_NLP is not None:
        return _SHARED_NLP
    if _SHARED_NLP_FAILED:
        return None
    try:
        import spacy

        _SHARED_NLP = spacy.load(SPACY_MODEL)
    except Exception:  # noqa: BLE001
        _SHARED_NLP_FAILED = True
        return None
    return _SHARED_NLP


def normalize_entity_name(name: str) -> str:
    """Normalise un nom d'entité (sans accents, minuscule, sans espaces de bord)."""
    nfkd = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.lower().strip()


def extract_json_block(text: str) -> dict | None:
    """Extrait le premier objet JSON d'une réponse LLM (texte avant/après toléré)."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def detect_language(text: str) -> str:
    """Détecte fr/en grossièrement à partir des caractères accentués."""
    if not text:
        return "en"
    accented = sum(
        1 for c in text if c in "àâäéèêëïîôöùûüçÀÂÄÉÈÊËÏÎÔÖÙÛÜÇ"
    )
    letters = sum(1 for c in text if c.isalpha()) or 1
    if any(c in text for c in "çèê") or accented / letters > 0.015:
        return "fr"
    return "en"


def estimate_tokens(text: str) -> int:
    """Estime grossièrement le nombre de tokens (≈ longueur/4 en, /3 en fr)."""
    if not text:
        return 0
    divisor = 3 if detect_language(text) == "fr" else 4
    return len(text) // divisor


# Modèles de repli si GET /v1/models échoue (réseau / clé invalide).
DEFAULT_LLM_MODELS = [
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
]
DEFAULT_LLM_BASE_URL = "https://www.fantasyai.cloud/api/v1"

# Prompt système (extraction ACH structurée, JSON strict, zéro invention).
LLM_SYSTEM_PROMPT = (
    "Tu es un analyste structuré spécialisé dans l'Analysis of Competing "
    "Hypotheses (ACH).\n"
    "Tu vas recevoir un texte. Tu dois en extraire une analyse au format JSON "
    "STRICT.\n"
    "Ne mets AUCUN texte en dehors du JSON. Pas d'introduction, pas de "
    "commentaire. Juste l'objet JSON.\n\n"
    "RÈGLES ABSOLUES :\n"
    "- N'invente RIEN qui n'est pas explicitement dans le texte ou "
    "raisonnablement déductible.\n"
    "- Si tu n'as pas assez d'information pour une section, renvoie un tableau "
    "vide [].\n"
    "- Les scores ACH sont UNIQUEMENT : \"CC\", \"C\", \"N\", \"I\", \"II\".\n"
    "- Les types d'entités sont UNIQUEMENT : PERSON, ORG, LOC, MISC, DATE.\n"
    "  · ORG = médias, agences de presse, institutions, partis, administrations, "
    "entreprises, ONG (ex. CNN, AFP, Le Monde, FBI, Parti démocrate, ONU).\n"
    "  · LOC = lieux, villes, pays, régions (ex. Californie, État de New York).\n"
    "  · PERSON = uniquement des individus nommés. N'étiquette JAMAIS un média "
    "ou une institution en PERSON.\n"
    "- Les types de relations sont UNIQUEMENT : AFFIRMS, CONTRADICTS, SOURCE, "
    "AFFILIATED_WITH, CREATED, CO_OCCURS, DOCUMENTED_BY.\n"
    "  · AFFIRMS/CONTRADICTS : une personne ou une institution asserte/conteste "
    "explicitement quelque chose.\n"
    "  · SOURCE ou DOCUMENTED_BY : un média/une source rapporte une information "
    "(ex. « CNN → personne » est SOURCE, pas AFFIRMS).\n"
    "  · AFFILIATED_WITH : appartenance ou direction (ex. « Comey → FBI »).\n"
    "  · CO_OCCURS : simple co-occurrence sans lien sémantique précis.\n"
    "  Ne mets pas tout en AFFIRMS par défaut : choisis le type le plus juste.\n"
    "- Les horizons sont UNIQUEMENT : \"court_terme\", \"moyen_terme\", "
    "\"long_terme\", null.\n"
    "- credibility est un entier de 1 à 5."
)

# Squelette JSON attendu (inclus tel quel dans le prompt utilisateur).
LLM_JSON_TEMPLATE = """{
  "evidence_items": [
    {
      "content": "...",
      "source": "...",
      "credibility": 3,
      "ach_scores": {"H1": "C", "H2": "N"},
      "bayesian_likelihoods": {"H1": {"p_e_given_h": 0.7, "p_e_given_not_h": 0.3}}
    }
  ],
  "new_hypotheses": [
    {
      "label": "...",
      "description": "...",
      "prior_probability": 0.5,
      "horizon": "court_terme",
      "parent_hypothesis_label": null
    }
  ],
  "entities": [{"name": "...", "type": "PERSON"}],
  "relations": [{"source": "A", "target": "B", "type": "AFFIRMS"}],
  "predictions": [
    {"question": "...", "probability": 0.6, "deadline": "2027-06-01", "category": "..."}
  ],
  "bifurcation_nodes": [
    {
      "label": "Si X...",
      "condition_text": "...",
      "leads_to_hypotheses": ["H2"],
      "horizon": "moyen_terme",
      "parent_hypothesis": "H1"
    }
  ],
  "narrative_synthesis": {
    "resume_global": "...",
    "signaux_faibles": ["..."],
    "angles_morts": "..."
  }
}"""


def build_llm_user_prompt(
    mode: str,
    user_text: str,
    hypotheses: list[sqlite3.Row],
    evidence: list[sqlite3.Row],
    predictions: list[sqlite3.Row],
) -> str:
    """Construit le prompt utilisateur selon le mode et le contexte du projet."""
    parts: list[str] = []
    if mode == "completion":
        parts.append(
            "Complète UNIQUEMENT les cellules vides pour les hypothèses et preuves "
            "déjà existantes. N'ajoute PAS de nouvelles hypothèses sauf si le texte "
            "en introduit une clairement absente."
        )
    else:
        parts.append("Analyse ce texte de manière exhaustive.")

    hyp_lines = (
        "; ".join(f"{h['label']} : {h['description'] or ''}".strip() for h in hypotheses)
        or "(aucune)"
    )
    ev_previews = []
    for e in evidence:
        text = (e["content"] or "").strip().replace("\n", " ")
        ev_previews.append(text[:120] + ("…" if len(text) > 120 else ""))
    pred_lines = "; ".join(p["question"] for p in predictions) or "(aucune)"

    parts.append("")
    parts.append("Contexte du projet existant :")
    parts.append(f"- Hypothèses : {hyp_lines}")
    parts.append(
        f"- Preuves déjà enregistrées ({len(evidence)}) : "
        + ("; ".join(ev_previews) if ev_previews else "(aucune)")
    )
    parts.append(f"- Prédictions existantes ({len(predictions)}) : {pred_lines}")
    parts.append("")
    parts.append("Texte à analyser :")
    parts.append(user_text)
    parts.append("")
    parts.append(
        "Retourne UNIQUEMENT un objet JSON avec cette structure exacte "
        "(les champs vides sont des tableaux [] ou des objets {}):"
    )
    parts.append("")
    parts.append(LLM_JSON_TEMPLATE)
    return "\n".join(parts)


def build_yggdrasil_yaml(db, project_id: int) -> str:  # noqa: ANN001
    """Construit le front matter YAML Yggdrasil pour un projet.

    Identifiants positionnels (H1, E1, BN1, HZ1…) dans l'ordre de création.
    Renvoie "" si PyYAML est indisponible.
    """
    if not YAML_AVAILABLE:
        return ""
    project = db.get_project(project_id)
    if project is None:
        return ""

    hyps = db.list_hypotheses(project_id)
    evs = db.list_evidence(project_id)
    horizons = db.list_horizons(project_id)
    cells = db.get_cells(project_id)
    ents = db.list_entities(project_id)
    rels = db.list_relationships(project_id)
    preds = db.list_predictions(project_id)
    bns = db.list_bifurcation_nodes(project_id)

    hyp_pos = {h["id"]: f"H{i + 1}" for i, h in enumerate(hyps)}
    ev_pos = {e["id"]: f"E{i + 1}" for i, e in enumerate(evs)}
    hz_pos = {hz["id"]: f"HZ{i + 1}" for i, hz in enumerate(horizons)}

    hyp_ids = [h["id"] for h in hyps]
    priors = {
        h["id"]: (h["prior_probability"] if h["prior_probability"] is not None else 0.5)
        for h in hyps
    }
    likelihoods = {
        k: (c["p_h"], c["p_not_h"])
        for k, c in cells.items()
        if c["p_h"] is not None and c["p_not_h"] is not None
    }
    bayes = compute_bayesian(hyp_ids, priors, likelihoods) if hyps else {}
    evidence_cred = [(e["id"], e["credibility"]) for e in evs]
    scores = {k: int(c["score"]) for k, c in cells.items() if c["score"] is not None}
    diag = compute_diagnosticity(hyp_ids, evidence_cred, scores) if (hyps and evs) else {}

    data: dict[str, object] = {}
    data["project"] = {
        "name": project["name"],
        "created": str(project["created_at"])[:10] if project["created_at"] else None,
        "horizons": [
            {"id": hz_pos[hz["id"]], "label": hz["label"],
             "years_min": hz["years_min"], "years_max": hz["years_max"]}
            for hz in horizons
        ],
    }

    data["hypotheses"] = []
    for h in hyps:
        post = bayes.get(h["id"], {}).get("posterior")
        data["hypotheses"].append({
            "id": hyp_pos[h["id"]],
            "label": h["label"],
            "description": h["description"] or "",
            "prior": round(priors[h["id"]], 3),
            "posterior": round(post, 3) if post is not None
            else round(priors[h["id"]], 3),
            "horizon": hz_pos.get(h["horizon_id"]),
            "parent": hyp_pos.get(h["parent_hypothesis_id"]),
            "probability_history": db.compute_probability_history(project_id, h),
        })

    data["bifurcation_nodes"] = []
    for i, bn in enumerate(bns):
        leads_labels = [h["label"] for h in hyps if h["bifurcation_node_id"] == bn["id"]]
        leads_pos = [hyp_pos[h["id"]] for h in hyps if h["bifurcation_node_id"] == bn["id"]]
        data["bifurcation_nodes"].append({
            "id": f"BN{i + 1}",
            "label": bn["label"],
            "condition": bn["condition_text"] or "",
            "leads_to": leads_pos,
            "parent_hypothesis": hyp_pos.get(bn["parent_hypothesis_id"]),
            "horizon": hz_pos.get(bn["horizon_id"]),
            "tension_score": db.compute_tension_score(project_id, leads_labels),
        })

    data["evidence_items"] = []
    for e in evs:
        ev_id = e["id"]
        ach: dict[str, str] = {}
        bayesian: dict[str, dict] = {}
        for h in hyps:
            cell = cells.get((ev_id, h["id"]))
            if not cell:
                continue
            if cell["score"] is not None:
                ach[hyp_pos[h["id"]]] = SCORE_TO_CODE.get(int(cell["score"]), "")
            if cell["p_h"] is not None and cell["p_not_h"] is not None:
                bf = (cell["p_h"] / cell["p_not_h"]) if cell["p_not_h"] > 0 else None
                bayesian[hyp_pos[h["id"]]] = {
                    "p_e_given_h": round(cell["p_h"], 3),
                    "p_e_given_not_h": round(cell["p_not_h"], 3),
                    "bayes_factor": round(bf, 3) if bf is not None else None,
                }
        d = diag.get(ev_id)
        data["evidence_items"].append({
            "id": ev_pos[ev_id],
            "content": e["content"] or "",
            "source": e["source"] or "",
            "credibility": e["credibility"],
            "diagnosticite": round(d, 3) if d is not None else 0.0,
            "ach_scores": ach,
            "bayesian": bayesian,
        })

    degree: dict[int, int] = defaultdict(int)
    for r in rels:
        degree[r["source_entity_id"]] += 1
        degree[r["target_entity_id"]] += 1
    name_by_id = {en["id"]: en["name"] for en in ents}
    data["entities"] = [
        {"name": en["name"], "type": en["type"], "degree": degree.get(en["id"], 0)}
        for en in ents
    ]
    data["relations"] = [
        {"source": name_by_id.get(r["source_entity_id"], "?"),
         "target": name_by_id.get(r["target_entity_id"], "?"),
         "type": r["relation_type"], "weight": float(r["weight"] or 1.0)}
        for r in rels
    ]

    data["predictions"] = []
    for p in preds:
        outcome = p["outcome"]
        prob = p["probability"]
        brier = (
            round((prob - outcome) ** 2, 4)
            if outcome is not None and prob is not None else None
        )
        data["predictions"].append({
            "question": p["question"],
            "probability": round(prob, 3) if prob is not None else None,
            "deadline": p["deadline"],
            "outcome": outcome,
            "brier": brier,
        })

    dumped = yaml.dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    return "---\n# YGGDRASIL_IMPORT: v1\n" + dumped + "---\n"


# =============================================================================
#  Algorithmes de graphe (Python pur — Louvain, Brandes, disposition de secours)
# =============================================================================

def louvain_communities(
    nodes: list[int], edges: list[tuple[int, int, float]]
) -> dict[int, int]:
    """Détection de communautés par optimisation de modularité (déplacement local).

    Variante simplifiée de Louvain : la phase de déplacement local (phase 1) est
    itérée jusqu'à convergence. Renvoie un identifiant de communauté (0..k-1) par
    nœud.
    """
    adj: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    deg: dict[int, float] = defaultdict(float)
    m2 = 0.0
    for a, b, w in edges:
        if a == b:
            continue
        adj[a][b] += w
        adj[b][a] += w
        deg[a] += w
        deg[b] += w
        m2 += 2 * w
    if m2 == 0:
        return {n: i for i, n in enumerate(nodes)}

    comm = {n: n for n in nodes}
    comm_deg: dict[int, float] = defaultdict(float)
    for n in nodes:
        comm_deg[comm[n]] += deg[n]

    improved = True
    passes = 0
    while improved and passes < 100:
        improved = False
        passes += 1
        for n in nodes:
            ki = deg[n]
            current = comm[n]
            comm_deg[current] -= ki
            neigh: dict[int, float] = defaultdict(float)
            for nb, w in adj[n].items():
                if nb != n:
                    neigh[comm[nb]] += w
            best = current
            best_gain = neigh.get(current, 0.0) - comm_deg[current] * ki / m2
            for c, k_in in neigh.items():
                gain = k_in - comm_deg[c] * ki / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = c
            comm[n] = best
            comm_deg[best] += ki
            if best != current:
                improved = True

    labels: dict[int, int] = {}
    out: dict[int, int] = {}
    for n in nodes:
        c = comm[n]
        if c not in labels:
            labels[c] = len(labels)
        out[n] = labels[c]
    return out


def betweenness_centrality(
    nodes: list[int], adj: dict[int, set[int]]
) -> dict[int, float]:
    """Centralité d'intermédiarité (algorithme de Brandes, non pondéré, non dirigé)."""
    cb = {v: 0.0 for v in nodes}
    for s in nodes:
        stack: list[int] = []
        pred: dict[int, list[int]] = {w: [] for w in nodes}
        sigma = {t: 0.0 for t in nodes}
        sigma[s] = 1.0
        dist = {t: -1 for t in nodes}
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj.get(v, ()):  # type: ignore[arg-type]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    for v in cb:
        cb[v] /= 2.0  # graphe non dirigé : chaque paire comptée deux fois
    return cb


def spring_layout_pure(
    nodes: list[int],
    edges: list[tuple[int, int, float]],
    iterations: int = 80,
) -> dict[int, tuple[float, float]]:
    """Disposition de secours (Fruchterman-Reingold) — déterministe, sans dépendance.

    Utilisée seulement si networkx est absent. Placement initial sur un cercle
    (déterministe), puis répulsion/attraction itératives.
    """
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    pos = {
        node: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }
    adj: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for a, b, w in edges:
        adj[a][b] += w
        adj[b][a] += w

    k = math.sqrt(1.0 / n)  # distance idéale
    temp = 0.1
    for _ in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}
        for i, u in enumerate(nodes):
            ux, uy = pos[u]
            for v in nodes[i + 1:]:
                vx, vy = pos[v]
                dx, dy = ux - vx, uy - vy
                dist = math.hypot(dx, dy) or 1e-4
                rep = k * k / dist  # répulsion
                fx, fy = dx / dist * rep, dy / dist * rep
                disp[u][0] += fx
                disp[u][1] += fy
                disp[v][0] -= fx
                disp[v][1] -= fy
        for u in nodes:
            ux, uy = pos[u]
            for v, w in adj[u].items():
                vx, vy = pos[v]
                dx, dy = ux - vx, uy - vy
                dist = math.hypot(dx, dy) or 1e-4
                att = dist * dist / k * (1.0 + 0.1 * w)  # attraction
                disp[u][0] -= dx / dist * att
                disp[u][1] -= dy / dist * att
        for u in nodes:
            dx, dy = disp[u]
            d = math.hypot(dx, dy) or 1e-4
            nx_, ny_ = pos[u]
            pos[u] = (
                nx_ + dx / d * min(d, temp),
                ny_ + dy / d * min(d, temp),
            )
        temp = max(temp * 0.95, 0.01)
    return pos


# =============================================================================
#  Couche données : classe Database (SQL brut, sqlite3 stdlib, zéro ORM)
# =============================================================================

class Database:
    """Encapsule la connexion SQLite et le schéma de IRIS-Station."""

    def __init__(self, db_path: Path) -> None:
        """Ouvre (ou crée) la base, initialise le schéma et applique les migrations."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self._migrate()

    def _init_schema(self) -> None:
        """Crée les tables si elles n'existent pas (idempotent)."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                description TEXT,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id          INTEGER NOT NULL
                                    REFERENCES projects(id) ON DELETE CASCADE,
                label               TEXT NOT NULL,
                description         TEXT,
                prior_probability   REAL DEFAULT 0.5,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
                bifurcation_node_id INTEGER,
                horizon_id          INTEGER,
                parent_hypothesis_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS evidence_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL
                            REFERENCES projects(id) ON DELETE CASCADE,
                content     TEXT NOT NULL,
                source      TEXT,
                credibility INTEGER DEFAULT 3,
                added_at    TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id        INTEGER NOT NULL
                                  REFERENCES projects(id) ON DELETE CASCADE,
                question          TEXT NOT NULL,
                probability       REAL,
                created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
                deadline          TEXT,
                resolved_at       TEXT,
                outcome           INTEGER,
                category          TEXT,
                resolution_source TEXT
            );

            CREATE TABLE IF NOT EXISTS prediction_updates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id   INTEGER NOT NULL
                                REFERENCES predictions(id) ON DELETE CASCADE,
                old_probability REAL NOT NULL,
                new_probability REAL NOT NULL,
                rationale       TEXT,
                updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS matrix_scores (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id        INTEGER NOT NULL
                                  REFERENCES projects(id) ON DELETE CASCADE,
                evidence_id       INTEGER NOT NULL
                                  REFERENCES evidence_items(id) ON DELETE CASCADE,
                hypothesis_id     INTEGER NOT NULL
                                  REFERENCES hypotheses(id) ON DELETE CASCADE,
                consistency_score INTEGER CHECK(consistency_score BETWEEN 1 AND 5),
                notes             TEXT,
                p_e_given_h       REAL,
                p_e_given_not_h   REAL,
                UNIQUE(evidence_id, hypothesis_id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id                INTEGER NOT NULL
                                          REFERENCES projects(id) ON DELETE CASCADE,
                name                      TEXT NOT NULL,
                type                      TEXT,
                first_seen_in_evidence_id INTEGER
                                          REFERENCES evidence_items(id) ON DELETE SET NULL,
                metadata                  TEXT,
                UNIQUE(project_id, name, type)
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id       INTEGER NOT NULL
                                 REFERENCES projects(id) ON DELETE CASCADE,
                source_entity_id INTEGER NOT NULL
                                 REFERENCES entities(id) ON DELETE CASCADE,
                target_entity_id INTEGER NOT NULL
                                 REFERENCES entities(id) ON DELETE CASCADE,
                evidence_id      INTEGER REFERENCES evidence_items(id) ON DELETE SET NULL,
                relation_type    TEXT,
                weight           REAL DEFAULT 1.0,
                UNIQUE(project_id, source_entity_id, target_entity_id, relation_type)
            );

            CREATE TABLE IF NOT EXISTS entity_mentions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL
                            REFERENCES projects(id) ON DELETE CASCADE,
                entity_id   INTEGER NOT NULL
                            REFERENCES entities(id) ON DELETE CASCADE,
                evidence_id INTEGER NOT NULL
                            REFERENCES evidence_items(id) ON DELETE CASCADE,
                as_source   INTEGER DEFAULT 0,
                UNIQUE(project_id, entity_id, evidence_id)
            );

            CREATE TABLE IF NOT EXISTS horizons (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                label      TEXT NOT NULL,
                years_min  INTEGER,
                years_max  INTEGER
            );

            CREATE TABLE IF NOT EXISTS bifurcation_nodes (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id           INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                label                TEXT NOT NULL,
                condition_text       TEXT,
                parent_hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL,
                horizon_id           INTEGER REFERENCES horizons(id) ON DELETE SET NULL,
                created_at           TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def _migrate(self) -> None:
        """Ajoute les colonnes introduites après coup (bases antérieures)."""
        matrix_cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(matrix_scores)")
        }
        if "p_e_given_h" not in matrix_cols:
            self.conn.execute("ALTER TABLE matrix_scores ADD COLUMN p_e_given_h REAL")
        if "p_e_given_not_h" not in matrix_cols:
            self.conn.execute(
                "ALTER TABLE matrix_scores ADD COLUMN p_e_given_not_h REAL"
            )

        pred_cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(predictions)")
        }
        if "resolution_source" not in pred_cols:
            self.conn.execute(
                "ALTER TABLE predictions ADD COLUMN resolution_source TEXT"
            )

        # Phase 8 : colonnes Yggdrasil sur hypotheses (bases antérieures).
        hyp_cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(hypotheses)")
        }
        for col in ("bifurcation_node_id", "horizon_id", "parent_hypothesis_id"):
            if col not in hyp_cols:
                self.conn.execute(f"ALTER TABLE hypotheses ADD COLUMN {col} INTEGER")

        ev_cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(evidence_items)")
        }
        if "added_at" not in ev_cols:
            self.conn.execute(
                "ALTER TABLE evidence_items ADD COLUMN added_at TEXT "
                "DEFAULT CURRENT_TIMESTAMP"
            )

        self.conn.commit()
        self.ensure_default_horizons()

    def ensure_default_horizons(self) -> None:
        """Crée les 3 horizons par défaut pour chaque projet qui n'en a pas."""
        for (project_id,) in self.conn.execute("SELECT id FROM projects").fetchall():
            existing = self.conn.execute(
                "SELECT COUNT(*) FROM horizons WHERE project_id = ?", (project_id,)
            ).fetchone()[0]
            if existing == 0:
                self.conn.executemany(
                    "INSERT INTO horizons (project_id, label, years_min, years_max) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (project_id, "Court terme (0-2 ans)", 0, 2),
                        (project_id, "Moyen terme (2-5 ans)", 2, 5),
                        (project_id, "Long terme (5-20 ans)", 5, 20),
                    ],
                )
        self.conn.commit()

    # --- Projets -------------------------------------------------------------

    def create_project(self, name: str, description: str = "") -> int:
        """Insère un nouveau projet (avec ses horizons par défaut) et renvoie son id."""
        cur = self.conn.execute(
            "INSERT INTO projects (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        project_id = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO horizons (project_id, label, years_min, years_max) "
            "VALUES (?, ?, ?, ?)",
            [
                (project_id, "Court terme (0-2 ans)", 0, 2),
                (project_id, "Moyen terme (2-5 ans)", 2, 5),
                (project_id, "Long terme (5-20 ans)", 5, 20),
            ],
        )
        self.conn.commit()
        return project_id

    def list_projects(self) -> list[sqlite3.Row]:
        """Renvoie tous les projets, du plus récent au plus ancien."""
        return self.conn.execute(
            "SELECT id, name, description, created_at, updated_at "
            "FROM projects ORDER BY id DESC"
        ).fetchall()

    def get_project(self, project_id: int) -> sqlite3.Row | None:
        """Renvoie un projet par son identifiant, ou None."""
        return self.conn.execute(
            "SELECT id, name, description, created_at, updated_at "
            "FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    def delete_project(self, project_id: int) -> None:
        """Supprime un projet et TOUTES ses données liées (transaction atomique).

        Les suppressions sont explicites et ordonnées (enfants avant parents),
        afin de fonctionner même sur d'anciennes bases dont les clés étrangères
        n'auraient pas ON DELETE CASCADE.
        """
        self.conn.execute("BEGIN")
        try:
            ex = self.conn.execute
            ex("DELETE FROM matrix_scores WHERE project_id = ?", (project_id,))
            ex("DELETE FROM prediction_updates WHERE prediction_id IN "
               "(SELECT id FROM predictions WHERE project_id = ?)", (project_id,))
            ex("DELETE FROM predictions WHERE project_id = ?", (project_id,))
            ex("DELETE FROM entity_mentions WHERE project_id = ?", (project_id,))
            ex("DELETE FROM relationships WHERE project_id = ?", (project_id,))
            ex("DELETE FROM entities WHERE project_id = ?", (project_id,))
            ex("DELETE FROM bifurcation_nodes WHERE project_id = ?", (project_id,))
            ex("DELETE FROM evidence_items WHERE project_id = ?", (project_id,))
            ex("DELETE FROM hypotheses WHERE project_id = ?", (project_id,))
            ex("DELETE FROM horizons WHERE project_id = ?", (project_id,))
            ex("DELETE FROM projects WHERE id = ?", (project_id,))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # --- Hypothèses ----------------------------------------------------------

    def create_hypothesis(
        self,
        project_id: int,
        label: str,
        description: str = "",
        prior_probability: float = 0.5,
    ) -> int:
        """Insère une hypothèse et renvoie son identifiant."""
        cur = self.conn.execute(
            "INSERT INTO hypotheses (project_id, label, description, "
            "prior_probability) VALUES (?, ?, ?, ?)",
            (project_id, label, description, prior_probability),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_hypotheses(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les hypothèses d'un projet, dans l'ordre de création."""
        return self.conn.execute(
            "SELECT id, label, description, prior_probability, created_at, "
            "horizon_id, parent_hypothesis_id, bifurcation_node_id "
            "FROM hypotheses WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()

    # --- Preuves -------------------------------------------------------------

    def create_evidence(
        self,
        project_id: int,
        content: str,
        source: str = "",
        credibility: int = 3,
    ) -> int:
        """Insère une preuve et renvoie son identifiant."""
        cur = self.conn.execute(
            "INSERT INTO evidence_items (project_id, content, source, credibility) "
            "VALUES (?, ?, ?, ?)",
            (project_id, content, source, credibility),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_evidence(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les preuves d'un projet, dans l'ordre de création."""
        return self.conn.execute(
            "SELECT id, content, source, credibility "
            "FROM evidence_items WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()

    # --- Matrice (ACH + bayésien) --------------------------------------------

    def set_consistency(
        self, project_id: int, evidence_id: int, hypothesis_id: int, score: int | None
    ) -> None:
        """Définit (ou efface) le score de cohérence ACH ; préserve le bayésien."""
        self.conn.execute(
            "INSERT INTO matrix_scores "
            "(project_id, evidence_id, hypothesis_id, consistency_score) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(evidence_id, hypothesis_id) DO UPDATE SET "
            "consistency_score = excluded.consistency_score",
            (project_id, evidence_id, hypothesis_id, score),
        )
        self.conn.commit()
        if score is None:
            self._cleanup_cell(evidence_id, hypothesis_id)

    def set_likelihoods(
        self,
        project_id: int,
        evidence_id: int,
        hypothesis_id: int,
        p_e_given_h: float | None,
        p_e_given_not_h: float | None,
    ) -> None:
        """Définit (ou efface) les vraisemblances bayésiennes ; préserve l'ACH."""
        self.conn.execute(
            "INSERT INTO matrix_scores "
            "(project_id, evidence_id, hypothesis_id, p_e_given_h, p_e_given_not_h) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(evidence_id, hypothesis_id) DO UPDATE SET "
            "p_e_given_h = excluded.p_e_given_h, "
            "p_e_given_not_h = excluded.p_e_given_not_h",
            (project_id, evidence_id, hypothesis_id, p_e_given_h, p_e_given_not_h),
        )
        self.conn.commit()
        if p_e_given_h is None and p_e_given_not_h is None:
            self._cleanup_cell(evidence_id, hypothesis_id)

    def _cleanup_cell(self, evidence_id: int, hypothesis_id: int) -> None:
        """Supprime une cellule entièrement vide (ACH et bayésien nuls)."""
        self.conn.execute(
            "DELETE FROM matrix_scores WHERE evidence_id = ? AND hypothesis_id = ? "
            "AND consistency_score IS NULL AND p_e_given_h IS NULL "
            "AND p_e_given_not_h IS NULL",
            (evidence_id, hypothesis_id),
        )
        self.conn.commit()

    def get_cells(self, project_id: int) -> dict[tuple[int, int], dict[str, float | None]]:
        """Charge la matrice complète d'un projet."""
        rows = self.conn.execute(
            "SELECT evidence_id, hypothesis_id, consistency_score, "
            "p_e_given_h, p_e_given_not_h "
            "FROM matrix_scores WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {
            (r["evidence_id"], r["hypothesis_id"]): {
                "score": r["consistency_score"],
                "p_h": r["p_e_given_h"],
                "p_not_h": r["p_e_given_not_h"],
            }
            for r in rows
        }

    # --- Prédictions ---------------------------------------------------------

    def create_prediction(
        self,
        project_id: int,
        question: str,
        probability: float,
        deadline: str | None,
        category: str,
    ) -> int:
        """Insère une prédiction et renvoie son identifiant."""
        cur = self.conn.execute(
            "INSERT INTO predictions (project_id, question, probability, deadline, "
            "category) VALUES (?, ?, ?, ?, ?)",
            (project_id, question, probability, deadline, category),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_predictions(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les prédictions d'un projet, de la plus récente à la plus ancienne."""
        return self.conn.execute(
            "SELECT id, question, probability, category, deadline, created_at, "
            "resolved_at, outcome, resolution_source "
            "FROM predictions WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()

    def get_prediction(self, prediction_id: int) -> sqlite3.Row | None:
        """Renvoie une prédiction par son identifiant, ou None."""
        return self.conn.execute(
            "SELECT id, question, probability, category, deadline, created_at, "
            "resolved_at, outcome, resolution_source "
            "FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

    def add_prediction_update(
        self,
        prediction_id: int,
        old_probability: float,
        new_probability: float,
        rationale: str = "",
    ) -> None:
        """Historise une mise à jour de probabilité et met à jour la valeur courante."""
        self.conn.execute(
            "INSERT INTO prediction_updates "
            "(prediction_id, old_probability, new_probability, rationale) "
            "VALUES (?, ?, ?, ?)",
            (prediction_id, old_probability, new_probability, rationale),
        )
        self.conn.execute(
            "UPDATE predictions SET probability = ? WHERE id = ?",
            (new_probability, prediction_id),
        )
        self.conn.commit()

    def resolve_prediction(
        self,
        prediction_id: int,
        outcome: int,
        resolution_source: str,
        resolved_at: str,
    ) -> None:
        """Marque une prédiction comme résolue (issue, source, date)."""
        self.conn.execute(
            "UPDATE predictions SET outcome = ?, resolution_source = ?, "
            "resolved_at = ? WHERE id = ?",
            (outcome, resolution_source, resolved_at, prediction_id),
        )
        self.conn.commit()

    def count_prediction_updates(self, prediction_id: int) -> int:
        """Renvoie le nombre de mises à jour historisées d'une prédiction."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM prediction_updates WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def distinct_categories(self, project_id: int) -> list[str]:
        """Renvoie les catégories déjà utilisées dans un projet."""
        rows = self.conn.execute(
            "SELECT DISTINCT category FROM predictions "
            "WHERE project_id = ? AND category IS NOT NULL AND category <> '' "
            "ORDER BY category",
            (project_id,),
        ).fetchall()
        return [r["category"] for r in rows]

    # --- Graphe de connaissances (entités & relations) -----------------------

    def clear_graph(self, project_id: int) -> None:
        """Supprime entités, relations et mentions d'un projet (avant rebuild)."""
        self.conn.execute("DELETE FROM entity_mentions WHERE project_id = ?", (project_id,))
        self.conn.execute("DELETE FROM relationships WHERE project_id = ?", (project_id,))
        self.conn.execute("DELETE FROM entities WHERE project_id = ?", (project_id,))
        self.conn.commit()

    def add_entity(
        self, project_id: int, name: str, type_: str, first_seen_evidence_id: int | None
    ) -> int:
        """Insère une entité (sans doublon) et renvoie son identifiant.

        Unicité sur (project_id, name, type) : une entité déjà connue n'est jamais
        dupliquée ; sa première occurrence (first_seen_in_evidence_id) est conservée.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO entities "
            "(project_id, name, type, first_seen_in_evidence_id) VALUES (?, ?, ?, ?)",
            (project_id, name, type_, first_seen_evidence_id),
        )
        row = self.conn.execute(
            "SELECT id FROM entities WHERE project_id = ? AND name = ? AND type = ?",
            (project_id, name, type_),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def add_entity_mention(
        self, project_id: int, entity_id: int, evidence_id: int
    ) -> None:
        """Enregistre la présence d'une entité dans une preuve (sans doublon)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_mentions "
            "(project_id, entity_id, evidence_id, as_source) VALUES (?, ?, ?, 0)",
            (project_id, entity_id, evidence_id),
        )
        self.conn.commit()

    def mark_source_mention(
        self, project_id: int, entity_id: int, evidence_id: int
    ) -> None:
        """Marque une entité comme ayant servi de source dans une preuve donnée."""
        self.conn.execute(
            "INSERT INTO entity_mentions "
            "(project_id, entity_id, evidence_id, as_source) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(project_id, entity_id, evidence_id) DO UPDATE SET as_source = 1",
            (project_id, entity_id, evidence_id),
        )
        self.conn.commit()

    def update_entity_metadata(self, entity_id: int, metadata: str) -> None:
        """Met à jour la colonne metadata (JSON) d'une entité."""
        self.conn.execute(
            "UPDATE entities SET metadata = ? WHERE id = ?", (metadata, entity_id)
        )
        self.conn.commit()

    def source_reliability(self, project_id: int) -> tuple[list[sqlite3.Row], int]:
        """Calcule le taux de citation comme source des entités PERSON/ORG.

        Returns:
            (lignes, total_preuves) où chaque ligne a name, type, metadata,
            appears (preuves où l'entité apparaît) et sourced (preuves où elle est
            citée comme source).
        """
        rows = self.conn.execute(
            "SELECT e.id, e.name, e.type, e.metadata, "
            "COUNT(*) AS appears, "
            "SUM(CASE WHEN m.as_source = 1 THEN 1 ELSE 0 END) AS sourced "
            "FROM entity_mentions m JOIN entities e ON e.id = m.entity_id "
            "WHERE m.project_id = ? AND e.type IN ('PERSON', 'ORG') "
            "GROUP BY e.id HAVING sourced > 0 "
            "ORDER BY sourced DESC, appears DESC",
            (project_id,),
        ).fetchall()
        total_row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM evidence_items WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return rows, (int(total_row["n"]) if total_row else 0)

    def add_cooccurrence(
        self, project_id: int, entity_a: int, entity_b: int, evidence_id: int | None
    ) -> None:
        """Ajoute (ou renforce) une relation de co-occurrence non dirigée.

        Le couple est canonisé (id min, id max) pour rester non dirigé ; chaque
        nouvelle co-occurrence incrémente le poids de 1.
        """
        src, tgt = sorted((entity_a, entity_b))
        self.conn.execute(
            "INSERT INTO relationships "
            "(project_id, source_entity_id, target_entity_id, evidence_id, "
            "relation_type, weight) VALUES (?, ?, ?, ?, 'CO_OCCURS', 1.0) "
            "ON CONFLICT(project_id, source_entity_id, target_entity_id, relation_type) "
            "DO UPDATE SET weight = weight + 1.0",
            (project_id, src, tgt, evidence_id),
        )
        self.conn.commit()

    def add_pattern_relation(
        self,
        project_id: int,
        source_id: int,
        target_id: int,
        evidence_id: int | None,
        relation_type: str,
    ) -> None:
        """Ajoute (ou renforce) une relation dirigée détectée par pattern.

        Poids initial 3.0 (plus fort que la co-occurrence) ; +1 à chaque nouvelle
        détection.
        """
        self.conn.execute(
            "INSERT INTO relationships "
            "(project_id, source_entity_id, target_entity_id, evidence_id, "
            "relation_type, weight) VALUES (?, ?, ?, ?, ?, 3.0) "
            "ON CONFLICT(project_id, source_entity_id, target_entity_id, relation_type) "
            "DO UPDATE SET weight = weight + 1.0",
            (project_id, source_id, target_id, evidence_id, relation_type),
        )
        self.conn.commit()

    def list_entities(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les entités d'un projet."""
        return self.conn.execute(
            "SELECT id, name, type, first_seen_in_evidence_id "
            "FROM entities WHERE project_id = ? ORDER BY name ASC",
            (project_id,),
        ).fetchall()

    def list_relationships(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les relations d'un projet."""
        return self.conn.execute(
            "SELECT id, source_entity_id, target_entity_id, relation_type, weight "
            "FROM relationships WHERE project_id = ?",
            (project_id,),
        ).fetchall()

    def shortest_path(
        self, project_id: int, source_id: int, target_id: int
    ) -> list[int] | None:
        """Plus court chemin (en sauts) entre deux entités, via CTE récursive SQLite.

        Le graphe est traité comme non dirigé (arêtes dépliées dans les deux sens).
        Renvoie la liste des identifiants d'entités du chemin, ou None si aucun.
        """
        if source_id == target_id:
            return [source_id]
        query = """
            WITH RECURSIVE
            edges(a, b) AS (
                SELECT source_entity_id, target_entity_id
                FROM relationships WHERE project_id = :pid
                UNION ALL
                SELECT target_entity_id, source_entity_id
                FROM relationships WHERE project_id = :pid
            ),
            paths(node, depth, path) AS (
                SELECT :src, 0, ',' || :src || ','
                UNION ALL
                SELECT e.b, p.depth + 1, p.path || e.b || ','
                FROM edges e JOIN paths p ON e.a = p.node
                WHERE p.depth < 12
                  AND p.path NOT LIKE '%,' || e.b || ',%'
            )
            SELECT path FROM paths WHERE node = :dst ORDER BY depth ASC LIMIT 1
        """
        row = self.conn.execute(
            query, {"pid": project_id, "src": source_id, "dst": target_id}
        ).fetchone()
        if row is None:
            return None
        return [int(x) for x in row["path"].strip(",").split(",") if x]

    # --- Horizons & nœuds de bifurcation (Phase 8) ---------------------------

    def list_horizons(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les horizons d'un projet, dans l'ordre de création."""
        return self.conn.execute(
            "SELECT id, label, years_min, years_max FROM horizons "
            "WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()

    def list_bifurcation_nodes(self, project_id: int) -> list[sqlite3.Row]:
        """Renvoie les nœuds de bifurcation d'un projet, dans l'ordre de création."""
        return self.conn.execute(
            "SELECT id, label, condition_text, parent_hypothesis_id, horizon_id "
            "FROM bifurcation_nodes WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()

    def horizon_id_by_rank(self, project_id: int, rank: int) -> int | None:
        """Renvoie l'id de l'horizon de rang donné (0=court, 1=moyen, 2=long)."""
        horizons = self.list_horizons(project_id)
        if 0 <= rank < len(horizons):
            return int(horizons[rank]["id"])
        return None

    def _get_existing_hypothesis_id(self, project_id: int, label: str) -> int | None:
        """Renvoie l'id d'une hypothèse existante par son label, ou None."""
        if not label:
            return None
        row = self.conn.execute(
            "SELECT id FROM hypotheses WHERE project_id = ? AND label = ?",
            (project_id, label),
        ).fetchone()
        return int(row["id"]) if row else None

    def _get_existing_entity_id(self, project_id: int, name: str) -> int | None:
        """Renvoie l'id d'une entité existante (comparaison normalisée), ou None."""
        target = normalize_entity_name(name)
        for row in self.conn.execute(
            "SELECT id, name FROM entities WHERE project_id = ?", (project_id,)
        ).fetchall():
            if normalize_entity_name(row["name"]) == target:
                return int(row["id"])
        return None

    # --- Détection de doublons (Phase 8 C.1) ---------------------------------

    def find_similar_evidences(
        self, project_id: int, text: str, threshold: float = 0.85
    ) -> list[tuple[int, str, float]]:
        """Renvoie les preuves dont la similarité cosinus (embeddings spaCy) dépasse
        threshold. Retourne [] si spaCy/numpy indisponible (dégradation silencieuse)."""
        try:
            import numpy as np
            import spacy
        except Exception:  # noqa: BLE001
            return []
        try:
            nlp = _get_shared_nlp()
        except Exception:  # noqa: BLE001
            return []
        if nlp is None:
            return []
        target = nlp(text or "").vector
        norm_target = float(np.linalg.norm(target))
        if norm_target == 0:
            return []
        similar: list[tuple[int, str, float]] = []
        for ev_id, content in self.conn.execute(
            "SELECT id, content FROM evidence_items WHERE project_id = ?", (project_id,)
        ).fetchall():
            vec = nlp(content or "").vector
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                continue
            sim = float(np.dot(target, vec)) / (norm_target * norm)
            if sim >= threshold:
                similar.append((int(ev_id), content, round(sim, 3)))
        return sorted(similar, key=lambda x: x[2], reverse=True)

    # --- Enrichissements Yggdrasil (Phase 8 D) -------------------------------

    def compute_probability_history(
        self, project_id: int, hypothesis_row: sqlite3.Row
    ) -> list[dict[str, float | str]]:
        """Reconstruit l'historique de la probabilité bayésienne d'une hypothèse.

        Recalcule le posterior après intégration de chaque preuve (ordre
        chronologique d'ajout). Renvoie [{date, value}].
        """
        hyp_id = hypothesis_row["id"]
        prior = hypothesis_row["prior_probability"]
        if prior is None:
            prior = 0.5
        prior = min(max(prior, 1e-6), 1 - 1e-6)

        created = "2026-01-01"
        try:
            if hypothesis_row["created_at"]:
                created = str(hypothesis_row["created_at"])[:10]
        except (IndexError, KeyError):
            pass

        history: list[dict[str, float | str]] = [
            {"date": created, "value": round(prior, 3)}
        ]
        log_odds = math.log(prior / (1 - prior))
        rows = self.conn.execute(
            "SELECT e.added_at, ms.p_e_given_h, ms.p_e_given_not_h "
            "FROM matrix_scores ms JOIN evidence_items e ON ms.evidence_id = e.id "
            "WHERE ms.hypothesis_id = ? AND ms.p_e_given_h IS NOT NULL "
            "AND ms.p_e_given_not_h IS NOT NULL ORDER BY e.added_at ASC",
            (hyp_id,),
        ).fetchall()
        for r in rows:
            p_h, p_nh = r["p_e_given_h"], r["p_e_given_not_h"]
            if p_nh and p_nh > 0:
                log_odds += math.log(p_h / p_nh)
            posterior = 1 / (1 + math.exp(-log_odds))
            history.append(
                {"date": str(r["added_at"])[:10] if r["added_at"] else created,
                 "value": round(posterior, 3)}
            )
        return history

    def _hypothesis_posterior(self, hyp_id: int, prior: float | None) -> float:
        """Recalcule le posterior bayésien d'une hypothèse depuis ses vraisemblances."""
        if prior is None:
            prior = 0.5
        prior = min(max(prior, 1e-6), 1 - 1e-6)
        log_odds = math.log(prior / (1 - prior))
        for r in self.conn.execute(
            "SELECT p_e_given_h, p_e_given_not_h FROM matrix_scores "
            "WHERE hypothesis_id = ? AND p_e_given_h IS NOT NULL "
            "AND p_e_given_not_h IS NOT NULL",
            (hyp_id,),
        ).fetchall():
            if r["p_e_given_not_h"] and r["p_e_given_not_h"] > 0:
                log_odds += math.log(r["p_e_given_h"] / r["p_e_given_not_h"])
        return 1 / (1 + math.exp(-log_odds))

    def compute_tension_score(
        self, project_id: int, leads_to_labels: list[str]
    ) -> float | None:
        """Score de tension d'un nœud : 1 − (max − min) des posteriors des filles.

        Renvoie None si moins de 2 hypothèses filles résolvables.
        """
        posteriors: list[float] = []
        for label in leads_to_labels:
            row = self.conn.execute(
                "SELECT id, prior_probability FROM hypotheses "
                "WHERE project_id = ? AND label = ?",
                (project_id, label),
            ).fetchone()
            if row is None:
                continue
            posteriors.append(
                self._hypothesis_posterior(row["id"], row["prior_probability"])
            )
        if len(posteriors) < 2:
            return None
        return round(1.0 - (max(posteriors) - min(posteriors)), 3)

    # --- Insertion atomique de l'analyse révisée (Phase 8 B.4) ---------------

    def apply_reviewed_analysis(self, project_id: int, reviewed: dict) -> None:
        """Insère toutes les données validées en une transaction atomique.

        En cas d'erreur sur une insertion, l'ensemble est annulé (ROLLBACK).
        Le dictionnaire `reviewed` suit la structure produite par l'interface de
        révision (chaque élément porte un drapeau `selected`).
        """
        consistency_map = {"CC": 1, "C": 2, "N": 3, "I": 4, "II": 5}
        self.conn.execute("BEGIN")
        try:
            # 1. Hypothèses (label -> id).
            hyp_id_map: dict[str, int] = {}
            for hyp in reviewed.get("hypotheses", []):
                if not hyp.get("selected"):
                    continue
                horizon_id = self._horizon_str_to_id(project_id, hyp.get("horizon"))
                parent_id = self._resolve_parent_id(
                    project_id, hyp.get("parent_hypothesis_label"), hyp_id_map
                )
                cur = self.conn.execute(
                    "INSERT INTO hypotheses (project_id, label, description, "
                    "prior_probability, horizon_id, parent_hypothesis_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, hyp["label"], hyp.get("description", ""),
                     hyp.get("prior_probability", 0.5), horizon_id, parent_id),
                )
                hyp_id_map[hyp["label"]] = int(cur.lastrowid)

            # 2. Preuves + scores ACH + vraisemblances bayésiennes.
            for ev in reviewed.get("evidence_items", []):
                if not ev.get("selected"):
                    continue
                cur = self.conn.execute(
                    "INSERT INTO evidence_items (project_id, content, source, "
                    "credibility) VALUES (?, ?, ?, ?)",
                    (project_id, ev["content"], ev.get("source", ""),
                     ev.get("credibility", 3)),
                )
                ev_id = int(cur.lastrowid)
                for hyp_label, score in (ev.get("ach_scores") or {}).items():
                    hyp_id = hyp_id_map.get(hyp_label) or self._get_existing_hypothesis_id(
                        project_id, hyp_label
                    )
                    consistency = consistency_map.get(score)
                    if hyp_id and consistency:
                        self.conn.execute(
                            "INSERT INTO matrix_scores (project_id, evidence_id, "
                            "hypothesis_id, consistency_score) VALUES (?, ?, ?, ?) "
                            "ON CONFLICT(evidence_id, hypothesis_id) DO UPDATE SET "
                            "consistency_score = excluded.consistency_score",
                            (project_id, ev_id, hyp_id, consistency),
                        )
                for hyp_label, lk in (ev.get("bayesian_likelihoods") or {}).items():
                    hyp_id = hyp_id_map.get(hyp_label) or self._get_existing_hypothesis_id(
                        project_id, hyp_label
                    )
                    if hyp_id and lk.get("p_e_given_h") is not None:
                        self.conn.execute(
                            "INSERT INTO matrix_scores (project_id, evidence_id, "
                            "hypothesis_id, p_e_given_h, p_e_given_not_h) "
                            "VALUES (?, ?, ?, ?, ?) "
                            "ON CONFLICT(evidence_id, hypothesis_id) DO UPDATE SET "
                            "p_e_given_h = excluded.p_e_given_h, "
                            "p_e_given_not_h = excluded.p_e_given_not_h",
                            (project_id, ev_id, hyp_id, lk.get("p_e_given_h"),
                             lk.get("p_e_given_not_h")),
                        )

            # 3. Entités (dédoublonnage normalisé).
            entity_id_map: dict[str, int] = {}
            for ent in reviewed.get("entities", []):
                if not ent.get("selected"):
                    continue
                existing = self._get_existing_entity_id(project_id, ent["name"])
                if existing:
                    entity_id_map[ent["name"]] = existing
                    continue
                cur = self.conn.execute(
                    "INSERT INTO entities (project_id, name, type) VALUES (?, ?, ?)",
                    (project_id, ent["name"],
                     refine_entity_type(ent["name"], ent.get("type", "MISC"))),
                )
                entity_id_map[ent["name"]] = int(cur.lastrowid)

            # 4. Relations.
            for rel in reviewed.get("relations", []):
                if not rel.get("selected"):
                    continue
                src = entity_id_map.get(rel["source"]) or self._get_existing_entity_id(
                    project_id, rel["source"]
                )
                tgt = entity_id_map.get(rel["target"]) or self._get_existing_entity_id(
                    project_id, rel["target"]
                )
                if src and tgt and src != tgt:
                    rtype = refine_relation_type(
                        rel["source"], rel["target"], rel.get("type", "CO_OCCURS")
                    )
                    self.conn.execute(
                        "INSERT INTO relationships (project_id, source_entity_id, "
                        "target_entity_id, relation_type, weight) "
                        "VALUES (?, ?, ?, ?, 3.0) "
                        "ON CONFLICT(project_id, source_entity_id, target_entity_id, "
                        "relation_type) DO UPDATE SET weight = weight + 1.0",
                        (project_id, src, tgt, rtype),
                    )

            # 5. Prédictions.
            for pred in reviewed.get("predictions", []):
                if not pred.get("selected"):
                    continue
                self.conn.execute(
                    "INSERT INTO predictions (project_id, question, probability, "
                    "deadline, category) VALUES (?, ?, ?, ?, ?)",
                    (project_id, pred["question"], pred.get("probability", 0.5),
                     pred.get("deadline"), pred.get("category", "")),
                )

            # 6. Nœuds de bifurcation + rattachement des filles.
            for bn in reviewed.get("bifurcation_nodes", []):
                if not bn.get("selected"):
                    continue
                parent_id = hyp_id_map.get(bn.get("parent_hypothesis")) or \
                    self._get_existing_hypothesis_id(project_id, bn.get("parent_hypothesis"))
                horizon_id = self._horizon_str_to_id(project_id, bn.get("horizon"))
                cur = self.conn.execute(
                    "INSERT INTO bifurcation_nodes (project_id, label, condition_text, "
                    "parent_hypothesis_id, horizon_id) VALUES (?, ?, ?, ?, ?)",
                    (project_id, bn["label"], bn.get("condition_text", ""),
                     parent_id, horizon_id),
                )
                bn_id = int(cur.lastrowid)
                for hyp_label in bn.get("leads_to_hypotheses", []):
                    hyp_id = hyp_id_map.get(hyp_label) or self._get_existing_hypothesis_id(
                        project_id, hyp_label
                    )
                    if hyp_id:
                        self.conn.execute(
                            "UPDATE hypotheses SET bifurcation_node_id = ? WHERE id = ?",
                            (bn_id, hyp_id),
                        )

            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _horizon_str_to_id(self, project_id: int, horizon: str | None) -> int | None:
        """Convertit 'court_terme'/'moyen_terme'/'long_terme' en id d'horizon."""
        mapping = {"court_terme": 0, "moyen_terme": 1, "long_terme": 2}
        rank = mapping.get((horizon or "").strip())
        return self.horizon_id_by_rank(project_id, rank) if rank is not None else None

    def _resolve_parent_id(
        self, project_id: int, parent_label: str | None, hyp_id_map: dict[str, int]
    ) -> int | None:
        """Résout l'id de l'hypothèse parente (nouvelle ou existante)."""
        if not parent_label:
            return None
        return hyp_id_map.get(parent_label) or self._get_existing_hypothesis_id(
            project_id, parent_label
        )

    def backup_to(self, dest_path: Path) -> None:
        """Copie sûre de la base vers dest_path (API backup SQLite, base vivante)."""
        dest = sqlite3.connect(str(dest_path))
        try:
            with dest:
                self.conn.backup(dest)
        finally:
            dest.close()

    def close(self) -> None:
        """Ferme proprement la connexion SQLite (idempotent)."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]


# =============================================================================
#  Vérification / téléchargement du modèle spaCy (gestion de dépendance)
# =============================================================================

class SpacyModelWorker(QThread):
    """Vérifie la présence du modèle spaCy et le télécharge si nécessaire.

    AUCUN traitement NLP : on garantit seulement que `spacy.load(MODEL)` ne lèvera
    pas d'exception en Phase ≥ 5.
    """

    progress = Signal(str)
    finished_check = Signal(bool, str)

    def __init__(self, model_name: str = SPACY_MODEL) -> None:
        """Initialise le worker."""
        super().__init__()
        self.model_name = model_name

    def run(self) -> None:  # noqa: D401 - exécuté par QThread.start()
        """Exécute la vérification puis, au besoin, le téléchargement."""
        try:
            import spacy
        except ImportError:
            self.finished_check.emit(
                False, "spaCy non installé (pip install spacy) — requis dès la Phase 5."
            )
            return

        self.progress.emit(f"Vérification du modèle « {self.model_name} »…")
        try:
            spacy.load(self.model_name)
            self.finished_check.emit(True, f"Modèle « {self.model_name} » présent.")
            return
        except OSError:
            pass

        if getattr(sys, "frozen", False):
            self.finished_check.emit(
                False,
                f"Modèle « {self.model_name} » absent. Téléchargement automatique "
                "indisponible en exécutable empaqueté (à traiter en Phase 5).",
            )
            return

        self.progress.emit(f"Téléchargement du modèle « {self.model_name} »…")
        try:
            from spacy.cli import download

            download(self.model_name)
            spacy.load(self.model_name)
            self.finished_check.emit(
                True, f"Modèle « {self.model_name} » téléchargé et validé."
            )
        except Exception as exc:  # noqa: BLE001
            self.finished_check.emit(False, f"Échec du téléchargement du modèle : {exc}")


class GraphBuildWorker(QThread):
    """Exécute l'extraction NLP (spaCy) sur toutes les preuves d'un projet.

    Tout le traitement NLP a lieu dans ce thread d'arrière-plan ; aucune écriture
    SQLite n'y est faite (sqlite3 n'est pas thread-safe par connexion). Les
    résultats bruts sont émis, puis persistés par le thread principal.
    """

    progress = Signal(str)
    finished_build = Signal(object)  # list[dict]
    failed = Signal(str)

    def __init__(self, evidence: list[tuple[int, str]]) -> None:
        """Initialise le worker avec la liste (id, texte) des preuves."""
        super().__init__()
        self.evidence = evidence

    def run(self) -> None:  # noqa: D401 - exécuté par QThread.start()
        """Charge spaCy, analyse chaque preuve et émet les résultats."""
        try:
            import spacy

            nlp = spacy.load(SPACY_MODEL)
        except ImportError:
            self.failed.emit("spaCy n'est pas installé (pip install spacy).")
            return
        except OSError:
            self.failed.emit(
                f"Modèle « {SPACY_MODEL} » introuvable. Installez-le avec :\n"
                f"python -m spacy download {SPACY_MODEL}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Échec du chargement de spaCy : {exc}")
            return

        matcher = build_matcher(nlp, load_patterns())
        date_matcher = build_date_matcher(nlp)
        results: list[dict] = []
        total = len(self.evidence)
        for i, (evidence_id, text) in enumerate(self.evidence, start=1):
            self.progress.emit(f"Analyse NLP {i}/{total}…")
            try:
                entities, relations, entity_meta = analyze_text(
                    nlp, matcher, date_matcher, text or ""
                )
            except Exception:  # noqa: BLE001 - une preuve ne doit pas tout bloquer
                entities, relations, entity_meta = [], [], []
            results.append(
                {
                    "evidence_id": evidence_id,
                    "entities": entities,
                    "relations": relations,
                    "entity_meta": entity_meta,
                }
            )
        self.finished_build.emit(results)


class SummaryWorker(QThread):
    """Calcule un résumé extractif (TextRank) et des mots-clés (TF-IDF).

    Le NLP s'exécute dans ce thread ; aucun accès SQLite. Le TF-IDF est calculé
    sur tout le corpus du projet pour pondérer correctement les termes.
    """

    finished_summary = Signal(object)  # dict
    failed = Signal(str)

    def __init__(self, target_index: int, corpus: list[tuple[int, str]]) -> None:
        """Initialise avec l'indice de la preuve cible et le corpus (id, texte)."""
        super().__init__()
        self.target_index = target_index
        self.corpus = corpus

    def run(self) -> None:  # noqa: D401 - exécuté par QThread.start()
        """Charge spaCy, calcule résumé et mots-clés, émet le résultat."""
        try:
            import spacy

            nlp = spacy.load(SPACY_MODEL)
        except ImportError:
            self.failed.emit("spaCy n'est pas installé (pip install spacy).")
            return
        except OSError:
            self.failed.emit(
                f"Modèle « {SPACY_MODEL} » introuvable. Installez-le avec :\n"
                f"python -m spacy download {SPACY_MODEL}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Échec du chargement de spaCy : {exc}")
            return

        texts = [t or "" for _, t in self.corpus]
        try:
            docs = list(nlp.pipe(texts))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Échec de l'analyse NLP : {exc}")
            return

        corpus_lemmas = [content_lemmas(doc) for doc in docs]
        tfidf = compute_tfidf(corpus_lemmas)
        target_doc = docs[self.target_index]
        sentences, selected = summarize_doc(target_doc)
        keywords = [w for w, _ in top_keywords(tfidf[self.target_index], 10)]
        self.finished_summary.emit(
            {"sentences": sentences, "selected": selected, "keywords": keywords}
        )


class ReportSummaryWorker(QThread):
    """Calcule les résumés TextRank de toutes les preuves pour le rapport Markdown."""

    done = Signal(object)  # dict {evidence_id: résumé}
    failed = Signal(str)

    def __init__(self, evidence: list[tuple[int, str]]) -> None:
        """Initialise avec la liste (id, texte) des preuves."""
        super().__init__()
        self.evidence = evidence

    def run(self) -> None:  # noqa: D401
        """Résume chaque preuve. Utilise spaCy si possible, sinon un repli
        pur-Python (TextRank TF-IDF) pour que les résumés restent disponibles
        même si spaCy est absent ou incompatible avec le modèle."""
        summaries: dict[int, str] = {}
        nlp = None
        try:
            import spacy

            nlp = spacy.load(SPACY_MODEL)
        except Exception:  # noqa: BLE001 - spaCy absent/incompatible : repli
            nlp = None

        if nlp is not None:
            try:
                docs = list(nlp.pipe([t or "" for _, t in self.evidence]))
                for (evidence_id, _), doc in zip(self.evidence, docs):
                    sentences, selected = summarize_doc(doc)
                    summaries[evidence_id] = " ".join(sentences[i] for i in selected)
                self.done.emit(summaries)
                return
            except Exception:  # noqa: BLE001 - échec en cours de traitement : repli
                summaries = {}

        # Repli pur-Python (sans spaCy).
        for evidence_id, text in self.evidence:
            sentences, selected = summarize_text_plain(text or "")
            summaries[evidence_id] = " ".join(sentences[i] for i in selected)
        self.done.emit(summaries)


# =============================================================================
#  Dialogues — projets, hypothèses, preuves, sensibilité, bayésien
# =============================================================================

class NewProjectDialog(QDialog):
    """Boîte de dialogue de création d'un projet (nom + description)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Construit le dialogue et ses champs."""
        super().__init__(parent)
        self.setWindowTitle("Nouveau projet")
        self.setMinimumWidth(440)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Ex. : Origine de la panne réseau du 12/06")
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Description (facultatif)")
        self.description_edit.setMinimumHeight(120)

        form = QFormLayout()
        form.addRow("Nom du projet :", self.name_edit)
        form.addRow("Description :", self.description_edit)

        self.buttons = QDialogButtonBox()
        self.create_btn = self.buttons.addButton(
            "Créer", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.create_btn.setEnabled(False)
        self.name_edit.textChanged.connect(
            lambda text: self.create_btn.setEnabled(bool(text.strip()))
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def get_data(self) -> tuple[str, str]:
        """Renvoie (nom, description)."""
        return (
            self.name_edit.text().strip(),
            self.description_edit.toPlainText().strip(),
        )


class AddHypothesisDialog(QDialog):
    """Dialogue d'ajout d'une hypothèse (label, description, probabilité a priori).

    Les trois champs sont réunis dans un formulaire unique ; un QInputDialog ne
    pouvant héberger un QTextEdit ni un QDoubleSpinBox, le label utilise un
    QLineEdit (équivalent fonctionnel du champ texte d'un QInputDialog).
    """

    def __init__(self, suggested_label: str = "", parent: QWidget | None = None) -> None:
        """Construit le dialogue."""
        super().__init__(parent)
        self.setWindowTitle("Ajouter une hypothèse")
        self.setMinimumWidth(460)

        self.label_edit = QLineEdit(suggested_label)
        self.label_edit.setPlaceholderText("Ex. : H1")
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Énoncé de l'hypothèse")
        self.description_edit.setMinimumHeight(110)
        self.prior_spin = QDoubleSpinBox()
        self.prior_spin.setRange(0.01, 0.99)
        self.prior_spin.setSingleStep(0.01)
        self.prior_spin.setDecimals(2)
        self.prior_spin.setValue(0.50)

        form = QFormLayout()
        form.addRow("Label :", self.label_edit)
        form.addRow("Description :", self.description_edit)
        form.addRow("Probabilité a priori :", self.prior_spin)

        self.buttons = QDialogButtonBox()
        self.create_btn = self.buttons.addButton(
            "Ajouter", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.create_btn.setEnabled(bool(suggested_label.strip()))
        self.label_edit.textChanged.connect(
            lambda text: self.create_btn.setEnabled(bool(text.strip()))
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def get_data(self) -> tuple[str, str, float]:
        """Renvoie (label, description, probabilité a priori)."""
        return (
            self.label_edit.text().strip(),
            self.description_edit.toPlainText().strip(),
            float(self.prior_spin.value()),
        )


class AddEvidenceDialog(QDialog):
    """Dialogue d'ajout d'une preuve (contenu, source, crédibilité 1-5)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Construit le dialogue."""
        super().__init__(parent)
        self.setWindowTitle("Ajouter une preuve")
        self.setMinimumWidth(460)

        self.content_edit = QTextEdit()
        self.content_edit.setPlaceholderText("Contenu de la preuve / de l'indice")
        self.content_edit.setMinimumHeight(110)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("URL, document, témoignage…")
        self.credibility_spin = QSpinBox()
        self.credibility_spin.setRange(1, 5)
        self.credibility_spin.setValue(3)

        form = QFormLayout()
        form.addRow("Contenu :", self.content_edit)
        form.addRow("Source :", self.source_edit)
        form.addRow("Crédibilité (1-5) :", self.credibility_spin)

        self.buttons = QDialogButtonBox()
        self.create_btn = self.buttons.addButton(
            "Ajouter", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.create_btn.setEnabled(False)
        self.content_edit.textChanged.connect(
            lambda: self.create_btn.setEnabled(
                bool(self.content_edit.toPlainText().strip())
            )
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def get_data(self) -> tuple[str, str, int]:
        """Renvoie (contenu, source, crédibilité)."""
        return (
            self.content_edit.toPlainText().strip(),
            self.source_edit.text().strip(),
            int(self.credibility_spin.value()),
        )


class SensitivityDialog(QDialog):
    """Affiche le rapport d'analyse de sensibilité (lecture seule)."""

    def __init__(self, report: str, parent: QWidget | None = None) -> None:
        """Construit le dialogue."""
        super().__init__(parent)
        self.setWindowTitle("Analyse de sensibilité")
        self.setMinimumSize(560, 420)

        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(report)
        view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(view)
        layout.addWidget(buttons)


class EvidencePickerDialog(QDialog):
    """Petit sélecteur : choisir la preuve à résumer."""

    def __init__(self, evidence: list[sqlite3.Row], parent: QWidget | None = None) -> None:
        """Construit le sélecteur à partir des preuves du projet."""
        super().__init__(parent)
        self.setWindowTitle("Résumer une preuve")
        self.setMinimumWidth(460)

        self.combo = QComboBox()
        for i, e in enumerate(evidence):
            preview = (e["content"] or "").strip().replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:70] + "…"
            self.combo.addItem(f"E{i + 1} — {preview}", e["id"])

        form = QFormLayout()
        form.addRow("Preuve :", self.combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def selected_id(self) -> int | None:
        """Renvoie l'identifiant de la preuve choisie."""
        return self.combo.currentData()


class SummaryDialog(QDialog):
    """Affiche le résumé extractif (phrases surlignées) et les mots-clés TF-IDF."""

    def __init__(
        self,
        evidence_label: str,
        sentences: list[str],
        selected: list[int],
        keywords: list[str],
        parent: QWidget | None = None,
    ) -> None:
        """Construit le dialogue de résumé."""
        super().__init__(parent)
        self.setWindowTitle(f"Résumé extractif — {evidence_label}")
        self.setMinimumSize(640, 540)
        selected_set = set(selected)

        # Texte source avec les phrases retenues surlignées (ordre original).
        source = QTextEdit()
        source.setReadOnly(True)
        parts = []
        for i, sent in enumerate(sentences):
            text = escape(sent)
            if i in selected_set:
                parts.append(
                    f"<span style='background:#fff3b0;font-weight:600'>{text}</span>"
                )
            else:
                parts.append(f"<span style='color:#666'>{text}</span>")
        source.setHtml(" ".join(parts) if parts else "<i>Texte vide.</i>")

        # Résumé = phrases retenues, dans l'ordre du texte.
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(
            " ".join(sentences[i] for i in selected) if selected else "—"
        )
        summary.setMaximumHeight(150)

        kw_label = QLabel(
            "Mots-clés (TF-IDF) : " + (", ".join(keywords) if keywords else "—")
        )
        kw_label.setWordWrap(True)
        kw_label.setStyleSheet("font-weight: 600;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Texte source</b> (phrases retenues surlignées) :"))
        layout.addWidget(source, 1)
        layout.addWidget(QLabel("<b>Résumé extractif</b> :"))
        layout.addWidget(summary)
        layout.addWidget(kw_label)
        layout.addWidget(buttons)


class BayesianIntegrationDialog(QDialog):
    """Saisie des vraisemblances P(E|H) et P(E|¬H) d'une preuve, par hypothèse."""

    def __init__(
        self,
        evidence_label: str,
        evidence_content: str,
        hypotheses: list[sqlite3.Row],
        existing: dict[int, tuple[float | None, float | None]],
        parent: QWidget | None = None,
    ) -> None:
        """Construit le dialogue."""
        super().__init__(parent)
        self.setWindowTitle(f"Calcul bayésien — {evidence_label}")
        self.setMinimumSize(640, 480)
        self._combos: dict[int, tuple[QComboBox, QComboBox]] = {}

        intro = QLabel(
            f"<b>{escape(evidence_label)}</b> — {escape(evidence_content)}<br>"
            "<span style='color:#666'>Pour chaque hypothèse, indiquez la "
            "probabilité d'observer cette preuve si l'hypothèse est vraie, "
            "puis si elle est fausse.</span>"
        )
        intro.setWordWrap(True)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("<b>Hypothèse</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Si VRAIE : P(E|H)</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Si FAUSSE : P(E|¬H)</b>"), 0, 2)

        for row, hyp in enumerate(hypotheses, start=1):
            name = QLabel(hyp["label"])
            if hyp["description"]:
                name.setToolTip(hyp["description"])
            cur_h, cur_not_h = existing.get(hyp["id"], (None, None))
            combo_h = self._make_likelihood_combo(cur_h)
            combo_not_h = self._make_likelihood_combo(cur_not_h)
            grid.addWidget(name, row, 0)
            grid.addWidget(combo_h, row, 1)
            grid.addWidget(combo_not_h, row, 2)
            self._combos[hyp["id"]] = (combo_h, combo_not_h)
        grid.setRowStretch(len(hypotheses) + 1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)

    @staticmethod
    def _make_likelihood_combo(current: float | None) -> QComboBox:
        """Construit une combobox d'échelle 1-5 (avec option « non défini »)."""
        combo = QComboBox()
        for text, value in LIKELIHOOD_OPTIONS:
            combo.addItem(text, value)
        if current is not None:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data is not None and abs(data - current) < 1e-6:
                    combo.setCurrentIndex(i)
                    break
        return combo

    def get_data(self) -> dict[int, tuple[float | None, float | None]]:
        """Renvoie, par hypothèse, le couple (P(E|H), P(E|¬H))."""
        return {
            hyp_id: (combo_h.currentData(), combo_not_h.currentData())
            for hyp_id, (combo_h, combo_not_h) in self._combos.items()
        }


# =============================================================================
#  Dialogues — prédictions (Phase 4)
# =============================================================================

class NewPredictionDialog(QDialog):
    """Dialogue de création d'une prédiction."""

    def __init__(self, categories: list[str], parent: QWidget | None = None) -> None:
        """Construit le dialogue.

        Args:
            categories: catégories proposées dans la liste déroulante (éditable).
            parent: widget parent.
        """
        super().__init__(parent)
        self.setWindowTitle("Nouvelle prédiction")
        self.setMinimumWidth(480)

        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText(
            "Ex. : Le taux directeur sera relevé avant fin Q3 ?"
        )

        self.probability_spin = QSpinBox()
        self.probability_spin.setRange(0, 100)
        self.probability_spin.setSuffix(" %")
        self.probability_spin.setValue(50)

        self.deadline_edit = QDateEdit()
        self.deadline_edit.setCalendarPopup(True)
        self.deadline_edit.setDisplayFormat("yyyy-MM-dd")
        self.deadline_edit.setDate(QDate.currentDate().addDays(30))

        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.category_combo.addItems(categories)
        self.category_combo.setCurrentText("")

        form = QFormLayout()
        form.addRow("Question :", self.question_edit)
        form.addRow("Probabilité initiale :", self.probability_spin)
        form.addRow("Échéance :", self.deadline_edit)
        form.addRow("Catégorie :", self.category_combo)

        self.buttons = QDialogButtonBox()
        self.create_btn = self.buttons.addButton(
            "Créer", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.create_btn.setEnabled(False)
        self.question_edit.textChanged.connect(
            lambda text: self.create_btn.setEnabled(bool(text.strip()))
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

    def get_data(self) -> tuple[str, float, str, str]:
        """Renvoie (question, probabilité [0-1], échéance ISO, catégorie)."""
        return (
            self.question_edit.text().strip(),
            self.probability_spin.value() / 100.0,
            self.deadline_edit.date().toString("yyyy-MM-dd"),
            self.category_combo.currentText().strip(),
        )


class UpdatePredictionDialog(QDialog):
    """Dialogue de mise à jour de la probabilité d'une prédiction."""

    def __init__(
        self, question: str, current_probability: float, parent: QWidget | None = None
    ) -> None:
        """Construit le dialogue.

        Args:
            question: intitulé de la prédiction (rappel).
            current_probability: probabilité courante [0-1].
            parent: widget parent.
        """
        super().__init__(parent)
        self.setWindowTitle("Mettre à jour la prédiction")
        self.setMinimumWidth(460)

        intro = QLabel(
            f"<b>{escape(question)}</b><br>"
            f"<span style='color:#666'>Probabilité actuelle : "
            f"{current_probability * 100:.0f} %</span>"
        )
        intro.setWordWrap(True)

        self.probability_spin = QSpinBox()
        self.probability_spin.setRange(0, 100)
        self.probability_spin.setSuffix(" %")
        self.probability_spin.setValue(round(current_probability * 100))

        self.rationale_edit = QTextEdit()
        self.rationale_edit.setPlaceholderText(
            "Justification de la révision (nouvel élément, source…)"
        )
        self.rationale_edit.setMinimumHeight(100)

        form = QFormLayout()
        form.addRow("Nouvelle probabilité :", self.probability_spin)
        form.addRow("Justification :", self.rationale_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def get_data(self) -> tuple[float, str]:
        """Renvoie (nouvelle probabilité [0-1], justification)."""
        return (
            self.probability_spin.value() / 100.0,
            self.rationale_edit.toPlainText().strip(),
        )


class ResolvePredictionDialog(QDialog):
    """Dialogue de résolution d'une prédiction (issue, source, date)."""

    def __init__(self, question: str, parent: QWidget | None = None) -> None:
        """Construit le dialogue."""
        super().__init__(parent)
        self.setWindowTitle("Résoudre la prédiction")
        self.setMinimumWidth(460)

        intro = QLabel(f"<b>{escape(question)}</b>")
        intro.setWordWrap(True)

        self.outcome_combo = QComboBox()
        self.outcome_combo.addItem("Oui — réalisé", 1)
        self.outcome_combo.addItem("Non — non réalisé", 0)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("URL ou référence confirmant l'issue")

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate())

        form = QFormLayout()
        form.addRow("Issue :", self.outcome_combo)
        form.addRow("Source de résolution :", self.source_edit)
        form.addRow("Date de résolution :", self.date_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def get_data(self) -> tuple[int, str, str]:
        """Renvoie (issue 0/1, source, date ISO)."""
        return (
            int(self.outcome_combo.currentData()),
            self.source_edit.text().strip(),
            self.date_edit.date().toString("yyyy-MM-dd"),
        )


# =============================================================================
#  Diagramme en barres a priori vs a posteriori (QPainter, sans dépendance)
# =============================================================================

class BayesianBarChart(QWidget):
    """Diagramme horizontal comparant probabilité a priori et a posteriori."""

    PRIOR_COLOR = QColor("#a9c6e8")
    POST_COLOR = QColor("#2f6fb3")

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise le diagramme (sans données)."""
        super().__init__(parent)
        self._data: list[tuple[str, float, float]] = []
        self.setMinimumHeight(150)

    def set_data(self, data: list[tuple[str, float, float]]) -> None:
        """Définit les données (label, a priori, a posteriori) et redessine."""
        self._data = data
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: D102
        return QSize(420, 44 + max(1, len(self._data)) * 46)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return QSize(300, 120)

    def paintEvent(self, event) -> None:  # noqa: D102, ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if not self._data:
            painter.setPen(QColor("#999"))
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter, "Aucune donnée bayésienne"
            )
            return

        legend_font = QFont(self.font())
        legend_font.setPointSizeF(max(8.0, self.font().pointSizeF() - 1))
        painter.setFont(legend_font)
        painter.fillRect(QRectF(8, 8, 12, 12), self.PRIOR_COLOR)
        painter.setPen(QColor("#333"))
        painter.drawText(QRectF(24, 6, 80, 16), Qt.AlignmentFlag.AlignVCenter, "a priori")
        painter.fillRect(QRectF(110, 8, 12, 12), self.POST_COLOR)
        painter.drawText(
            QRectF(126, 6, 90, 16), Qt.AlignmentFlag.AlignVCenter, "a posteriori"
        )

        left_margin = 78.0
        right_margin = 52.0
        top = 30.0
        bottom = 8.0
        n = len(self._data)
        avail_h = max(1.0, rect.height() - top - bottom)
        row_h = avail_h / n
        bar_area_w = max(1.0, rect.width() - left_margin - right_margin)
        bar_h = min(13.0, row_h / 2.0 - 3.0)

        for i, (label, prior, posterior) in enumerate(self._data):
            prior = min(max(prior, 0.0), 1.0)
            posterior = min(max(posterior, 0.0), 1.0)
            y = top + i * row_h

            painter.setPen(QColor("#222"))
            painter.drawText(
                QRectF(2, y, left_margin - 8, row_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                label,
            )
            painter.fillRect(
                QRectF(left_margin, y + row_h / 2 - bar_h - 1, bar_area_w * prior, bar_h),
                self.PRIOR_COLOR,
            )
            painter.fillRect(
                QRectF(left_margin, y + row_h / 2 + 1, bar_area_w * posterior, bar_h),
                self.POST_COLOR,
            )
            painter.setPen(QColor("#222"))
            painter.drawText(
                QRectF(rect.width() - right_margin + 2, y, right_margin - 4, row_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"{posterior * 100:.0f} %",
            )


# =============================================================================
#  Diagramme de fiabilité (matplotlib, FigureCanvasQTAgg)
# =============================================================================

if MATPLOTLIB_AVAILABLE:

    class ReliabilityCanvas(FigureCanvasQTAgg):
        """Diagramme de fiabilité : probabilité annoncée vs fréquence observée."""

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialise la figure matplotlib intégrée."""
            self.figure = Figure(figsize=(4.2, 3.2), layout="tight")
            super().__init__(self.figure)
            self.setParent(parent)
            self.ax = self.figure.add_subplot(111)
            self.setMinimumHeight(240)
            self.plot([])

        def plot(self, bins: list[dict[str, float | int]]) -> None:
            """Trace les déciles fournis + la diagonale de calibration parfaite."""
            self.ax.clear()
            self.ax.plot(
                [0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1,
                label="calibration parfaite",
            )
            if bins:
                xs = [b["mean_p"] for b in bins]
                ys = [b["obs"] for b in bins]
                self.ax.plot(
                    xs, ys, "-o", color="#2f6fb3", linewidth=1.6, markersize=6,
                    label="observé",
                )
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel("Probabilité annoncée")
            self.ax.set_ylabel("Fréquence observée")
            self.ax.set_title("Diagramme de fiabilité")
            self.ax.grid(True, linewidth=0.3, alpha=0.5)
            self.ax.legend(fontsize=8, loc="best")
            self.draw()


# =============================================================================
#  Canvas du graphe de connaissances (matplotlib ; networkx pour la disposition)
# =============================================================================

# Couleurs des nœuds par type d'entité.
ENTITY_COLORS: dict[str, str] = {
    "PERSON": "#2f6fb3",  # bleu
    "ORG": "#c0392b",     # rouge
    "LOC": "#27ae60",     # vert
    "MISC": "#7f8c8d",    # gris
    "DATE": "#f1c40f",    # jaune
}
COMMUNITY_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
]

if MATPLOTLIB_AVAILABLE:

    class GraphCanvas(FigureCanvasQTAgg):
        """Rendu matplotlib du graphe (nœuds = entités, arêtes = relations).

        Disposition calculée par networkx si disponible, sinon par une routine
        Fruchterman-Reingold pure et déterministe. Double-clic sur un nœud =
        surlignage de ses voisins directs.
        """

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialise la figure et connecte l'événement de clic."""
            self.figure = Figure(figsize=(5, 4), layout="tight")
            super().__init__(self.figure)
            self.setParent(parent)
            self.setMinimumHeight(360)
            self.ax = self.figure.add_subplot(111)

            self._entities: list[sqlite3.Row] = []
            self._rels: list[sqlite3.Row] = []
            self._pos: dict[int, tuple[float, float]] = {}
            self._adj: dict[int, set[int]] = {}
            self._degree: dict[int, int] = {}
            self._name: dict[int, str] = {}
            self._etype: dict[int, str] = {}
            self._communities: dict[int, int] = {}
            self._color_mode = "type"  # ou "community"
            self._highlight: int | None = None

            self.mpl_connect("button_press_event", self._on_click)
            self._render_empty()

        def _render_empty(self) -> None:
            """Affiche un message lorsque le graphe est vide."""
            self.ax.clear()
            self.ax.text(
                0.5, 0.5, "Graphe vide — cliquez sur « Construire le graphe ».",
                ha="center", va="center", color="#888", fontsize=10,
            )
            self.ax.axis("off")
            self.draw()

        def set_graph(
            self,
            entities: list[sqlite3.Row],
            relationships: list[sqlite3.Row],
            communities: dict[int, int] | None = None,
            color_mode: str = "type",
        ) -> None:
            """Charge un nouveau graphe, calcule la disposition et le dessine."""
            self._entities = entities
            self._rels = relationships
            self._communities = communities or {}
            self._color_mode = color_mode
            self._highlight = None
            self._name = {e["id"]: e["name"] for e in entities}
            self._etype = {e["id"]: (e["type"] or "MISC") for e in entities}

            self._adj = {e["id"]: set() for e in entities}
            self._degree = {e["id"]: 0 for e in entities}
            for r in relationships:
                s, t = r["source_entity_id"], r["target_entity_id"]
                if s in self._adj and t in self._adj:
                    self._adj[s].add(t)
                    self._adj[t].add(s)
            for nid in self._degree:
                self._degree[nid] = len(self._adj[nid])

            self._pos = self._compute_layout()
            self._draw()

        def set_color_mode(self, mode: str, communities: dict[int, int] | None = None) -> None:
            """Bascule entre coloration par type et par communauté."""
            self._color_mode = mode
            if communities is not None:
                self._communities = communities
            self._draw()

        def _compute_layout(self) -> dict[int, tuple[float, float]]:
            """Calcule les positions des nœuds (networkx si présent, sinon pur)."""
            ids = [e["id"] for e in self._entities]
            if not ids:
                return {}
            edges = [
                (r["source_entity_id"], r["target_entity_id"], r["weight"] or 1.0)
                for r in self._rels
                if r["source_entity_id"] in self._adj
                and r["target_entity_id"] in self._adj
            ]
            if NETWORKX_AVAILABLE:
                graph = nx.Graph()
                graph.add_nodes_from(ids)
                for s, t, w in edges:
                    graph.add_edge(s, t, weight=w)
                try:
                    if len(ids) < 50 and graph.number_of_edges() > 0:
                        return nx.kamada_kawai_layout(graph)
                    return nx.spring_layout(graph, seed=42, weight="weight")
                except Exception:  # noqa: BLE001 - repli déterministe
                    return spring_layout_pure(ids, edges)
            return spring_layout_pure(ids, edges)

        def _node_color(self, nid: int) -> str:
            """Renvoie la couleur d'un nœud selon le mode courant."""
            if self._color_mode == "community" and self._communities:
                cid = self._communities.get(nid, 0)
                return COMMUNITY_PALETTE[cid % len(COMMUNITY_PALETTE)]
            return ENTITY_COLORS.get(self._etype.get(nid, "MISC"), "#7f8c8d")

        def _draw(self) -> None:
            """Dessine arêtes, nœuds et étiquettes (avec surlignage éventuel)."""
            self.ax.clear()
            if not self._entities:
                self._render_empty()
                return

            highlight = self._highlight
            focus = None
            if highlight is not None:
                focus = {highlight} | self._adj.get(highlight, set())

            # Arêtes.
            for r in self._rels:
                s, t = r["source_entity_id"], r["target_entity_id"]
                if s not in self._pos or t not in self._pos:
                    continue
                x = [self._pos[s][0], self._pos[t][0]]
                y = [self._pos[s][1], self._pos[t][1]]
                weight = r["weight"] or 1.0
                lw = 0.5 + 0.35 * float(weight)
                dim = focus is not None and not (s in focus and t in focus)
                self.ax.plot(
                    x, y, color="#e2e2e2" if dim else "#b5b5b5",
                    linewidth=lw, zorder=1, solid_capstyle="round",
                )

            # Nœuds + étiquettes.
            for e in self._entities:
                nid = e["id"]
                if nid not in self._pos:
                    continue
                x, y = self._pos[nid]
                size = 90 + 55 * self._degree.get(nid, 0)
                dim = focus is not None and nid not in focus
                alpha = 0.22 if dim else 1.0
                self.ax.scatter(
                    [x], [y], s=size, c=self._node_color(nid), alpha=alpha,
                    edgecolors="white", linewidths=0.8, zorder=2,
                )
                self.ax.annotate(
                    self._name.get(nid, ""), (x, y),
                    fontsize=7, alpha=0.35 if dim else 0.9,
                    xytext=(0, 7), textcoords="offset points", ha="center", zorder=3,
                )

            self.ax.axis("off")
            self.ax.margins(0.12)
            self.draw()

        def _on_click(self, event) -> None:  # noqa: ANN001
            """Double-clic : surligne le nœud le plus proche et ses voisins."""
            if not event.dblclick or event.inaxes != self.ax:
                return
            if event.xdata is None or not self._pos:
                return
            xs = [p[0] for p in self._pos.values()]
            ys = [p[1] for p in self._pos.values()]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
            threshold = (0.07 * span) ** 2

            best, best_d = None, float("inf")
            for nid, (x, y) in self._pos.items():
                d = (x - event.xdata) ** 2 + (y - event.ydata) ** 2
                if d < best_d:
                    best_d, best = d, nid
            if best is not None and best_d <= threshold:
                self._highlight = None if self._highlight == best else best
            else:
                self._highlight = None
            self._draw()


# =============================================================================
#  Vue ACH + bayésien (onglet « Matrice ACH »)
# =============================================================================

class ACHView(QWidget):
    """Matrice ACH d'un projet, augmentée du moteur bayésien."""

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        """Construit la vue (état initial : aucun projet chargé)."""
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None
        self._hyps: list[sqlite3.Row] = []
        self._evs: list[sqlite3.Row] = []
        self._hyp_ids: list[int] = []
        self._cells: dict[tuple[int, int], dict[str, float | None]] = {}
        self._scores: dict[tuple[int, int], int] = {}
        self._likelihoods: dict[tuple[int, int], tuple[float, float]] = {}
        self._priors: dict[int, float | None] = {}
        self._combos: dict[tuple[int, int], QComboBox] = {}
        self._bf_labels: dict[tuple[int, int], QLabel] = {}

        self.header = QLabel("Sélectionnez un projet ou créez-en un nouveau")
        self.header.setStyleSheet("font-size: 16px; font-weight: 600; margin: 4px 0;")

        self.add_hyp_btn = QPushButton("➕ Hypothèse")
        self.add_ev_btn = QPushButton("➕ Preuve")
        self.summarize_btn = QPushButton("📝 Résumer une preuve")
        self.sensitivity_btn = QPushButton("🔬 Analyse de sensibilité")
        self.add_hyp_btn.clicked.connect(self.add_hypothesis)
        self.add_ev_btn.clicked.connect(self.add_evidence)
        self.summarize_btn.clicked.connect(self.summarize_evidence)
        self.sensitivity_btn.clicked.connect(self.run_sensitivity)
        button_row = QHBoxLayout()
        button_row.addWidget(self.add_hyp_btn)
        button_row.addWidget(self.add_ev_btn)
        button_row.addWidget(self.summarize_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.sensitivity_btn)

        self.empty_label = QLabel(
            "Ajoutez au moins une hypothèse et une preuve pour construire la matrice."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #777;")

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setDefaultSectionSize(58)

        self.ranking_label = QLabel("Classement ACH : —")
        self.ranking_label.setStyleSheet("font-weight: 600; margin-top: 4px;")
        self.ranking_label.setWordWrap(True)

        bayes_title = QLabel("Probabilités bayésiennes")
        bayes_title.setStyleSheet("font-weight: 600; margin-top: 6px;")

        self.bayes_table = QTableWidget(0, 4)
        self.bayes_table.setHorizontalHeaderLabels(
            ["Hypothèse", "A priori", "Vraisemblance cumulée", "A posteriori"]
        )
        self.bayes_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.bayes_table.verticalHeader().setVisible(False)
        self.bayes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.bayes_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.bayes_table.setMinimumHeight(120)

        self.bar_chart = BayesianBarChart()

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.ranking_label)
        bottom_layout.addWidget(bayes_title)
        bottom_layout.addWidget(self.bayes_table)
        bottom_layout.addWidget(self.bar_chart, 1)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(bottom)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(button_row)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.splitter, 1)

        self._set_controls_enabled(False)
        self.splitter.hide()

    # --- API publique --------------------------------------------------------

    def set_database(self, db: Database) -> None:
        """Remplace la base de données active et réinitialise la vue."""
        self.db = db
        self.clear()

    def load_project(self, project_id: int) -> None:
        """Charge un projet et reconstruit la matrice depuis la base."""
        project = self.db.get_project(project_id)
        if project is None:
            self.clear()
            return
        self.project_id = project_id
        self.header.setText(escape(project["name"] or "Projet"))
        self._set_controls_enabled(True)
        self._rebuild()

    def clear(self) -> None:
        """Réinitialise la vue à l'état « aucun projet »."""
        self.project_id = None
        self.header.setText("Sélectionnez un projet ou créez-en un nouveau")
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._combos.clear()
        self._bf_labels.clear()
        self.bayes_table.setRowCount(0)
        self.bar_chart.set_data([])
        self.splitter.hide()
        self.empty_label.setText(
            "Ajoutez au moins une hypothèse et une preuve pour construire la matrice."
        )
        self.empty_label.show()
        self.ranking_label.setText("Classement ACH : —")
        self._set_controls_enabled(False)

    # --- Dérivation des structures de calcul ---------------------------------

    def _derive(self) -> None:
        """Recalcule les vues dérivées (_scores, _likelihoods, _priors)."""
        self._scores = {
            k: int(v["score"]) for k, v in self._cells.items() if v["score"] is not None
        }
        self._likelihoods = {
            k: (float(v["p_h"]), float(v["p_not_h"]))
            for k, v in self._cells.items()
            if v["p_h"] is not None and v["p_not_h"] is not None
        }
        self._priors = {
            h["id"]: (
                h["prior_probability"] if h["prior_probability"] is not None else 0.5
            )
            for h in self._hyps
        }

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Active/désactive les boutons d'action."""
        self.add_hyp_btn.setEnabled(enabled)
        self.add_ev_btn.setEnabled(enabled)
        self.summarize_btn.setEnabled(enabled)
        self.sensitivity_btn.setEnabled(enabled)

    def _rebuild(self) -> None:
        """Reconstruit entièrement la matrice à partir de la base."""
        assert self.project_id is not None
        self._hyps = self.db.list_hypotheses(self.project_id)
        self._evs = self.db.list_evidence(self.project_id)
        self._hyp_ids = [h["id"] for h in self._hyps]
        self._cells = self.db.get_cells(self.project_id)
        self._derive()

        if not self._hyps or not self._evs:
            self.splitter.hide()
            missing = []
            if not self._hyps:
                missing.append("au moins une hypothèse")
            if not self._evs:
                missing.append("au moins une preuve")
            self.empty_label.setText("Ajoutez " + " et ".join(missing) + ".")
            self.empty_label.show()
            self.bayes_table.setRowCount(0)
            self.bar_chart.set_data([])
            self.ranking_label.setText("Classement ACH : —")
            return

        self.empty_label.hide()
        self.splitter.show()

        n_ev = len(self._evs)
        n_hyp = len(self._hyps)
        diag_col = n_hyp
        bayes_col = n_hyp + 1

        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._combos.clear()
        self._bf_labels.clear()

        self.table.setRowCount(n_ev + 1)
        self.table.setColumnCount(n_hyp + 2)
        self.table.setHorizontalHeaderLabels(
            [h["label"] for h in self._hyps] + ["Diagnosticité", "Bayésien"]
        )

        header = self.table.horizontalHeader()
        for c in range(n_hyp):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(diag_col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(bayes_col, QHeaderView.ResizeMode.ResizeToContents)

        for r, e in enumerate(self._evs):
            item = QTableWidgetItem(f"E{r + 1}")
            item.setToolTip(
                f"{e['content']}\n\nSource : {e['source'] or '—'}\n"
                f"Crédibilité : {e['credibility']}"
            )
            self.table.setVerticalHeaderItem(r, item)
        self.table.setVerticalHeaderItem(n_ev, QTableWidgetItem("Score d'incohérence"))

        for r, e in enumerate(self._evs):
            for c, h in enumerate(self._hyps):
                code = SCORE_TO_CODE.get(self._scores.get((e["id"], h["id"])))
                self.table.setCellWidget(
                    r, c, self._make_matrix_cell(e["id"], h["id"], code)
                )
            self.table.setItem(r, diag_col, self._readonly_item("", center=True))
            btn = QPushButton("Intégrer")
            btn.setToolTip(
                "Saisir P(E|H) et P(E|¬H) pour intégrer cette preuve au calcul "
                "bayésien"
            )
            btn.clicked.connect(partial(self.open_bayesian_dialog, e["id"]))
            self.table.setCellWidget(r, bayes_col, btn)

        for c in range(n_hyp):
            self.table.setItem(n_ev, c, self._readonly_item("", center=True))
        self.table.setItem(n_ev, diag_col, self._readonly_item("—", center=True))
        self.table.setItem(n_ev, bayes_col, self._readonly_item("—", center=True))

        self._recompute()

    def _make_matrix_cell(
        self, evidence_id: int, hypothesis_id: int, code: str | None
    ) -> QWidget:
        """Construit le widget composite d'une cellule (combobox ACH + BF)."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(1)

        combo = QComboBox()
        combo.addItems(COMBO_ITEMS)
        combo.setToolTip(CODE_TOOLTIP)
        combo.setCurrentText(code if code else "")
        combo.currentIndexChanged.connect(
            partial(self._on_cell_changed, evidence_id, hypothesis_id, combo)
        )

        bf_label = QLabel("")
        bf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bf_label.setStyleSheet(f"font-size: 10px; color: {BF_GREY};")

        v.addWidget(combo)
        v.addWidget(bf_label)

        self._combos[(evidence_id, hypothesis_id)] = combo
        self._bf_labels[(evidence_id, hypothesis_id)] = bf_label
        return container

    @staticmethod
    def _readonly_item(text: str, center: bool = False) -> QTableWidgetItem:
        """Crée une cellule non éditable."""
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if center:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    def _reload_and_recompute(self) -> None:
        """Recharge la matrice depuis la base puis recalcule (structure inchangée)."""
        if self.project_id is None:
            return
        self._cells = self.db.get_cells(self.project_id)
        self._derive()
        self._recompute()

    def _on_cell_changed(
        self, evidence_id: int, hypothesis_id: int, combo: QComboBox, _index: int
    ) -> None:
        """Persiste le score ACH d'une cellule puis recalcule."""
        if self.project_id is None:
            return
        code = combo.currentText()
        score = CODE_TO_SCORE[code] if code else None
        self.db.set_consistency(self.project_id, evidence_id, hypothesis_id, score)
        self._reload_and_recompute()

    def _recompute(self) -> None:
        """Recalcule ACH (incohérence, classement, diagnosticité) et bayésien."""
        if not self._hyps or not self._evs:
            return
        evidence = [(e["id"], e["credibility"]) for e in self._evs]
        n_ev = len(self._evs)
        n_hyp = len(self._hyps)
        diag_col = n_hyp

        inc = compute_incoherence(self._hyp_ids, evidence, self._scores)
        rank = compute_ranking(self._hyp_ids, inc)
        diag = compute_diagnosticity(self._hyp_ids, evidence, self._scores)

        for c, h in enumerate(self._hyps):
            v = inc[h["id"]]
            cell = self.table.item(n_ev, c)
            if cell is not None:
                cell.setText(f"{v:.2f}" if v is not None else "—")

        ranked_diag = sorted(
            ((e["id"], diag[e["id"]]) for e in self._evs if diag[e["id"]] is not None),
            key=lambda x: x[1],
            reverse=True,
        )
        top3 = {ev_id for ev_id, val in ranked_diag[:3] if val > 0}
        for r, e in enumerate(self._evs):
            d = diag[e["id"]]
            cell = self.table.item(r, diag_col)
            if cell is None:
                continue
            cell.setText(f"{d:.2f}" if d is not None else "—")
            cell.setBackground(GREEN if e["id"] in top3 else QBrush())

        labels = {h["id"]: h["label"] for h in self._hyps}
        if rank:
            self.ranking_label.setText(
                "Classement ACH : " + " > ".join(labels[h] for h in rank)
            )
        else:
            self.ranking_label.setText("Classement ACH : (aucune hypothèse évaluée)")

        for (ev_id, hyp_id), lbl in self._bf_labels.items():
            pair = self._likelihoods.get((ev_id, hyp_id))
            if pair is None:
                lbl.setText("")
                continue
            p_h, p_not_h = pair
            bf = p_h / p_not_h if p_not_h else float("inf")
            lbl.setText(f"BF {bf:.2f}")
            if bf > 1.05:
                lbl.setStyleSheet(f"font-size: 10px; color: {BF_GREEN};")
            elif bf < 0.95:
                lbl.setStyleSheet(f"font-size: 10px; color: {BF_RED};")
            else:
                lbl.setStyleSheet(f"font-size: 10px; color: {BF_GREY};")

        bayes = compute_bayesian(self._hyp_ids, self._priors, self._likelihoods)
        self.bayes_table.setRowCount(n_hyp)
        chart_data: list[tuple[str, float, float]] = []
        for i, h in enumerate(self._hyps):
            res = bayes[h["id"]]
            prior = float(res["prior"])
            posterior = float(res["posterior"])
            cum_lr = float(res["cumulative_lr"])
            values = [
                h["label"],
                f"{prior * 100:.1f} %",
                f"×{cum_lr:.2f}",
                f"{posterior * 100:.1f} %",
            ]
            for col, text in enumerate(values):
                cell = QTableWidgetItem(text)
                if col > 0:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.bayes_table.setItem(i, col, cell)
            chart_data.append((h["label"], prior, posterior))
        self.bar_chart.set_data(chart_data)

    # --- Actions -------------------------------------------------------------

    def add_hypothesis(self) -> None:
        """Ouvre le dialogue d'ajout d'hypothèse et reconstruit la matrice."""
        if self.project_id is None:
            return
        dlg = AddHypothesisDialog(f"H{len(self._hyps) + 1}", self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        label, description, prior = dlg.get_data()
        if not label:
            return
        self.db.create_hypothesis(self.project_id, label, description, prior)
        self._rebuild()

    def add_evidence(self) -> None:
        """Ouvre le dialogue d'ajout de preuve et reconstruit la matrice."""
        if self.project_id is None:
            return
        dlg = AddEvidenceDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        content, source, credibility = dlg.get_data()
        if not content:
            return
        self.db.create_evidence(self.project_id, content, source, credibility)
        self._rebuild()

    def open_bayesian_dialog(self, evidence_id: int) -> None:
        """Ouvre la saisie des vraisemblances pour une preuve donnée."""
        if self.project_id is None or not self._hyps:
            return
        index = next(
            (i for i, e in enumerate(self._evs) if e["id"] == evidence_id), None
        )
        if index is None:
            return
        evidence = self._evs[index]
        existing = {
            h["id"]: (
                self._cells.get((evidence_id, h["id"]), {}).get("p_h"),
                self._cells.get((evidence_id, h["id"]), {}).get("p_not_h"),
            )
            for h in self._hyps
        }
        dlg = BayesianIntegrationDialog(
            f"E{index + 1}", evidence["content"], self._hyps, existing, self
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        for hyp_id, (p_h, p_not_h) in dlg.get_data().items():
            self.db.set_likelihoods(self.project_id, evidence_id, hyp_id, p_h, p_not_h)
        self._reload_and_recompute()

    def run_sensitivity(self) -> None:
        """Calcule et affiche l'analyse de sensibilité du classement ACH."""
        if self.project_id is None:
            return
        if len(self._evs) < 2:
            QMessageBox.information(
                self,
                "Analyse de sensibilité",
                "Il faut au moins deux preuves pour analyser la sensibilité.",
            )
            return

        evidence = [(e["id"], e["credibility"]) for e in self._evs]
        base_rank, results = compute_sensitivity(self._hyp_ids, evidence, self._scores)
        labels = {h["id"]: h["label"] for h in self._hyps}
        ev_labels = {e["id"]: f"E{i + 1}" for i, e in enumerate(self._evs)}

        def fmt(rank: list[int]) -> str:
            return " > ".join(labels[h] for h in rank) if rank else "(aucun)"

        lines = [f"Classement de référence : {fmt(base_rank)}", ""]
        if not base_rank:
            lines.append(
                "Aucune hypothèse n'est encore évaluée : remplissez la matrice "
                "avant de lancer l'analyse."
            )
        else:
            flips = [r for r in results if r[1]]
            if not flips:
                lines.append(
                    "Aucune preuve ne fait basculer le classement. "
                    "Le résultat est robuste."
                )
            else:
                lines.append(
                    f"{len(flips)} preuve(s) critique(s) — leur retrait "
                    "modifie le classement :"
                )
                for ev_id, changed, new_rank in results:
                    if changed:
                        lines.append(f"  • {ev_labels[ev_id]} retirée → {fmt(new_rank)}")
                no_effect = [ev_labels[r[0]] for r in results if not r[1]]
                lines.append("")
                lines.append("Preuves sans effet sur le classement :")
                lines.append("  " + (", ".join(no_effect) if no_effect else "(aucune)"))

        SensitivityDialog("\n".join(lines), self).exec()

    # --- Résumé extractif (Phase 6) ------------------------------------------

    def summarize_evidence(self) -> None:
        """Sélectionne une preuve et lance le résumé extractif en arrière-plan."""
        if self.project_id is None:
            return
        evidence = self.db.list_evidence(self.project_id)
        if not evidence:
            QMessageBox.information(
                self, "Résumer une preuve", "Ce projet ne contient aucune preuve."
            )
            return
        picker = EvidencePickerDialog(evidence, self)
        if picker.exec() != QDialog.DialogCode.Accepted:
            return
        target_id = picker.selected_id()
        corpus = [(e["id"], e["content"]) for e in evidence]
        target_index = next(
            (i for i, (eid, _) in enumerate(corpus) if eid == target_id), None
        )
        if target_index is None:
            return
        self._summary_label = f"E{target_index + 1}"
        self.summarize_btn.setEnabled(False)
        self.summarize_btn.setText("📝 Résumé en cours…")
        self._summary_worker = SummaryWorker(target_index, corpus)
        self._summary_worker.finished_summary.connect(self._on_summary_ready)
        self._summary_worker.failed.connect(self._on_summary_failed)
        self._summary_worker.start()

    def _reset_summary_button(self) -> None:
        """Restaure le bouton de résumé."""
        self.summarize_btn.setText("📝 Résumer une preuve")
        self.summarize_btn.setEnabled(self.project_id is not None)

    def _on_summary_ready(self, result: dict) -> None:
        """Affiche le dialogue de résumé une fois le calcul terminé."""
        self._reset_summary_button()
        SummaryDialog(
            getattr(self, "_summary_label", "Preuve"),
            result["sentences"],
            result["selected"],
            result["keywords"],
            self,
        ).exec()

    def _on_summary_failed(self, message: str) -> None:
        """Signale un échec du résumé (spaCy/modèle manquant)."""
        self._reset_summary_button()
        QMessageBox.warning(self, "Résumé impossible", message)


# =============================================================================
#  Vue Prédictions + calibration (onglet « Prédictions »)
# =============================================================================

class PredictionsView(QWidget):
    """Suivi des prédictions et tableau de bord de calibration (Brier, Murphy)."""

    PERIODS = [("30 j", 30), ("90 j", 90), ("1 an", 365)]

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        """Construit la vue (état initial : aucun projet chargé)."""
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None
        self._predictions: list[sqlite3.Row] = []

        self.new_btn = QPushButton("➕ Nouvelle prédiction")
        self.update_btn = QPushButton("✎ Mettre à jour")
        self.resolve_btn = QPushButton("✓ Résoudre")
        self.new_btn.clicked.connect(self.new_prediction)
        self.update_btn.clicked.connect(self.update_selected)
        self.resolve_btn.clicked.connect(self.resolve_selected)
        button_row = QHBoxLayout()
        button_row.addWidget(self.new_btn)
        button_row.addWidget(self.update_btn)
        button_row.addWidget(self.resolve_btn)
        button_row.addStretch(1)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Question", "Probabilité", "Catégorie", "Échéance", "Statut", "Issue"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemDoubleClicked.connect(lambda _item: self.update_selected())
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 6):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)

        # --- Panneau de calibration ---
        self.murphy_label = QLabel("Brier = — | Calibration = — | Résolution = — "
                                   "| Incertitude = —")
        self.murphy_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        self.period_label = QLabel("Brier par période — 30 j : — | 90 j : — | 1 an : —")
        self.period_label.setWordWrap(True)

        self.category_table = QTableWidget(0, 3)
        self.category_table.setHorizontalHeaderLabels(["Catégorie", "Brier", "n"])
        self.category_table.verticalHeader().setVisible(False)
        self.category_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.category_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.category_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.category_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.category_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.category_table.setMinimumWidth(280)

        if MATPLOTLIB_AVAILABLE:
            self.reliability: QWidget = ReliabilityCanvas()
        else:
            self.reliability = QLabel(
                "Diagramme de fiabilité indisponible — installez matplotlib "
                "(pip install matplotlib)."
            )
            self.reliability.setWordWrap(True)
            self.reliability.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.reliability.setStyleSheet("color: #777;")

        calib_split = QSplitter(Qt.Orientation.Horizontal)
        cat_box = QWidget()
        cat_layout = QVBoxLayout(cat_box)
        cat_layout.setContentsMargins(0, 0, 0, 0)
        cat_title = QLabel("Brier par catégorie")
        cat_title.setStyleSheet("font-weight: 600;")
        cat_layout.addWidget(cat_title)
        cat_layout.addWidget(self.category_table)
        calib_split.addWidget(cat_box)
        calib_split.addWidget(self.reliability)
        calib_split.setStretchFactor(0, 2)
        calib_split.setStretchFactor(1, 3)

        calib_box = QWidget()
        calib_layout = QVBoxLayout(calib_box)
        calib_layout.setContentsMargins(0, 0, 0, 0)
        calib_title = QLabel("Calibration")
        calib_title.setStyleSheet("font-size: 14px; font-weight: 600; margin-top: 4px;")
        calib_layout.addWidget(calib_title)
        calib_layout.addWidget(self.murphy_label)
        calib_layout.addWidget(self.period_label)
        calib_layout.addWidget(calib_split, 1)

        self.empty_label = QLabel("Sélectionnez un projet pour gérer ses prédictions.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #777;")

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(calib_box)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.addLayout(button_row)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.splitter, 1)

        self._set_controls_enabled(False)
        self.splitter.hide()

    # --- API publique --------------------------------------------------------

    def set_database(self, db: Database) -> None:
        """Remplace la base de données active et réinitialise la vue."""
        self.db = db
        self.clear()

    def load_project(self, project_id: int) -> None:
        """Charge les prédictions d'un projet."""
        if self.db.get_project(project_id) is None:
            self.clear()
            return
        self.project_id = project_id
        self._set_controls_enabled(True)
        self.empty_label.hide()
        self.splitter.show()
        self._reload()

    def clear(self) -> None:
        """Réinitialise la vue à l'état « aucun projet »."""
        self.project_id = None
        self._predictions = []
        self.table.setRowCount(0)
        self.category_table.setRowCount(0)
        self.murphy_label.setText(
            "Brier = — | Calibration = — | Résolution = — | Incertitude = —"
        )
        self.period_label.setText("Brier par période — 30 j : — | 90 j : — | 1 an : —")
        if MATPLOTLIB_AVAILABLE:
            self.reliability.plot([])  # type: ignore[attr-defined]
        self.splitter.hide()
        self.empty_label.show()
        self._set_controls_enabled(False)

    # --- Internes ------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Active/désactive les boutons d'action."""
        self.new_btn.setEnabled(enabled)
        self.update_btn.setEnabled(enabled)
        self.resolve_btn.setEnabled(enabled)

    def _reload(self) -> None:
        """Recharge la liste des prédictions et le tableau de bord."""
        if self.project_id is None:
            return
        self._predictions = self.db.list_predictions(self.project_id)
        self._populate_table()
        self._recompute_calibration()

    def _populate_table(self) -> None:
        """Remplit la table des prédictions."""
        self.table.setRowCount(len(self._predictions))
        for r, p in enumerate(self._predictions):
            prob = p["probability"]
            resolved = p["outcome"] is not None
            if resolved:
                status = "Résolue"
                issue = "Oui" if p["outcome"] == 1 else "Non"
            else:
                status = "Ouverte"
                issue = "—"

            question_item = QTableWidgetItem(p["question"])
            question_item.setData(Qt.ItemDataRole.UserRole, p["id"])
            if p["resolution_source"]:
                question_item.setToolTip(f"Source : {p['resolution_source']}")
            cells = [
                question_item,
                QTableWidgetItem(f"{prob * 100:.0f} %" if prob is not None else "—"),
                QTableWidgetItem(p["category"] or "—"),
                QTableWidgetItem(p["deadline"] or "—"),
                QTableWidgetItem(status),
                QTableWidgetItem(issue),
            ]
            n_updates = self.db.count_prediction_updates(p["id"])
            if n_updates:
                cells[1].setToolTip(f"{n_updates} mise(s) à jour historisée(s)")
            for c, item in enumerate(cells):
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)

    def _selected_prediction(self) -> sqlite3.Row | None:
        """Renvoie la prédiction sélectionnée, ou None."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self._predictions):
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        pred_id = item.data(Qt.ItemDataRole.UserRole)
        return self.db.get_prediction(pred_id)

    def _resolved_pairs(
        self, predictions: list[sqlite3.Row]
    ) -> list[tuple[float, int]]:
        """Extrait les couples (probabilité, issue) des prédictions résolues."""
        return [
            (float(p["probability"]), int(p["outcome"]))
            for p in predictions
            if p["outcome"] is not None and p["probability"] is not None
        ]

    def _recompute_calibration(self) -> None:
        """Recalcule Brier (global/période/catégorie), Murphy et fiabilité."""
        pairs = self._resolved_pairs(self._predictions)

        if not pairs:
            self.murphy_label.setText(
                "Brier = — | Calibration = — | Résolution = — | Incertitude = —   "
                "(aucune prédiction résolue)"
            )
            self.period_label.setText(
                "Brier par période — 30 j : — | 90 j : — | 1 an : —"
            )
            self.category_table.setRowCount(0)
            if MATPLOTLIB_AVAILABLE:
                self.reliability.plot([])  # type: ignore[attr-defined]
            return

        decomp = murphy_decomposition(pairs)
        assert decomp is not None
        self.murphy_label.setText(
            f"Brier = {decomp['brier']:.3f} | Calibration = {decomp['rel']:.3f} "
            f"| Résolution = {decomp['res']:.3f} | Incertitude = {decomp['unc']:.3f} "
            f"  (n = {len(pairs)})"
        )

        # Par période (selon la date de résolution).
        today = date.today()
        period_parts = []
        for label, days in self.PERIODS:
            subset = [
                p
                for p in self._predictions
                if p["outcome"] is not None
                and (d := parse_iso_date(p["resolved_at"])) is not None
                and 0 <= (today - d).days <= days
            ]
            b = brier_score(self._resolved_pairs(subset))
            period_parts.append(
                f"{label} : {b:.3f} (n={len(subset)})" if b is not None else f"{label} : —"
            )
        self.period_label.setText("Brier par période — " + " | ".join(period_parts))

        # Par catégorie.
        by_category: dict[str, list[sqlite3.Row]] = {}
        for p in self._predictions:
            if p["outcome"] is not None:
                by_category.setdefault(p["category"] or "(sans catégorie)", []).append(p)
        self.category_table.setRowCount(len(by_category))
        for r, (cat, preds) in enumerate(sorted(by_category.items())):
            cat_pairs = self._resolved_pairs(preds)
            b = brier_score(cat_pairs)
            items = [
                QTableWidgetItem(cat),
                QTableWidgetItem(f"{b:.3f}" if b is not None else "—"),
                QTableWidgetItem(str(len(cat_pairs))),
            ]
            items[1].setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            items[2].setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            for c, item in enumerate(items):
                self.category_table.setItem(r, c, item)

        if MATPLOTLIB_AVAILABLE:
            self.reliability.plot(reliability_bins(pairs))  # type: ignore[attr-defined]

    # --- Actions -------------------------------------------------------------

    def new_prediction(self) -> None:
        """Ouvre le dialogue de création d'une prédiction."""
        if self.project_id is None:
            return
        categories = list(
            dict.fromkeys(DEFAULT_CATEGORIES + self.db.distinct_categories(self.project_id))
        )
        dlg = NewPredictionDialog(categories, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        question, probability, deadline, category = dlg.get_data()
        if not question:
            return
        self.db.create_prediction(
            self.project_id, question, probability, deadline, category
        )
        self._reload()

    def update_selected(self) -> None:
        """Ouvre le dialogue de mise à jour de la prédiction sélectionnée."""
        pred = self._selected_prediction()
        if pred is None:
            QMessageBox.information(
                self, "Mettre à jour", "Sélectionnez d'abord une prédiction."
            )
            return
        if pred["outcome"] is not None:
            QMessageBox.information(
                self,
                "Mettre à jour",
                "Cette prédiction est déjà résolue ; sa probabilité est figée.",
            )
            return
        current = float(pred["probability"]) if pred["probability"] is not None else 0.5
        dlg = UpdatePredictionDialog(pred["question"], current, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_p, rationale = dlg.get_data()
        if abs(new_p - current) < 1e-9:
            return  # pas de changement
        self.db.add_prediction_update(pred["id"], current, new_p, rationale)
        self._reload()

    def resolve_selected(self) -> None:
        """Ouvre le dialogue de résolution de la prédiction sélectionnée."""
        pred = self._selected_prediction()
        if pred is None:
            QMessageBox.information(
                self, "Résoudre", "Sélectionnez d'abord une prédiction."
            )
            return
        if pred["outcome"] is not None:
            confirm = QMessageBox.question(
                self,
                "Résoudre",
                "Cette prédiction est déjà résolue. Modifier sa résolution ?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        dlg = ResolvePredictionDialog(pred["question"], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        outcome, source, resolved_at = dlg.get_data()
        self.db.resolve_prediction(pred["id"], outcome, source, resolved_at)
        self._reload()


# =============================================================================
#  Vue Graphe de connaissances (onglet « Graphe »)
# =============================================================================

class GraphView(QWidget):
    """Graphe de connaissances : extraction NLP, rendu, et analyses de graphe."""

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        """Construit la vue (état initial : aucun projet chargé)."""
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None
        self._entities: list[sqlite3.Row] = []
        self._rels: list[sqlite3.Row] = []
        self._communities: dict[int, int] = {}
        self._worker: GraphBuildWorker | None = None

        # Barre d'actions.
        self.build_btn = QPushButton("⚙ Construire / Reconstruire le graphe")
        self.build_btn.clicked.connect(self.build_graph)
        self.community_btn = QPushButton("🎨 Colorer par communauté")
        self.community_btn.setCheckable(True)
        self.community_btn.clicked.connect(self.toggle_communities)
        action_row = QHBoxLayout()
        action_row.addWidget(self.build_btn)
        action_row.addWidget(self.community_btn)
        action_row.addStretch(1)

        # Plus court chemin.
        self.src_combo = QComboBox()
        self.dst_combo = QComboBox()
        self.path_btn = QPushButton("Plus court chemin")
        self.path_btn.clicked.connect(self.find_shortest_path)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("De :"))
        path_row.addWidget(self.src_combo, 1)
        path_row.addWidget(QLabel("à :"))
        path_row.addWidget(self.dst_combo, 1)
        path_row.addWidget(self.path_btn)

        self.info_label = QLabel("Graphe non construit.")
        self.info_label.setWordWrap(True)
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #2f6fb3;")
        self.bridges_label = QLabel("")
        self.bridges_label.setWordWrap(True)
        self.sources_label = QLabel("")
        self.sources_label.setWordWrap(True)

        if MATPLOTLIB_AVAILABLE:
            self.canvas: QWidget = GraphCanvas()
        else:
            self.canvas = QLabel(
                "Rendu du graphe indisponible — installez matplotlib "
                "(pip install matplotlib)."
            )
            self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.canvas.setStyleSheet("color: #777;")
            self.canvas.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(action_row)
        layout.addWidget(self.info_label)
        layout.addWidget(self.bridges_label)
        layout.addWidget(self.sources_label)
        layout.addLayout(path_row)
        layout.addWidget(self.path_label)
        layout.addWidget(self.canvas, 1)

        self._set_controls_enabled(False)

    # --- API publique --------------------------------------------------------

    def set_database(self, db: Database) -> None:
        """Remplace la base de données active et réinitialise la vue."""
        self.db = db
        self.clear()

    def load_project(self, project_id: int) -> None:
        """Charge le graphe existant d'un projet (sans relancer le NLP)."""
        if self.db.get_project(project_id) is None:
            self.clear()
            return
        self.project_id = project_id
        self._set_controls_enabled(True)
        self._reload_from_db()

    def clear(self) -> None:
        """Réinitialise la vue à l'état « aucun projet »."""
        self.project_id = None
        self._entities = []
        self._rels = []
        self._communities = {}
        self.community_btn.setChecked(False)
        self.src_combo.clear()
        self.dst_combo.clear()
        self.info_label.setText("Graphe non construit.")
        self.bridges_label.setText("")
        self.sources_label.setText("")
        self.path_label.setText("")
        if MATPLOTLIB_AVAILABLE:
            self.canvas.set_graph([], [])  # type: ignore[attr-defined]
        self._set_controls_enabled(False)

    # --- Internes ------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Active/désactive les contrôles."""
        self.build_btn.setEnabled(enabled)
        has_entities = enabled and bool(self._entities)
        self.community_btn.setEnabled(has_entities)
        self.path_btn.setEnabled(has_entities)
        self.src_combo.setEnabled(has_entities)
        self.dst_combo.setEnabled(has_entities)

    def _reload_from_db(self) -> None:
        """Recharge entités/relations depuis la base et rafraîchit l'affichage."""
        if self.project_id is None:
            return
        self._entities = self.db.list_entities(self.project_id)
        self._rels = self.db.list_relationships(self.project_id)
        self._communities = {}
        self.community_btn.setChecked(False)

        self.src_combo.clear()
        self.dst_combo.clear()
        for e in self._entities:
            label = f"{e['name']} ({e['type'] or '?'})"
            self.src_combo.addItem(label, e["id"])
            self.dst_combo.addItem(label, e["id"])
        if self.dst_combo.count() > 1:
            self.dst_combo.setCurrentIndex(1)

        n_rel = len(self._rels)
        self.info_label.setText(
            f"{len(self._entities)} entité(s), {n_rel} relation(s)."
            if self._entities
            else "Graphe vide : aucune entité détectée. Ajoutez des preuves "
            "puis reconstruisez."
        )
        self.path_label.setText("")
        self._update_bridges()
        self._update_sources()

        if MATPLOTLIB_AVAILABLE:
            self.canvas.set_graph(self._entities, self._rels, color_mode="type")  # type: ignore[attr-defined]
        self._set_controls_enabled(self.project_id is not None)

    def _update_sources(self) -> None:
        """Calcule et affiche le taux de citation comme source (alerte si dominance)."""
        if self.project_id is None or not self._entities:
            self.sources_label.setText("")
            return
        rows, total_evidence = self.db.source_reliability(self.project_id)
        if not rows:
            self.sources_label.setText("Fiabilité des sources : aucune source détectée.")
            return
        parts = []
        dominant = None
        for r in rows[:5]:
            appears = int(r["appears"]) or 1
            sourced = int(r["sourced"])
            rate = 100.0 * sourced / appears
            tag = ""
            if r["metadata"]:
                try:
                    if json.loads(r["metadata"]).get("anonymous_source"):
                        tag = " ⚠anonyme"
                except ValueError:
                    pass
            parts.append(f"{r['name']} : {rate:.0f}% ({sourced}/{appears}){tag}")
            if total_evidence and sourced / total_evidence > 0.5 and dominant is None:
                dominant = (r["name"], sourced, total_evidence)

        text = "Taux de citation comme source — " + " | ".join(parts)
        if dominant:
            text += (
                f"\n⚠ Source dominante : « {dominant[0]} » est citée dans "
                f"{dominant[1]}/{dominant[2]} preuves (> 50 %)."
            )
            self.sources_label.setStyleSheet("color: #b03030;")
        else:
            self.sources_label.setStyleSheet("")
        self.sources_label.setText(text)

    def _update_bridges(self) -> None:
        """Calcule la centralité d'intermédiarité et affiche le top 5."""
        if not self._entities:
            self.bridges_label.setText("")
            return
        ids = [e["id"] for e in self._entities]
        adj: dict[int, set[int]] = {i: set() for i in ids}
        for r in self._rels:
            s, t = r["source_entity_id"], r["target_entity_id"]
            if s in adj and t in adj:
                adj[s].add(t)
                adj[t].add(s)
        bc = betweenness_centrality(ids, adj)
        names = {e["id"]: e["name"] for e in self._entities}
        top = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top = [(nid, score) for nid, score in top if score > 0]
        if top:
            self.bridges_label.setText(
                "Entités-ponts (intermédiarité) : "
                + ", ".join(f"{names[nid]} ({score:.1f})" for nid, score in top)
            )
        else:
            self.bridges_label.setText("Entités-ponts (intermédiarité) : —")

    # --- Construction du graphe (NLP en arrière-plan) ------------------------

    def build_graph(self) -> None:
        """Lance l'extraction NLP sur toutes les preuves du projet."""
        if self.project_id is None:
            return
        evidence = [
            (e["id"], e["content"]) for e in self.db.list_evidence(self.project_id)
        ]
        if not evidence:
            QMessageBox.information(
                self,
                "Construire le graphe",
                "Ce projet ne contient aucune preuve à analyser.",
            )
            return
        self.build_btn.setEnabled(False)
        self.info_label.setText("Chargement du modèle spaCy et analyse en cours…")
        self._worker = GraphBuildWorker(evidence)
        self._worker.progress.connect(self.info_label.setText)
        self._worker.finished_build.connect(self._on_build_finished)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.start()

    def _on_build_failed(self, message: str) -> None:
        """Réagit à un échec d'extraction (spaCy/modèle manquant)."""
        self.build_btn.setEnabled(True)
        self.info_label.setText("Échec de la construction du graphe.")
        QMessageBox.warning(self, "Construction impossible", message)

    def _on_build_finished(self, results: list[dict]) -> None:
        """Persiste les résultats NLP en base puis rafraîchit l'affichage."""
        if self.project_id is None:
            self.build_btn.setEnabled(True)
            return
        pid = self.project_id
        self.db.clear_graph(pid)
        for item in results:
            ev_id = item["evidence_id"]
            ids: dict[tuple[str, str], int] = {}
            for name, etype in item["entities"]:
                eid = self.db.add_entity(pid, name, etype, ev_id)
                ids[(name, etype)] = eid
                self.db.add_entity_mention(pid, eid, ev_id)
            # Co-occurrence : toutes les paires distinctes de la preuve.
            keys = list(ids.values())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    self.db.add_cooccurrence(pid, keys[i], keys[j], ev_id)
            # Relations dirigées par pattern.
            for subj, subj_t, rel, obj, obj_t in item["relations"]:
                if not subj or not obj or subj_t is None or obj_t is None:
                    continue
                sid = self.db.add_entity(pid, subj, subj_t, ev_id)
                oid = self.db.add_entity(pid, obj, obj_t, ev_id)
                if sid != oid:
                    self.db.add_pattern_relation(pid, sid, oid, ev_id, rel)
                if rel in SOURCE_RELATIONS:  # entité citée comme source
                    self.db.mark_source_mention(pid, sid, ev_id)
            # Métadonnées d'entités (ex. source anonyme).
            for name, etype, meta in item.get("entity_meta", []):
                if not name or etype is None:
                    continue
                eid = self.db.add_entity(pid, name, etype, ev_id)
                self.db.add_entity_mention(pid, eid, ev_id)
                self.db.update_entity_metadata(eid, json.dumps(meta, ensure_ascii=False))

        self.build_btn.setEnabled(True)
        self._reload_from_db()

    # --- Analyses de graphe --------------------------------------------------

    def toggle_communities(self) -> None:
        """Bascule la coloration des nœuds entre type et communauté (Louvain)."""
        if not MATPLOTLIB_AVAILABLE or not self._entities:
            return
        if self.community_btn.isChecked():
            ids = [e["id"] for e in self._entities]
            edges = [
                (r["source_entity_id"], r["target_entity_id"], r["weight"] or 1.0)
                for r in self._rels
            ]
            self._communities = louvain_communities(ids, edges)
            n_comm = len(set(self._communities.values()))
            self.community_btn.setText(f"🎨 Communautés ({n_comm})")
            self.canvas.set_color_mode("community", self._communities)  # type: ignore[attr-defined]
        else:
            self.community_btn.setText("🎨 Colorer par communauté")
            self.canvas.set_color_mode("type")  # type: ignore[attr-defined]

    def find_shortest_path(self) -> None:
        """Calcule et affiche le plus court chemin entre deux entités (CTE SQL)."""
        if self.project_id is None or self.src_combo.count() == 0:
            return
        src_id = self.src_combo.currentData()
        dst_id = self.dst_combo.currentData()
        if src_id is None or dst_id is None:
            return
        path = self.db.shortest_path(self.project_id, int(src_id), int(dst_id))
        names = {e["id"]: e["name"] for e in self._entities}
        if not path:
            self.path_label.setText("Aucun chemin entre ces deux entités.")
            return
        self.path_label.setText(
            "Chemin ("
            + str(len(path) - 1)
            + " saut(s)) : "
            + " → ".join(names.get(nid, str(nid)) for nid in path)
        )


# =============================================================================
#  Client LLM FantasyAI Cloud (Phase 8) — asynchrone via QNetworkAccessManager
# =============================================================================

def llm_error_message(status: int, raw: str) -> str:
    """Traduit un code HTTP / une erreur réseau en message lisible (B.6)."""
    if status == 0:
        return raw or "Connexion impossible."
    if status == 401:
        return ("Clé API invalide ou manquante (401). Vérifiez-la dans "
                "Configuration › API LLM.")
    if status == 402:
        return ("Quota ou crédit insuffisant (402). Vérifiez votre compte "
                "FantasyAI Cloud.")
    if status == 429:
        return "Trop de requêtes (429). Patientez quelques instants puis réessayez."
    if 500 <= status < 600:
        return f"Erreur côté serveur ({status}). Réessayez plus tard."
    return f"Erreur de l'API ({status})."


class LlmApi(QObject):
    """Client OpenAI-compatible (FantasyAI Cloud) ; requêtes asynchrones non bloquantes.

    Les callbacks reçoivent (status: int, raw: str). status == 0 signale un échec
    réseau (raw contient alors le message). Sinon status est le code HTTP.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialise le gestionnaire d'accès réseau."""
        super().__init__(parent)
        self._nam = QNetworkAccessManager(self)

    def _request(self, url: str, api_key: str) -> QNetworkRequest:
        """Construit une requête avec en-têtes JSON + Bearer."""
        req = QNetworkRequest(QUrl(url))
        req.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        req.setRawHeader(b"Authorization", b"Bearer " + (api_key or "").encode("utf-8"))
        return req

    def get_models(self, base_url: str, api_key: str, on_done) -> None:  # noqa: ANN001
        """GET /models (liste des modèles disponibles)."""
        url = base_url.rstrip("/") + "/models"
        reply = self._nam.get(self._request(url, api_key))
        reply.finished.connect(lambda: self._finish(reply, on_done))

    def post_chat(  # noqa: ANN001
        self, base_url: str, api_key: str, payload: dict, on_done, timeout_ms: int = 60000
    ) -> None:
        """POST /chat/completions avec délai d'expiration (abort au-delà)."""
        url = base_url.rstrip("/") + "/chat/completions"
        body = QByteArray(json.dumps(payload).encode("utf-8"))
        reply = self._nam.post(self._request(url, api_key), body)
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(reply.abort)
        timer.start(timeout_ms)
        reply.finished.connect(lambda: (timer.stop(), self._finish(reply, on_done)))

    @staticmethod
    def _finish(reply, on_done) -> None:  # noqa: ANN001
        """Lit la réponse, libère le reply et appelle le callback."""
        status_attr = reply.attribute(
            QNetworkRequest.Attribute.HttpStatusCodeAttribute
        )
        status = int(status_attr) if status_attr is not None else 0
        err = reply.error()
        err_str = reply.errorString()
        raw = bytes(reply.readAll().data()).decode("utf-8", "replace")
        reply.deleteLater()
        if status == 0 and err != QNetworkReply.NetworkError.NoError:
            on_done(0, f"Connexion impossible : {err_str}")
        else:
            on_done(status, raw)


class ApiConfigDialog(QDialog):
    """Configuration de l'accès LLM : clé API, modèle, test de connexion (B.1)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Construit le formulaire et précharge les valeurs enregistrées."""
        super().__init__(parent)
        self.setWindowTitle("Configuration — API LLM")
        self.setMinimumWidth(520)
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.api = LlmApi(self)
        self._model_meta: dict[str, bool] = {}  # model_id -> supports_json_object

        base_url = self.settings.value("api_base_url", DEFAULT_LLM_BASE_URL, type=str)

        self.key_edit = QLineEdit(self.settings.value("api_key", "", type=str))
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("Collez votre clé FantasyAI Cloud")

        self.base_edit = QLineEdit(base_url or DEFAULT_LLM_BASE_URL)
        self.base_edit.setReadOnly(True)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_LLM_MODELS)
        saved_model = self.settings.value("api_model", "", type=str)
        if saved_model:
            if self.model_combo.findText(saved_model) < 0:
                self.model_combo.insertItem(0, saved_model)
            self.model_combo.setCurrentText(saved_model)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        btn_test = QPushButton("Tester la connexion")
        btn_test.clicked.connect(self.test_connection)
        btn_refresh = QPushButton("Rafraîchir les modèles")
        btn_refresh.clicked.connect(self.fetch_models)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Enregistrer")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Clé API :", self.key_edit)
        form.addRow("URL de base :", self.base_edit)
        form.addRow("Modèle :", self.model_combo)

        actions = QHBoxLayout()
        actions.addWidget(btn_test)
        actions.addWidget(btn_refresh)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(self.status_label)
        layout.addWidget(buttons)

        if self.key_edit.text().strip():
            self.fetch_models()

    def fetch_models(self) -> None:
        """Peuple la liste des modèles via GET /models (repli sur la liste par défaut)."""
        key = self.key_edit.text().strip()
        if not key:
            self.status_label.setText(
                "Saisissez une clé API pour récupérer la liste des modèles."
            )
            return
        self.status_label.setText("Récupération des modèles…")
        self.api.get_models(
            self.base_edit.text().strip(), key, self._on_models
        )

    def _on_models(self, status: int, raw: str) -> None:
        """Traite la réponse de GET /models."""
        if status != 200:
            self.status_label.setText(
                "Modèles indisponibles (" + llm_error_message(status, raw)
                + "). Liste par défaut conservée."
            )
            return
        try:
            data = json.loads(raw)
            models = data.get("data", data) if isinstance(data, dict) else data
        except ValueError:
            self.status_label.setText("Réponse /models illisible. Liste par défaut.")
            return
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self._model_meta.clear()
        ids: list[str] = []
        for m in models or []:
            if not isinstance(m, dict):
                continue
            if m.get("supports_chat", True) is False:
                continue
            mid = m.get("id")
            if not mid:
                continue
            ids.append(mid)
            self._model_meta[mid] = bool(m.get("supports_json_object", True))
        if not ids:
            ids = list(DEFAULT_LLM_MODELS)
        self.model_combo.addItems(ids)
        if current and self.model_combo.findText(current) >= 0:
            self.model_combo.setCurrentText(current)
        self.status_label.setText(f"{len(ids)} modèle(s) disponible(s).")

    def test_connection(self) -> None:
        """Teste la connexion en interrogeant GET /models."""
        key = self.key_edit.text().strip()
        if not key:
            self.status_label.setText("Aucune clé API saisie.")
            return
        self.status_label.setText("Test en cours…")
        self.api.get_models(
            self.base_edit.text().strip(),
            key,
            lambda status, raw: self.status_label.setText(
                "✅ Connexion réussie." if status == 200
                else "❌ " + llm_error_message(status, raw)
            ),
        )

    def save(self) -> None:
        """Enregistre la configuration dans QSettings (la clé n'est jamais en base)."""
        model = self.model_combo.currentText().strip()
        self.settings.setValue("api_key", self.key_edit.text().strip())
        self.settings.setValue("api_base_url", self.base_edit.text().strip())
        self.settings.setValue("api_model", model)
        supports_json = self._model_meta.get(model, True)
        self.settings.setValue("api_model_json", supports_json)
        self.accept()


# =============================================================================
#  Dialogue « Importer et analyser » (Phase 8 B.2 + B.4)
# =============================================================================

REVIEW_RELATION_TYPES = [
    "AFFIRMS", "CONTRADICTS", "SOURCE", "AFFILIATED_WITH",
    "CREATED", "CO_OCCURS", "DOCUMENTED_BY",
]
ACH_CODE_CHOICES = ["—", "CC", "C", "N", "I", "II"]
HORIZON_CHOICES = [
    ("(non spécifié)", None),
    ("Court terme", "court_terme"),
    ("Moyen terme", "moyen_terme"),
    ("Long terme", "long_terme"),
]
ANALYSIS_MODES = [
    ("Analyse complète", "complete"),
    ("Complétion (cellules vides)", "completion"),
    ("Exploration (dialogue libre)", "exploration"),
]


class ImportAnalyzeDialog(QDialog):
    """Analyse un texte via le LLM, puis révision humaine avant insertion atomique."""

    def __init__(
        self,
        db: Database,
        project_id: int,
        on_applied=None,  # noqa: ANN001
        parent: QWidget | None = None,
    ) -> None:
        """Construit les trois onglets (Texte source, Révision, Exploration)."""
        super().__init__(parent)
        self.db = db
        self.project_id = project_id
        self.on_applied = on_applied
        self.api = LlmApi(self)
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self._raw_response = ""
        self._analysis: dict = {}

        # Lignes de révision (chaque entrée garde ses widgets éditables).
        self.row_evidence: list[dict] = []
        self.row_hyp: list[dict] = []
        self.row_ent: list[dict] = []
        self.row_rel: list[dict] = []
        self.row_pred: list[dict] = []
        self.row_bn: list[dict] = []

        self.setWindowTitle("Importer et analyser un texte")
        self.setMinimumSize(1000, 720)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_source_tab(), "Texte source")
        self.tabs.addTab(self._build_review_tab(), "Révision")
        self.tabs.addTab(self._build_exploration_tab(), "Exploration")
        self.tabs.setTabEnabled(1, False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    # --- Onglet 1 : texte source --------------------------------------------

    def _build_source_tab(self) -> QWidget:
        """Onglet de saisie du texte + mode + bouton Analyser."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Collez le texte brut à analyser :"))
        self.source_text = QPlainTextEdit()
        self.source_text.setMinimumHeight(320)
        self.source_text.textChanged.connect(self._update_token_label)
        layout.addWidget(self.source_text)

        self.token_label = QLabel("≈ 0 token")
        layout.addWidget(self.token_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode :"))
        self.mode_combo = QComboBox()
        for label, value in ANALYSIS_MODES:
            self.mode_combo.addItem(label, value)
        controls.addWidget(self.mode_combo)
        controls.addStretch(1)
        self.analyze_btn = QPushButton("▶ Analyser")
        self.analyze_btn.clicked.connect(self.on_analyze)
        controls.addWidget(self.analyze_btn)
        layout.addLayout(controls)

        self.analyze_status = QLabel("")
        self.analyze_status.setWordWrap(True)
        layout.addWidget(self.analyze_status)
        return widget

    def _update_token_label(self) -> None:
        """Met à jour l'estimation de tokens et signale les textes longs."""
        n = estimate_tokens(self.source_text.toPlainText())
        if n > 4000:
            self.token_label.setText(
                f"≈ {n} tokens  —  ⚠ texte long, l'analyse peut être tronquée "
                "ou coûteuse."
            )
            self.token_label.setStyleSheet("color: #b8860b;")
        else:
            self.token_label.setText(f"≈ {n} tokens")
            self.token_label.setStyleSheet("")

    # --- Onglet 2 : révision -------------------------------------------------

    def _build_review_tab(self) -> QWidget:
        """Onglet de révision : propositions IA (gauche) + projet actuel (droite)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Colonne gauche : propositions (zone défilante).
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Proposé par l'IA</b> (décochez ou éditez)"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.proposals_host = QWidget()
        self.proposals_layout = QVBoxLayout(self.proposals_host)
        self.proposals_layout.addStretch(1)
        scroll.setWidget(self.proposals_host)
        left_layout.addWidget(scroll)
        splitter.addWidget(left)

        # Colonne droite : aperçu du projet (lecture seule).
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Déjà dans le projet</b>"))
        self.project_preview = QTextBrowser()
        right_layout.addWidget(self.project_preview)
        splitter.addWidget(right)
        splitter.setSizes([620, 360])
        layout.addWidget(splitter)

        # Boutons d'application.
        actions = QHBoxLayout()
        actions.addStretch(1)
        btn_cancel = QPushButton("❌ Annuler")
        btn_cancel.clicked.connect(self.reject)
        btn_continue = QPushButton("📝 Appliquer et continuer")
        btn_continue.clicked.connect(lambda: self._apply(keep_open=True))
        btn_apply = QPushButton("✅ Appliquer la sélection")
        btn_apply.clicked.connect(lambda: self._apply(keep_open=False))
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_continue)
        actions.addWidget(btn_apply)
        layout.addLayout(actions)
        return widget

    # --- Onglet 3 : exploration ---------------------------------------------

    def _build_exploration_tab(self) -> QWidget:
        """Onglet d'exploration : question libre + réponse markdown (sans JSON)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Question d'exploration (dialogue libre avec le LLM) :"))
        self.explore_question = QPlainTextEdit()
        self.explore_question.setMaximumHeight(120)
        layout.addWidget(self.explore_question)
        row = QHBoxLayout()
        row.addStretch(1)
        self.explore_btn = QPushButton("▶ Explorer")
        self.explore_btn.clicked.connect(self.on_explore)
        row.addWidget(self.explore_btn)
        layout.addLayout(row)
        self.explore_result = QTextBrowser()
        layout.addWidget(self.explore_result)
        return widget

    # --- Construction de la requête -----------------------------------------

    def _config(self) -> tuple[str, str, str, bool]:
        """Renvoie (api_key, base_url, model, supports_json) depuis QSettings."""
        return (
            self.settings.value("api_key", "", type=str),
            self.settings.value("api_base_url", DEFAULT_LLM_BASE_URL, type=str),
            self.settings.value("api_model", DEFAULT_LLM_MODELS[0], type=str),
            self.settings.value("api_model_json", True, type=bool),
        )

    def on_analyze(self) -> None:
        """Lance l'analyse structurée (ou bascule vers l'exploration)."""
        mode = self.mode_combo.currentData()
        text = self.source_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Analyse", "Le texte source est vide.")
            return
        if mode == "exploration":
            self.tabs.setCurrentIndex(2)
            self.explore_question.setPlainText(text)
            self.on_explore()
            return

        api_key, base_url, model, supports_json = self._config()
        if not api_key:
            QMessageBox.warning(self, "Analyse", "Aucune clé API configurée.")
            return

        user_prompt = build_llm_user_prompt(
            mode, text,
            self.db.list_hypotheses(self.project_id),
            self.db.list_evidence(self.project_id),
            self.db.list_predictions(self.project_id),
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        if supports_json:
            payload["response_format"] = {"type": "json_object"}

        self.analyze_btn.setEnabled(False)
        self.analyze_status.setText("Analyse en cours… (jusqu'à 60 s)")
        self.api.post_chat(base_url, api_key, payload, self._on_analysis_response)

    def _on_analysis_response(self, status: int, raw: str) -> None:
        """Traite la réponse structurée du LLM."""
        self.analyze_btn.setEnabled(True)
        self._raw_response = raw
        if status != 200:
            self.analyze_status.setText("Échec : " + llm_error_message(status, raw))
            QMessageBox.warning(
                self, "Analyse", llm_error_message(status, raw)
            )
            return
        content = self._extract_message_content(raw)
        analysis = extract_json_block(content) if content else None
        if analysis is None:
            self.analyze_status.setText("Réponse non exploitable (JSON introuvable).")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Analyse")
            box.setText("Impossible d'extraire un JSON valide de la réponse.")
            box.setDetailedText((content or raw)[:4000])
            box.exec()
            return
        self._analysis = analysis
        self.populate_review(analysis)
        self.analyze_status.setText("Analyse reçue — passez à l'onglet Révision.")
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)

    @staticmethod
    def _extract_message_content(raw: str) -> str | None:
        """Extrait le texte de réponse, que ce soit en JSON unique ou en flux SSE.

        Gère deux formats : la réponse non-streamée
        ``{"choices":[{"message":{"content": "..."}}]}`` et le flux Server-Sent
        Events ``data: {"choices":[{"delta":{"content":"..."}}]}`` (réassemblé).
        """
        # Cas 1 — réponse non-streamée (un seul objet JSON).
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("choices"):
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                if msg.get("content") is not None:
                    return msg["content"]
                delta = choice.get("delta") or {}
                if delta.get("content") is not None:
                    return delta["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            pass

        # Cas 2 — flux SSE : réassembler les fragments delta.content.
        if "data:" in raw:
            pieces: list[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                frag = (choice.get("delta") or {}).get("content")
                if frag is None:
                    frag = (choice.get("message") or {}).get("content")
                if frag:
                    pieces.append(frag)
            if pieces:
                return "".join(pieces)

        return raw  # repli : l'extraction JSON tentera sa chance sur le brut

    def on_explore(self) -> None:
        """Mode exploration : question libre, réponse en markdown (pas de JSON)."""
        question = self.explore_question.toPlainText().strip()
        if not question:
            QMessageBox.information(self, "Exploration", "Saisissez une question.")
            return
        api_key, base_url, model, _ = self._config()
        if not api_key:
            QMessageBox.warning(self, "Exploration", "Aucune clé API configurée.")
            return
        hyps = self.db.list_hypotheses(self.project_id)
        context = "; ".join(
            f"{h['label']} : {h['description'] or ''}".strip() for h in hyps
        ) or "(aucune hypothèse)"
        prompt = (
            "Tu es un analyste qui aide à explorer des hypothèses concurrentes. "
            "Réponds en français, en markdown, de façon concise et structurée. "
            "N'invente pas de faits.\n\n"
            f"Hypothèses du projet : {context}\n\nQuestion : {question}"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "stream": False,
        }
        self.explore_btn.setEnabled(False)
        self.explore_result.setPlainText("Exploration en cours…")
        self.api.post_chat(base_url, api_key, payload, self._on_exploration_response)

    def _on_exploration_response(self, status: int, raw: str) -> None:
        """Affiche la réponse d'exploration en markdown."""
        self.explore_btn.setEnabled(True)
        if status != 200:
            self.explore_result.setPlainText(
                "Échec : " + llm_error_message(status, raw)
            )
            return
        content = self._extract_message_content(raw) or "(réponse vide)"
        try:
            self.explore_result.setMarkdown(content)
        except Exception:  # noqa: BLE001 - repli texte brut
            self.explore_result.setPlainText(content)

    # --- Aperçu du projet (colonne droite) -----------------------------------

    def _refresh_project_preview(self) -> None:
        """Construit l'aperçu (lecture seule) de l'état actuel du projet."""
        hyps = self.db.list_hypotheses(self.project_id)
        evs = self.db.list_evidence(self.project_id)
        preds = self.db.list_predictions(self.project_id)
        ents = self.db.list_entities(self.project_id)
        html = ["<h4>Hypothèses</h4><ul>"]
        html += [f"<li>{escape(h['label'])}</li>" for h in hyps] or ["<li>—</li>"]
        html.append("</ul><h4>Preuves</h4><ul>")
        html += [
            f"<li>{escape((e['content'] or '')[:80])}</li>" for e in evs[:12]
        ] or ["<li>—</li>"]
        html.append("</ul><h4>Prédictions</h4><ul>")
        html += [f"<li>{escape(p['question'])}</li>" for p in preds[:12]] or ["<li>—</li>"]
        html.append(f"</ul><h4>Entités ({len(ents)})</h4>")
        self.project_preview.setHtml("".join(html))

    # --- Population de la révision -------------------------------------------

    def _clear_proposals(self) -> None:
        """Vide la zone de propositions et les listes de lignes."""
        for lst in (self.row_evidence, self.row_hyp, self.row_ent,
                    self.row_rel, self.row_pred, self.row_bn):
            lst.clear()
        while self.proposals_layout.count():
            item = self.proposals_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.proposals_layout.addStretch(1)

    def _section(self, title: str, count: int) -> QVBoxLayout:
        """Crée une section repliable et renvoie le layout d'accueil des cartes."""
        box = QGroupBox(f"{title} ({count})")
        box.setCheckable(True)
        box.setChecked(True)
        inner = QVBoxLayout(box)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(content)
        box.toggled.connect(content.setVisible)
        self.proposals_layout.insertWidget(
            self.proposals_layout.count() - 1, box
        )
        return content_layout

    @staticmethod
    def _card() -> tuple[QFrame, QVBoxLayout]:
        """Crée un cadre de carte pour un élément proposé."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 6, 6, 6)
        return frame, lay

    def populate_review(self, analysis: dict) -> None:
        """Remplit l'onglet de révision à partir du JSON d'analyse."""
        self._clear_proposals()
        self._refresh_project_preview()

        existing_hyps = self.db.list_hypotheses(self.project_id)
        existing_labels = [h["label"] for h in existing_hyps]
        new_hyps = analysis.get("new_hypotheses") or []
        new_labels = [h.get("label", "") for h in new_hyps if h.get("label")]
        all_labels = existing_labels + new_labels

        existing_ents = self.db.list_entities(self.project_id)
        existing_ent_names = [e["name"] for e in existing_ents]
        prop_ent_names = [e.get("name", "") for e in (analysis.get("entities") or [])]
        all_entity_names = existing_ent_names + [
            n for n in prop_ent_names if n and n not in existing_ent_names
        ]

        def to_label(key: str) -> str | None:
            if key in all_labels:
                return key
            m = re.fullmatch(r"H(\d+)", str(key))
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(all_labels):
                    return all_labels[idx]
            return None

        self._build_hyp_section(new_hyps, all_labels)
        self._build_evidence_section(analysis.get("evidence_items") or [],
                                     all_labels, to_label)
        self._build_entity_section(analysis.get("entities") or [])
        self._build_relation_section(analysis.get("relations") or [],
                                     all_entity_names)
        self._build_prediction_section(analysis.get("predictions") or [])
        self._build_bifurcation_section(analysis.get("bifurcation_nodes") or [],
                                        all_labels)
        self._build_narrative_section(analysis.get("narrative_synthesis") or {})

    def _horizon_combo(self, selected: str | None) -> QComboBox:
        """Combo d'horizon (valeurs court/moyen/long)."""
        combo = QComboBox()
        for label, value in HORIZON_CHOICES:
            combo.addItem(label, value)
        for i in range(combo.count()):
            if combo.itemData(i) == selected:
                combo.setCurrentIndex(i)
                break
        return combo

    def _label_combo(self, labels: list[str], selected: str | None,
                     with_none: bool = True) -> QComboBox:
        """Combo de labels d'hypothèses (avec option vide)."""
        combo = QComboBox()
        if with_none:
            combo.addItem("(aucun)", None)
        for lab in labels:
            combo.addItem(lab, lab)
        if selected:
            idx = combo.findData(selected)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        return combo

    def _build_hyp_section(self, new_hyps: list[dict], all_labels: list[str]) -> None:
        """Section des hypothèses candidates."""
        layout = self._section("Hypothèses candidates", len(new_hyps))
        for h in new_hyps:
            frame, lay = self._card()
            check = QCheckBox(h.get("label", "(sans label)"))
            check.setChecked(True)
            lay.addWidget(check)
            label_edit = QLineEdit(h.get("label", ""))
            desc_edit = QLineEdit(h.get("description", ""))
            prior = QDoubleSpinBox()
            prior.setRange(0.0, 1.0)
            prior.setSingleStep(0.05)
            prior.setValue(float(h.get("prior_probability", 0.5) or 0.5))
            horizon = self._horizon_combo(h.get("horizon"))
            parent = self._label_combo(
                [lbl for lbl in all_labels if lbl != h.get("label")],
                h.get("parent_hypothesis_label"),
            )
            form = QFormLayout()
            form.addRow("Label :", label_edit)
            form.addRow("Description :", desc_edit)
            form.addRow("A priori :", prior)
            form.addRow("Horizon :", horizon)
            form.addRow("Parent :", parent)
            lay.addLayout(form)
            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_hyp.append({
                "frame": frame, "check": check, "label": label_edit,
                "desc": desc_edit, "prior": prior, "horizon": horizon,
                "parent": parent, "removed": False,
            })

    def _build_evidence_section(self, items: list[dict], all_labels: list[str],
                                to_label) -> None:  # noqa: ANN001
        """Section des preuves (avec marqueur de doublon et scores ACH)."""
        layout = self._section("Preuves", len(items))
        for ev in items:
            frame, lay = self._card()
            content = ev.get("content", "")
            check = QCheckBox("Preuve")
            check.setChecked(True)
            header = QHBoxLayout()
            header.addWidget(check)
            header.addStretch(1)
            dup = self.db.find_similar_evidences(self.project_id, content)
            marker = QLabel("🔴 doublon probable" if dup else "🟢 nouveau")
            if dup:
                marker.setToolTip(
                    "Proche de : " + "; ".join(
                        f"{c[:50]} ({s:.0%})" for _, c, s in dup[:3]
                    )
                )
            header.addWidget(marker)
            lay.addLayout(header)

            content_edit = QTextEdit(content)
            content_edit.setMaximumHeight(70)
            source_edit = QLineEdit(ev.get("source", ""))
            cred = QSpinBox()
            cred.setRange(1, 5)
            cred.setValue(int(ev.get("credibility", 3) or 3))
            form = QFormLayout()
            form.addRow("Contenu :", content_edit)
            form.addRow("Source :", source_edit)
            form.addRow("Crédibilité :", cred)
            lay.addLayout(form)

            # Scores ACH par hypothèse (remappés vers les labels).
            raw_ach = ev.get("ach_scores") or {}
            ach_by_label: dict[str, str] = {}
            for k, v in raw_ach.items():
                lbl = to_label(k)
                if lbl:
                    ach_by_label[lbl] = v
            raw_bayes = ev.get("bayesian_likelihoods") or {}
            bayes_by_label: dict[str, dict] = {}
            for k, v in raw_bayes.items():
                lbl = to_label(k)
                if lbl and isinstance(v, dict):
                    bayes_by_label[lbl] = v

            ach_combos: dict[str, QComboBox] = {}
            if all_labels:
                ach_form = QFormLayout()
                for lbl in all_labels:
                    combo = QComboBox()
                    combo.addItems(ACH_CODE_CHOICES)
                    code = ach_by_label.get(lbl)
                    if code in ACH_CODE_CHOICES:
                        combo.setCurrentText(code)
                    ach_form.addRow(f"ACH · {lbl} :", combo)
                    ach_combos[lbl] = combo
                lay.addLayout(ach_form)

            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_evidence.append({
                "frame": frame, "check": check, "content": content_edit,
                "source": source_edit, "cred": cred, "ach": ach_combos,
                "bayes": bayes_by_label, "removed": False,
            })

    def _build_entity_section(self, items: list[dict]) -> None:
        """Section des entités (avec marqueur de fusion)."""
        layout = self._section("Entités", len(items))
        for ent in items:
            frame, lay = self._card()
            name = ent.get("name", "")
            check = QCheckBox(name or "(entité)")
            check.setChecked(True)
            existing = self.db._get_existing_entity_id(self.project_id, name)
            header = QHBoxLayout()
            header.addWidget(check)
            header.addStretch(1)
            header.addWidget(QLabel("🟠 fusion" if existing else "🟢 nouvelle"))
            lay.addLayout(header)
            name_edit = QLineEdit(name)
            type_combo = QComboBox()
            type_combo.addItems(ENTITY_TYPES)
            if ent.get("type") in ENTITY_TYPES:
                type_combo.setCurrentText(ent["type"])
            form = QFormLayout()
            form.addRow("Nom :", name_edit)
            form.addRow("Type :", type_combo)
            lay.addLayout(form)
            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_ent.append({
                "frame": frame, "check": check, "name": name_edit,
                "type": type_combo, "removed": False,
            })

    def _build_relation_section(self, items: list[dict],
                                entity_names: list[str]) -> None:
        """Section des relations (source/cible parmi les entités)."""
        layout = self._section("Relations", len(items))
        for rel in items:
            frame, lay = self._card()
            check = QCheckBox("Relation")
            check.setChecked(True)
            lay.addWidget(check)
            src = QComboBox()
            src.setEditable(True)
            src.addItems(entity_names)
            src.setCurrentText(rel.get("source", ""))
            tgt = QComboBox()
            tgt.setEditable(True)
            tgt.addItems(entity_names)
            tgt.setCurrentText(rel.get("target", ""))
            rtype = QComboBox()
            rtype.addItems(REVIEW_RELATION_TYPES)
            if rel.get("type") in REVIEW_RELATION_TYPES:
                rtype.setCurrentText(rel["type"])
            form = QFormLayout()
            form.addRow("Source :", src)
            form.addRow("Cible :", tgt)
            form.addRow("Type :", rtype)
            lay.addLayout(form)
            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_rel.append({
                "frame": frame, "check": check, "source": src,
                "target": tgt, "type": rtype, "removed": False,
            })

    def _build_prediction_section(self, items: list[dict]) -> None:
        """Section des prédictions."""
        layout = self._section("Prédictions", len(items))
        for pred in items:
            frame, lay = self._card()
            check = QCheckBox("Prédiction")
            check.setChecked(True)
            lay.addWidget(check)
            question = QTextEdit(pred.get("question", ""))
            question.setMaximumHeight(60)
            prob = QDoubleSpinBox()
            prob.setRange(0.0, 1.0)
            prob.setSingleStep(0.05)
            prob.setValue(float(pred.get("probability", 0.5) or 0.5))
            deadline = QDateEdit()
            deadline.setCalendarPopup(True)
            deadline.setDisplayFormat("yyyy-MM-dd")
            parsed = QDate.fromString(str(pred.get("deadline", "")), "yyyy-MM-dd")
            deadline.setDate(parsed if parsed.isValid() else QDate.currentDate())
            category = QLineEdit(pred.get("category", ""))
            form = QFormLayout()
            form.addRow("Question :", question)
            form.addRow("Probabilité :", prob)
            form.addRow("Échéance :", deadline)
            form.addRow("Catégorie :", category)
            lay.addLayout(form)
            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_pred.append({
                "frame": frame, "check": check, "question": question,
                "prob": prob, "deadline": deadline, "category": category,
                "removed": False,
            })

    def _build_bifurcation_section(self, items: list[dict],
                                   all_labels: list[str]) -> None:
        """Section des nœuds de bifurcation (avec filles cochables)."""
        layout = self._section("Nœuds de bifurcation", len(items))
        for bn in items:
            frame, lay = self._card()
            check = QCheckBox(bn.get("label", "(nœud)"))
            check.setChecked(True)
            lay.addWidget(check)
            label_edit = QLineEdit(bn.get("label", ""))
            cond_edit = QTextEdit(bn.get("condition_text", ""))
            cond_edit.setMaximumHeight(60)
            horizon = self._horizon_combo(bn.get("horizon"))
            parent = self._label_combo(all_labels, bn.get("parent_hypothesis"))
            form = QFormLayout()
            form.addRow("Label :", label_edit)
            form.addRow("Condition :", cond_edit)
            form.addRow("Horizon :", horizon)
            form.addRow("Hypothèse parente :", parent)
            lay.addLayout(form)
            leads = bn.get("leads_to_hypotheses") or []
            lead_checks: dict[str, QCheckBox] = {}
            if all_labels:
                lay.addWidget(QLabel("Mène aux hypothèses :"))
                for lbl in all_labels:
                    cb = QCheckBox(lbl)
                    if lbl in leads:
                        cb.setChecked(True)
                    lay.addWidget(cb)
                    lead_checks[lbl] = cb
            self._add_delete(lay, frame)
            layout.addWidget(frame)
            self.row_bn.append({
                "frame": frame, "check": check, "label": label_edit,
                "condition": cond_edit, "horizon": horizon, "parent": parent,
                "leads": lead_checks, "removed": False,
            })

    def _build_narrative_section(self, synthesis: dict) -> None:
        """Section de synthèse narrative (lecture seule)."""
        if not synthesis:
            return
        layout = self._section("Synthèse narrative", 1)
        frame, lay = self._card()
        browser = QTextBrowser()
        parts = []
        if synthesis.get("resume_global"):
            parts.append(f"**Résumé global**\n\n{synthesis['resume_global']}")
        weak = synthesis.get("signaux_faibles") or []
        if weak:
            parts.append("**Signaux faibles**\n\n" + "\n".join(f"- {s}" for s in weak))
        if synthesis.get("angles_morts"):
            parts.append(f"**Angles morts**\n\n{synthesis['angles_morts']}")
        try:
            browser.setMarkdown("\n\n".join(parts))
        except Exception:  # noqa: BLE001
            browser.setPlainText("\n\n".join(parts))
        lay.addWidget(browser)
        layout.addWidget(frame)

    def _add_delete(self, layout: QVBoxLayout, frame: QFrame) -> None:
        """Ajoute un bouton 🗑 qui masque et marque la carte comme supprimée."""
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("🗑 Retirer")
        btn.clicked.connect(lambda: self._remove_card(frame))
        row.addWidget(btn)
        layout.addLayout(row)

    def _remove_card(self, frame: QFrame) -> None:
        """Marque la carte comme retirée (exclue de la collecte) et la masque."""
        for lst in (self.row_evidence, self.row_hyp, self.row_ent,
                    self.row_rel, self.row_pred, self.row_bn):
            for row in lst:
                if row["frame"] is frame:
                    row["removed"] = True
        frame.setVisible(False)

    # --- Collecte & application ----------------------------------------------

    def collect_reviewed_data(self) -> dict:
        """Construit le dictionnaire attendu par apply_reviewed_analysis."""
        hypotheses = [
            {
                "selected": r["check"].isChecked(),
                "label": r["label"].text().strip(),
                "description": r["desc"].text().strip(),
                "prior_probability": r["prior"].value(),
                "horizon": r["horizon"].currentData(),
                "parent_hypothesis_label": r["parent"].currentData(),
            }
            for r in self.row_hyp if not r["removed"] and r["label"].text().strip()
        ]
        evidence_items = []
        for r in self.row_evidence:
            if r["removed"] or not r["content"].toPlainText().strip():
                continue
            ach = {
                lbl: combo.currentText()
                for lbl, combo in r["ach"].items()
                if combo.currentText() != "—"
            }
            evidence_items.append({
                "selected": r["check"].isChecked(),
                "content": r["content"].toPlainText().strip(),
                "source": r["source"].text().strip(),
                "credibility": r["cred"].value(),
                "ach_scores": ach,
                "bayesian_likelihoods": r["bayes"],
            })
        entities = [
            {
                "selected": r["check"].isChecked(),
                "name": r["name"].text().strip(),
                "type": r["type"].currentText(),
            }
            for r in self.row_ent if not r["removed"] and r["name"].text().strip()
        ]
        relations = [
            {
                "selected": r["check"].isChecked(),
                "source": r["source"].currentText().strip(),
                "target": r["target"].currentText().strip(),
                "type": r["type"].currentText(),
            }
            for r in self.row_rel
            if not r["removed"] and r["source"].currentText().strip()
            and r["target"].currentText().strip()
        ]
        predictions = [
            {
                "selected": r["check"].isChecked(),
                "question": r["question"].toPlainText().strip(),
                "probability": r["prob"].value(),
                "deadline": r["deadline"].date().toString("yyyy-MM-dd"),
                "category": r["category"].text().strip(),
            }
            for r in self.row_pred
            if not r["removed"] and r["question"].toPlainText().strip()
        ]
        bifurcation_nodes = [
            {
                "selected": r["check"].isChecked(),
                "label": r["label"].text().strip(),
                "condition_text": r["condition"].toPlainText().strip(),
                "horizon": r["horizon"].currentData(),
                "parent_hypothesis": r["parent"].currentData(),
                "leads_to_hypotheses": [
                    lbl for lbl, cb in r["leads"].items() if cb.isChecked()
                ],
            }
            for r in self.row_bn if not r["removed"] and r["label"].text().strip()
        ]
        return {
            "hypotheses": hypotheses,
            "evidence_items": evidence_items,
            "entities": entities,
            "relations": relations,
            "predictions": predictions,
            "bifurcation_nodes": bifurcation_nodes,
        }

    def _apply(self, keep_open: bool) -> None:
        """Applique la sélection en transaction atomique."""
        data = self.collect_reviewed_data()
        selected_total = sum(
            1 for section in data.values() for item in section
            if item.get("selected")
        )
        if selected_total == 0:
            QMessageBox.information(
                self, "Application", "Aucun élément sélectionné."
            )
            return
        try:
            self.db.apply_reviewed_analysis(self.project_id, data)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Insertion impossible",
                f"Aucune donnée insérée (transaction annulée) :\n{exc}",
            )
            return
        if self.on_applied:
            self.on_applied()
        if keep_open:
            QMessageBox.information(
                self, "Appliqué",
                f"{selected_total} élément(s) inséré(s). "
                "Vous pouvez analyser un autre texte.",
            )
            self.source_text.clear()
            self.explore_result.clear()
            self._clear_proposals()
            self.tabs.setTabEnabled(1, False)
            self.tabs.setCurrentIndex(0)
        else:
            QMessageBox.information(
                self, "Appliqué", f"{selected_total} élément(s) inséré(s)."
            )
            self.accept()


# =============================================================================
#  Dashboard (onglet « Vue d'ensemble »)
# =============================================================================

class DashboardCard(QFrame):
    """Carte du dashboard : QFrame bordé, titre, contenu ; double-clic navigable."""

    doubleClicked = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Construit une carte avec un titre et une zone de contenu."""
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(1)
        self.setToolTip("Double-cliquez pour ouvrir l'onglet correspondant.")

        layout = QVBoxLayout(self)
        self.title_label = QLabel(f"<b>{escape(title)}</b>")
        self.title_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.title_label)
        self.body = QVBoxLayout()
        layout.addLayout(self.body)
        layout.addStretch(1)

    def add_widget(self, widget: QWidget) -> None:
        """Ajoute un widget à la zone de contenu."""
        self.body.addWidget(widget)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001, N802
        """Émet doubleClicked au double-clic."""
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class DashboardView(QWidget):
    """Vue d'ensemble d'un projet : grille 2×2 de cartes synthétiques."""

    request_tab = Signal(int)  # index d'onglet à activer

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        """Construit la grille de cartes."""
        super().__init__(parent)
        self.db = db
        self.project_id: int | None = None

        # Carte ACH.
        self.ach_card = DashboardCard("Analyse (ACH)")
        self.ach_content = QLabel("—")
        self.ach_content.setWordWrap(True)
        self.ach_card.add_widget(self.ach_content)
        self.ach_card.doubleClicked.connect(lambda: self.request_tab.emit(1))

        # Carte bayésienne.
        self.bayes_card = DashboardCard("Mise à jour bayésienne")
        self.bayes_lead = QLabel("—")
        self.bayes_lead.setWordWrap(True)
        self.bayes_chart = BayesianBarChart()
        self.bayes_chart.setMinimumHeight(120)
        self.bayes_card.add_widget(self.bayes_lead)
        self.bayes_card.add_widget(self.bayes_chart)
        self.bayes_card.doubleClicked.connect(lambda: self.request_tab.emit(1))

        # Carte prédictions.
        self.pred_card = DashboardCard("Prédictions & calibration")
        self.pred_content = QLabel("—")
        self.pred_content.setWordWrap(True)
        self.pred_card.add_widget(self.pred_content)
        if MATPLOTLIB_AVAILABLE:
            self.pred_canvas: QWidget = ReliabilityCanvas()
            self.pred_canvas.setMinimumHeight(150)
            self.pred_card.add_widget(self.pred_canvas)
        else:
            self.pred_canvas = QLabel("(diagramme indisponible)")
            self.pred_canvas.setStyleSheet("color: #777;")
            self.pred_card.add_widget(self.pred_canvas)
        self.pred_card.doubleClicked.connect(lambda: self.request_tab.emit(2))

        # Carte graphe.
        self.graph_card = DashboardCard("Graphe de connaissances")
        self.graph_content = QLabel("—")
        self.graph_content.setWordWrap(True)
        self.graph_card.add_widget(self.graph_content)
        self.graph_card.doubleClicked.connect(lambda: self.request_tab.emit(3))

        grid = QGridLayout(self)
        grid.addWidget(self.ach_card, 0, 0)
        grid.addWidget(self.bayes_card, 0, 1)
        grid.addWidget(self.pred_card, 1, 0)
        grid.addWidget(self.graph_card, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    # --- API publique --------------------------------------------------------

    def set_database(self, db: Database) -> None:
        """Remplace la base active et réinitialise la vue."""
        self.db = db
        self.clear()

    def load_project(self, project_id: int) -> None:
        """Charge un projet et rafraîchit les cartes."""
        if self.db.get_project(project_id) is None:
            self.clear()
            return
        self.project_id = project_id
        self.refresh()

    def clear(self) -> None:
        """Réinitialise les cartes."""
        self.project_id = None
        self.ach_content.setText("Aucun projet sélectionné.")
        self.bayes_lead.setText("—")
        self.bayes_chart.set_data([])
        self.pred_content.setText("—")
        if MATPLOTLIB_AVAILABLE:
            self.pred_canvas.plot([])  # type: ignore[attr-defined]
        self.graph_content.setText("—")

    def refresh(self) -> None:
        """Recalcule et met à jour les quatre cartes depuis la base."""
        pid = self.project_id
        if pid is None:
            self.clear()
            return

        hyps = self.db.list_hypotheses(pid)
        evs = self.db.list_evidence(pid)
        cells = self.db.get_cells(pid)
        hyp_ids = [h["id"] for h in hyps]
        labels = {h["id"]: h["label"] for h in hyps}
        scores = {k: int(v["score"]) for k, v in cells.items() if v["score"] is not None}
        likelihoods = {
            k: (float(v["p_h"]), float(v["p_not_h"]))
            for k, v in cells.items()
            if v["p_h"] is not None and v["p_not_h"] is not None
        }
        priors = {
            h["id"]: (h["prior_probability"] if h["prior_probability"] is not None else 0.5)
            for h in hyps
        }
        evidence = [(e["id"], e["credibility"]) for e in evs]
        ev_labels = {e["id"]: f"E{i + 1}" for i, e in enumerate(evs)}

        # --- Carte ACH ---
        if hyps and evs:
            inc = compute_incoherence(hyp_ids, evidence, scores)
            rank = compute_ranking(hyp_ids, inc)
            diag = compute_diagnosticity(hyp_ids, evidence, scores)
            top3 = " > ".join(labels[h] for h in rank[:3]) if rank else "(non évalué)"
            best_ev = max(
                ((eid, d) for eid, d in diag.items() if d is not None),
                key=lambda kv: kv[1],
                default=None,
            )
            best_txt = (
                f"{ev_labels.get(best_ev[0], '?')} ({best_ev[1]:.2f})"
                if best_ev and best_ev[1] > 0
                else "—"
            )
            self.ach_content.setText(
                f"Preuves : {len(evs)} &nbsp;·&nbsp; Hypothèses : {len(hyps)}<br>"
                f"<b>Classement (top 3)</b> : {escape(top3)}<br>"
                f"<b>Preuve la plus diagnostique</b> : {escape(best_txt)}"
            )
        else:
            self.ach_content.setText(
                f"Preuves : {len(evs)} · Hypothèses : {len(hyps)}<br>"
                "Complétez la matrice pour le classement."
            )

        # --- Carte bayésienne ---
        if hyps:
            bayes = compute_bayesian(hyp_ids, priors, likelihoods)
            chart = [
                (h["label"], float(bayes[h["id"]]["prior"]),
                 float(bayes[h["id"]]["posterior"]))
                for h in hyps
            ]
            self.bayes_chart.set_data(chart)
            lead = max(hyps, key=lambda h: bayes[h["id"]]["posterior"])
            self.bayes_lead.setText(
                f"<b>En tête</b> : {escape(lead['label'])} "
                f"({bayes[lead['id']]['posterior'] * 100:.0f} %)"
            )
        else:
            self.bayes_chart.set_data([])
            self.bayes_lead.setText("Aucune hypothèse.")

        # --- Carte prédictions ---
        preds = self.db.list_predictions(pid)
        pairs = [
            (float(p["probability"]), int(p["outcome"]))
            for p in preds
            if p["outcome"] is not None and p["probability"] is not None
        ]
        active = sum(1 for p in preds if p["outcome"] is None)
        resolved = sum(1 for p in preds if p["outcome"] is not None)
        brier = brier_score(pairs)
        brier_txt = f"{brier:.3f}" if brier is not None else "— (aucune résolue)"
        self.pred_content.setText(
            f"<b>Brier global</b> : {brier_txt}<br>"
            f"Actives : {active} &nbsp;·&nbsp; Résolues : {resolved}"
        )
        if MATPLOTLIB_AVAILABLE:
            self.pred_canvas.plot(reliability_bins(pairs))  # type: ignore[attr-defined]

        # --- Carte graphe ---
        ents = self.db.list_entities(pid)
        rels = self.db.list_relationships(pid)
        degree: dict[int, int] = defaultdict(int)
        for r in rels:
            degree[r["source_entity_id"]] += 1
            degree[r["target_entity_id"]] += 1
        ent_names = {e["id"]: e["name"] for e in ents}
        top_conn = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_txt = (
            ", ".join(f"{ent_names.get(nid, '?')} ({deg})" for nid, deg in top_conn)
            if top_conn else "—"
        )
        self.graph_content.setText(
            f"Entités : {len(ents)} &nbsp;·&nbsp; Relations : {len(rels)}<br>"
            f"<b>Top 3 connectées</b> : {escape(top_txt)}"
        )


# =============================================================================
#  Vue projet (onglets ACH / Prédictions / Graphe)
# =============================================================================

class ProjectView(QWidget):
    """Conteneur à onglets pour un projet : Matrice ACH, Prédictions, Graphe."""

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        """Construit les onglets (Vue d'ensemble, ACH, Prédictions, Graphe)."""
        super().__init__(parent)
        self.db = db

        self.tabs = QTabWidget()
        self.dashboard = DashboardView(db)
        self.ach_view = ACHView(db)
        self.predictions_view = PredictionsView(db)
        self.graph_view = GraphView(db)

        self.tabs.addTab(self.dashboard, "Vue d'ensemble")   # index 0
        self.tabs.addTab(self.ach_view, "Matrice ACH")       # index 1
        self.tabs.addTab(self.predictions_view, "Prédictions")  # index 2
        self.tabs.addTab(self.graph_view, "Graphe")          # index 3

        self.dashboard.request_tab.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def _on_tab_changed(self, index: int) -> None:
        """Rafraîchit le dashboard quand on revient dessus (données à jour)."""
        if index == 0:
            self.dashboard.refresh()

    def set_database(self, db: Database) -> None:
        """Propage le changement de base de données aux onglets."""
        self.db = db
        self.dashboard.set_database(db)
        self.ach_view.set_database(db)
        self.predictions_view.set_database(db)
        self.graph_view.set_database(db)

    def load_project(self, project_id: int) -> None:
        """Charge un projet dans tous les onglets."""
        self.dashboard.load_project(project_id)
        self.ach_view.load_project(project_id)
        self.predictions_view.load_project(project_id)
        self.graph_view.load_project(project_id)

    def clear(self) -> None:
        """Réinitialise tous les onglets."""
        self.dashboard.clear()
        self.ach_view.clear()
        self.predictions_view.clear()
        self.graph_view.clear()


# =============================================================================
#  Fenêtre principale
# =============================================================================

class MainWindow(QMainWindow):
    """Fenêtre principale : menu, barre d'outils, liste de projets, vue projet."""

    def __init__(self, db: Database) -> None:
        """Construit l'interface et charge les projets existants."""
        super().__init__()
        self.db = db
        self._active_project_id: int | None = None
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800)

        self._build_central()
        self._build_menus()
        self._build_toolbar()

        # Barre de statut intelligente (résumé permanent).
        self.status_summary = QLabel("Aucun projet")
        self.statusBar().addPermanentWidget(self.status_summary)
        self.statusBar().showMessage("Prêt")

        self.project_view.tabs.currentChanged.connect(
            lambda _i: self._update_status_summary()
        )

        self._center_on_screen()
        self.refresh_projects()
        self._launch_spacy_check()
        self._restore_session()

    # --- Construction de l'interface -----------------------------------------

    def _build_central(self) -> None:
        """Crée la zone centrale : liste de projets + vue projet à onglets."""
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.project_list = QListWidget()
        self.project_list.currentItemChanged.connect(self._on_project_selected)
        self.project_list.itemDoubleClicked.connect(self._on_project_activated)
        left_layout.addWidget(self.project_list)

        project_buttons = QHBoxLayout()
        btn_new_project = QPushButton("＋ Nouveau")
        btn_new_project.clicked.connect(self.new_project)
        self.btn_delete_project = QPushButton("🗑 Supprimer")
        self.btn_delete_project.clicked.connect(self.delete_current_project)
        self.btn_delete_project.setEnabled(False)
        project_buttons.addWidget(btn_new_project)
        project_buttons.addWidget(self.btn_delete_project)
        left_layout.addLayout(project_buttons)

        splitter.addWidget(left_panel)

        self.project_view = ProjectView(self.db)
        splitter.addWidget(self.project_view)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([280, 920])

        self.stack.addWidget(splitter)
        self.stack.setCurrentIndex(0)

    def _build_menus(self) -> None:
        """Crée la barre de menus (Fichier, Aide) et ses actions."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Fichier")
        self.act_new = QAction("Nouveau projet", self)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)
        self.act_new.triggered.connect(self.new_project)
        file_menu.addAction(self.act_new)

        self.act_delete_project = QAction("Supprimer le projet…", self)
        self.act_delete_project.triggered.connect(self.delete_current_project)
        file_menu.addAction(self.act_delete_project)

        self.act_open = QAction("Ouvrir un document…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_document)
        file_menu.addAction(self.act_open)

        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Exporter")
        act_md = QAction("Rapport complet (Markdown)…", self)
        act_md.triggered.connect(self.export_markdown_report)
        export_menu.addAction(act_md)
        act_ach_csv = QAction("Matrice ACH (CSV)…", self)
        act_ach_csv.triggered.connect(self.export_ach_csv)
        export_menu.addAction(act_ach_csv)
        act_pred_csv = QAction("Prédictions (CSV)…", self)
        act_pred_csv.triggered.connect(self.export_predictions_csv)
        export_menu.addAction(act_pred_csv)
        act_graphml = QAction("Graphe (GraphML)…", self)
        act_graphml.triggered.connect(self.export_graphml)
        export_menu.addAction(act_graphml)

        self.act_save_copy = QAction("Sauvegarder une copie…", self)
        self.act_save_copy.triggered.connect(self.save_copy)
        file_menu.addAction(self.act_save_copy)

        file_menu.addSeparator()
        self.act_import_analyze = QAction("Importer et analyser…", self)
        self.act_import_analyze.triggered.connect(self.open_import_analyze)
        file_menu.addAction(self.act_import_analyze)

        self.act_patterns = QAction("Modifier les patterns NLP…", self)
        self.act_patterns.triggered.connect(self.edit_patterns)
        file_menu.addAction(self.act_patterns)

        file_menu.addSeparator()
        self.act_quit = QAction("Quitter", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)
        file_menu.addAction(self.act_quit)

        config_menu = menubar.addMenu("&Configuration")
        self.act_api_config = QAction("API LLM…", self)
        self.act_api_config.triggered.connect(self.open_api_config)
        config_menu.addAction(self.act_api_config)

        help_menu = menubar.addMenu("&Aide")
        self.act_about = QAction("À propos", self)
        self.act_about.triggered.connect(self.show_about)
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        """Crée la barre d'outils avec des icônes Unicode."""
        toolbar = QToolBar("Principale")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        tb_new = QAction("📁 Nouveau", self)
        tb_new.triggered.connect(self.new_project)
        toolbar.addAction(tb_new)

        tb_open = QAction("📂 Ouvrir", self)
        tb_open.triggered.connect(self.open_document)
        toolbar.addAction(tb_open)

        toolbar.addSeparator()
        self.tb_analyze = QAction("🤖 Analyser", self)
        self.tb_analyze.triggered.connect(self.open_import_analyze)
        toolbar.addAction(self.tb_analyze)

        toolbar.addSeparator()
        tb_about = QAction("❓ À propos", self)
        tb_about.triggered.connect(self.show_about)
        toolbar.addAction(tb_about)
        self._update_analyze_enabled()

    def _center_on_screen(self) -> None:
        """Centre la fenêtre sur l'écran courant."""
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        geo = self.frameGeometry()
        geo.moveCenter(available.center())
        self.move(geo.topLeft())

    # --- Logique applicative -------------------------------------------------

    def refresh_projects(self, select_id: int | None = None) -> None:
        """Recharge la liste des projets depuis la base."""
        self.project_list.blockSignals(True)
        self.project_list.clear()
        self.project_list.blockSignals(False)
        for row in self.db.list_projects():
            item = QListWidgetItem(row["name"])
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.project_list.addItem(item)
            if select_id is not None and row["id"] == select_id:
                self.project_list.setCurrentItem(item)
        if self.project_list.currentItem() is None:
            self.project_view.clear()

    def new_project(self) -> None:
        """Ouvre le dialogue de création puis insère le projet en base."""
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, description = dialog.get_data()
        if not name:
            return
        project_id = self.db.create_project(name, description)
        self.refresh_projects(select_id=project_id)
        self.statusBar().showMessage(f"Projet « {name} » créé.", 4000)

    def delete_current_project(self) -> None:
        """Supprime le projet sélectionné après confirmation (irréversible)."""
        item = self.project_list.currentItem()
        pid = self._active_project_id
        if item is None or pid is None or self.db.get_project(pid) is None:
            QMessageBox.information(
                self, "Suppression", "Sélectionnez d'abord un projet à supprimer."
            )
            return
        name = item.text()
        confirm = QMessageBox.warning(
            self,
            "Supprimer le projet",
            f"Supprimer définitivement le projet « {name} » et TOUTES ses données "
            "(hypothèses, preuves, matrice, prédictions, graphe, bifurcations) ?\n\n"
            "Cette action est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.db.delete_project(pid)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Suppression impossible", str(exc))
            return
        if self.settings.value("last_project_id", 0, type=int) == pid:
            self.settings.remove("last_project_id")
        self._active_project_id = None
        self.project_view.clear()
        self.refresh_projects()
        if hasattr(self, "btn_delete_project"):
            self.btn_delete_project.setEnabled(False)
        self._update_status_summary()
        self.statusBar().showMessage(f"Projet « {name} » supprimé.", 5000)

    def open_document(self) -> None:
        """Ouvre un autre document IRIS-Station (base SQLite) via un sélecteur."""
        start_dir = str(app_data_dir())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir un document IRIS-Station",
            start_dir,
            "Documents IRIS-Station (*.db *.sqlite *.sqlite3);;Tous les fichiers (*)",
        )
        if not path:
            return
        try:
            new_db = Database(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Ouverture impossible",
                f"Impossible d'ouvrir « {path} » :\n{exc}",
            )
            return

        old_db = self.db
        self.db = new_db
        self.project_view.set_database(new_db)
        if old_db is not None and old_db is not new_db:
            old_db.close()
        self.refresh_projects()
        self.statusBar().showMessage(f"Document ouvert : {path}", 6000)
        self.settings.setValue("last_db_path", str(self.db.db_path))

    def edit_patterns(self) -> None:
        """Ouvre patterns.json dans l'éditeur système (création si nécessaire)."""
        load_patterns()  # garantit l'existence du fichier
        path = patterns_path()
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        if not opened:
            QMessageBox.information(
                self,
                "Patterns NLP",
                "Le fichier de patterns se trouve ici :\n"
                f"{path}\n\n"
                "Modifiez-le puis reconstruisez le graphe pour appliquer vos règles.",
            )
        else:
            self.statusBar().showMessage(
                "patterns.json ouvert — reconstruisez le graphe après modification.",
                6000,
            )

    # --- Analyse assistée par LLM (Phase 8) ----------------------------------

    def _update_analyze_enabled(self) -> None:
        """Active « 🤖 Analyser » seulement si une clé API est configurée."""
        key = QSettings(ORG_NAME, APP_NAME).value("api_key", "", type=str)
        has_key = bool(key and key.strip())
        if hasattr(self, "tb_analyze"):
            self.tb_analyze.setEnabled(has_key)
            self.tb_analyze.setToolTip(
                "" if has_key
                else "Configurez d'abord la clé API (Configuration › API LLM)."
            )
        if hasattr(self, "act_import_analyze"):
            self.act_import_analyze.setEnabled(has_key)

    def open_api_config(self) -> None:
        """Ouvre le dialogue de configuration de l'API LLM."""
        dialog = ApiConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._update_analyze_enabled()
            self.statusBar().showMessage("Configuration API enregistrée.", 4000)

    def open_import_analyze(self) -> None:
        """Ouvre le dialogue d'import/analyse assistée par LLM."""
        pid = self._require_project()
        if pid is None:
            return
        key = QSettings(ORG_NAME, APP_NAME).value("api_key", "", type=str)
        if not (key and key.strip()):
            QMessageBox.information(
                self, "Analyse",
                "Aucune clé API configurée.\nMenu Configuration › API LLM.",
            )
            return
        dialog = ImportAnalyzeDialog(
            self.db, pid, on_applied=self._on_analysis_applied, parent=self
        )
        dialog.exec()

    def _on_analysis_applied(self) -> None:
        """Rafraîchit la vue projet après insertion d'une analyse validée."""
        if self._active_project_id is not None:
            self.project_view.load_project(self._active_project_id)
            self._update_status_summary()
            self.statusBar().showMessage("Analyse appliquée au projet.", 4000)

    # --- Export & sauvegarde (Phase 7) ---------------------------------------

    def _require_project(self) -> int | None:
        """Renvoie le projet actif, ou None (avec message) s'il n'y en a pas."""
        pid = self._active_project_id
        if pid is None or self.db.get_project(pid) is None:
            QMessageBox.information(
                self, "Export", "Sélectionnez d'abord un projet."
            )
            return None
        return pid

    def _choose_save_path(self, default_name: str, filt: str) -> str | None:
        """Ouvre un sélecteur d'enregistrement et renvoie le chemin choisi."""
        docs = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        ) or str(Path.home())
        start = str(Path(docs) / default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer sous", start, filt)
        return path or None

    @staticmethod
    def _safe_name(name: str) -> str:
        """Nettoie un nom de projet pour en faire un nom de fichier."""
        keep = [c if c.isalnum() or c in " -_" else "_" for c in (name or "projet")]
        return "".join(keep).strip().replace(" ", "_") or "projet"

    def save_copy(self) -> None:
        """Enregistre une copie horodatée du document (.db)."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"{self.db.db_path.stem}_{stamp}.db"
        path = self._choose_save_path(default, "Base IRIS-Station (*.db)")
        if not path:
            return
        try:
            self.db.backup_to(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Sauvegarde impossible", str(exc))
            return
        self.statusBar().showMessage(f"Copie enregistrée : {path}", 6000)

    def export_ach_csv(self) -> None:
        """Exporte la matrice ACH (codes de cohérence + incohérence) en CSV."""
        pid = self._require_project()
        if pid is None:
            return
        project = self.db.get_project(pid)
        path = self._choose_save_path(
            f"{self._safe_name(project['name'])}_matrice_ACH.csv", "CSV (*.csv)"
        )
        if not path:
            return
        hyps = self.db.list_hypotheses(pid)
        evs = self.db.list_evidence(pid)
        cells = self.db.get_cells(pid)
        scores = {k: int(v["score"]) for k, v in cells.items() if v["score"] is not None}
        hyp_ids = [h["id"] for h in hyps]
        evidence = [(e["id"], e["credibility"]) for e in evs]
        inc = compute_incoherence(hyp_ids, evidence, scores)
        diag = compute_diagnosticity(hyp_ids, evidence, scores)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["Preuve"] + [h["label"] for h in hyps] + ["Diagnosticité"]
                )
                for i, e in enumerate(evs):
                    row = [f"E{i + 1}"]
                    for h in hyps:
                        row.append(SCORE_TO_CODE.get(scores.get((e["id"], h["id"])), ""))
                    d = diag.get(e["id"])
                    row.append(f"{d:.3f}" if d is not None else "")
                    writer.writerow(row)
                inc_row = ["Score d'incohérence"]
                for h in hyps:
                    v = inc.get(h["id"])
                    inc_row.append(f"{v:.3f}" if v is not None else "")
                inc_row.append("")
                writer.writerow(inc_row)
        except OSError as exc:
            QMessageBox.critical(self, "Export impossible", str(exc))
            return
        self.statusBar().showMessage(f"Matrice ACH exportée : {path}", 6000)

    def export_predictions_csv(self) -> None:
        """Exporte les prédictions (avec Brier individuel) en CSV."""
        pid = self._require_project()
        if pid is None:
            return
        project = self.db.get_project(pid)
        path = self._choose_save_path(
            f"{self._safe_name(project['name'])}_predictions.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["question", "probabilité", "catégorie", "échéance", "créée",
                     "résolue", "issue", "source", "brier"]
                )
                for p in self.db.list_predictions(pid):
                    prob = p["probability"]
                    outcome = p["outcome"]
                    brier = (
                        f"{(prob - outcome) ** 2:.4f}"
                        if outcome is not None and prob is not None else ""
                    )
                    writer.writerow([
                        p["question"],
                        f"{prob:.4f}" if prob is not None else "",
                        p["category"] or "",
                        p["deadline"] or "",
                        p["created_at"] or "",
                        p["resolved_at"] or "",
                        "" if outcome is None else ("oui" if outcome == 1 else "non"),
                        p["resolution_source"] or "",
                        brier,
                    ])
        except OSError as exc:
            QMessageBox.critical(self, "Export impossible", str(exc))
            return
        self.statusBar().showMessage(f"Prédictions exportées : {path}", 6000)

    def export_graphml(self) -> None:
        """Exporte le graphe de connaissances au format GraphML (Gephi/Cytoscape)."""
        pid = self._require_project()
        if pid is None:
            return
        entities = self.db.list_entities(pid)
        if not entities:
            QMessageBox.information(
                self, "Export GraphML",
                "Le graphe est vide. Construisez-le d'abord dans l'onglet Graphe.",
            )
            return
        project = self.db.get_project(pid)
        path = self._choose_save_path(
            f"{self._safe_name(project['name'])}_graphe.graphml", "GraphML (*.graphml)"
        )
        if not path:
            return
        rels = self.db.list_relationships(pid)
        lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            "<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>",
            "  <key id='d_label' for='node' attr.name='label' attr.type='string'/>",
            "  <key id='d_type' for='node' attr.name='type' attr.type='string'/>",
            "  <key id='d_rel' for='edge' attr.name='relation' attr.type='string'/>",
            "  <key id='d_weight' for='edge' attr.name='weight' attr.type='double'/>",
            "  <graph edgedefault='directed'>",
        ]
        for e in entities:
            lines.append(
                f"    <node id='n{e['id']}'>"
                f"<data key='d_label'>{escape(e['name'] or '')}</data>"
                f"<data key='d_type'>{escape(e['type'] or '')}</data></node>"
            )
        for r in rels:
            directed = "false" if r["relation_type"] == "CO_OCCURS" else "true"
            lines.append(
                f"    <edge source='n{r['source_entity_id']}' "
                f"target='n{r['target_entity_id']}' directed='{directed}'>"
                f"<data key='d_rel'>{escape(r['relation_type'] or '')}</data>"
                f"<data key='d_weight'>{float(r['weight'] or 1.0)}</data></edge>"
            )
        lines += ["  </graph>", "</graphml>"]
        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Export impossible", str(exc))
            return
        self.statusBar().showMessage(f"Graphe exporté : {path}", 6000)

    def export_markdown_report(self) -> None:
        """Exporte un rapport Markdown complet (résumés TextRank en arrière-plan)."""
        pid = self._require_project()
        if pid is None:
            return
        project = self.db.get_project(pid)
        path = self._choose_save_path(
            f"{self._safe_name(project['name'])}_rapport.md", "Markdown (*.md)"
        )
        if not path:
            return
        self._report_path = path
        self._report_yaml = build_yggdrasil_yaml(self.db, pid)
        self._report_body = self._build_report_text(pid)
        evidence = [(e["id"], e["content"]) for e in self.db.list_evidence(pid)]
        self._report_labels = {
            eid: f"E{i + 1}" for i, (eid, _) in enumerate(evidence)
        }
        if not evidence:
            self._write_report("## 5. Résumés (TextRank)\n\n_Aucune preuve à résumer._")
            return
        self.statusBar().showMessage("Génération des résumés TextRank…")
        self._report_worker = ReportSummaryWorker(evidence)
        self._report_worker.done.connect(self._on_report_summaries)
        self._report_worker.failed.connect(self._on_report_summaries_failed)
        self._report_worker.start()

    def _on_report_summaries(self, summaries: dict) -> None:
        """Assemble la section des résumés puis écrit le rapport."""
        lines = ["## 5. Résumés (TextRank)", ""]
        for eid, label in self._report_labels.items():
            text = (summaries.get(eid) or "").strip()
            lines.append(f"### {label}")
            lines.append(text if text else "_(résumé indisponible)_")
            lines.append("")
        self._write_report("\n".join(lines))

    def _on_report_summaries_failed(self, message: str) -> None:
        """Écrit le rapport sans résumés si le NLP a échoué."""
        self._write_report(
            "## 5. Résumés (TextRank)\n\n_Résumés indisponibles : " + message + "_"
        )

    def _write_report(self, summaries_section: str) -> None:
        """Écrit le front matter Yggdrasil, le corps et les résumés sur disque."""
        front_matter = getattr(self, "_report_yaml", "") or ""
        try:
            Path(self._report_path).write_text(
                front_matter + self._report_body + "\n" + summaries_section + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Export impossible", str(exc))
            return
        note = "" if front_matter else (
            "\n\n(Note : PyYAML absent — front matter Yggdrasil non inclus.)"
        )
        self.statusBar().showMessage(f"Rapport exporté : {self._report_path}", 6000)
        QMessageBox.information(
            self, "Export", f"Rapport enregistré :\n{self._report_path}{note}"
        )

    def _build_report_text(self, pid: int) -> str:
        """Construit les sections ACH / Bayésien / Prédictions / Graphe du rapport."""
        def cell(text: object) -> str:
            return str(text if text is not None else "").replace("|", "/").replace("\n", " ")

        project = self.db.get_project(pid)
        hyps = self.db.list_hypotheses(pid)
        evs = self.db.list_evidence(pid)
        cells = self.db.get_cells(pid)
        hyp_ids = [h["id"] for h in hyps]
        labels = {h["id"]: h["label"] for h in hyps}
        scores = {k: int(v["score"]) for k, v in cells.items() if v["score"] is not None}
        likelihoods = {
            k: (float(v["p_h"]), float(v["p_not_h"]))
            for k, v in cells.items()
            if v["p_h"] is not None and v["p_not_h"] is not None
        }
        priors = {
            h["id"]: (h["prior_probability"] if h["prior_probability"] is not None else 0.5)
            for h in hyps
        }
        evidence = [(e["id"], e["credibility"]) for e in evs]
        ev_labels = {e["id"]: f"E{i + 1}" for i, e in enumerate(evs)}

        out = [
            f"# Rapport — {cell(project['name'])}",
            f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## 1. Analyse des hypothèses concurrentes (ACH)",
            "",
            "### Hypothèses",
        ]
        for h in hyps:
            prior = h["prior_probability"] if h["prior_probability"] is not None else 0.5
            out.append(
                f"- **{cell(h['label'])}** — {cell(h['description'])} "
                f"(a priori : {prior:.2f})"
            )
        out += ["", "### Preuves"]
        for i, e in enumerate(evs):
            out.append(
                f"- **E{i + 1}** — {cell(e['content'])} "
                f"(source : {cell(e['source']) or '—'}, crédibilité : {e['credibility']})"
            )

        if hyps and evs:
            inc = compute_incoherence(hyp_ids, evidence, scores)
            rank = compute_ranking(hyp_ids, inc)
            diag = compute_diagnosticity(hyp_ids, evidence, scores)
            out += ["", "### Matrice & résultats", ""]
            out.append(
                "Classement : "
                + (" > ".join(labels[h] for h in rank) if rank else "(non évalué)")
            )
            out.append("")
            out.append("| Preuve | " + " | ".join(cell(h["label"]) for h in hyps) + " |")
            out.append("|" + "---|" * (len(hyps) + 1))
            for i, e in enumerate(evs):
                cells_txt = [
                    SCORE_TO_CODE.get(scores.get((e["id"], h["id"])), "")
                    for h in hyps
                ]
                out.append(f"| E{i + 1} | " + " | ".join(cells_txt) + " |")
            inc_txt = [
                f"{inc[h['id']]:.2f}" if inc.get(h["id"]) is not None else "—"
                for h in hyps
            ]
            out.append("| Score d'incohérence | " + " | ".join(inc_txt) + " |")
            best_ev = max(
                ((eid, d) for eid, d in diag.items() if d is not None),
                key=lambda kv: kv[1], default=None,
            )
            if best_ev and best_ev[1] > 0:
                out.append("")
                out.append(
                    f"Preuve la plus diagnostique : {ev_labels[best_ev[0]]} "
                    f"({best_ev[1]:.2f})"
                )

        # --- Bayésien ---
        out += ["", "## 2. Mise à jour bayésienne", ""]
        if hyps:
            bayes = compute_bayesian(hyp_ids, priors, likelihoods)
            out.append("| Hypothèse | A priori | Vraisemblance cumulée | A posteriori |")
            out.append("|---|---|---|---|")
            for h in hyps:
                r = bayes[h["id"]]
                out.append(
                    f"| {cell(h['label'])} | {r['prior'] * 100:.1f} % | "
                    f"×{r['cumulative_lr']:.2f} | {r['posterior'] * 100:.1f} % |"
                )
        else:
            out.append("_Aucune hypothèse._")

        # --- Prédictions ---
        out += ["", "## 3. Prédictions & calibration", ""]
        preds = self.db.list_predictions(pid)
        pairs = [
            (float(p["probability"]), int(p["outcome"]))
            for p in preds
            if p["outcome"] is not None and p["probability"] is not None
        ]
        decomp = murphy_decomposition(pairs)
        if decomp:
            out.append(
                f"Brier = {decomp['brier']:.3f} | Calibration = {decomp['rel']:.3f} "
                f"| Résolution = {decomp['res']:.3f} | Incertitude = {decomp['unc']:.3f} "
                f"(n = {len(pairs)})"
            )
        else:
            out.append("_Aucune prédiction résolue._")
        out += ["", "| Question | Proba | Catégorie | Statut | Issue | Brier |",
                "|---|---|---|---|---|---|"]
        for p in preds:
            outcome = p["outcome"]
            prob = p["probability"]
            brier = (
                f"{(prob - outcome) ** 2:.3f}"
                if outcome is not None and prob is not None else "—"
            )
            issue = "—" if outcome is None else ("Oui" if outcome == 1 else "Non")
            status = "Résolue" if outcome is not None else "Ouverte"
            out.append(
                f"| {cell(p['question'])} | "
                f"{prob * 100:.0f} % | {cell(p['category']) or '—'} | "
                f"{status} | {issue} | {brier} |"
            )

        # --- Graphe ---
        out += ["", "## 4. Graphe de connaissances", ""]
        ents = self.db.list_entities(pid)
        rels = self.db.list_relationships(pid)
        out.append(f"Entités : {len(ents)} · Relations : {len(rels)}")
        if ents:
            degree: dict[int, int] = defaultdict(int)
            for r in rels:
                degree[r["source_entity_id"]] += 1
                degree[r["target_entity_id"]] += 1
            names = {e["id"]: e["name"] for e in ents}
            top = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:3]
            if top:
                out.append(
                    "Top entités connectées : "
                    + ", ".join(f"{cell(names.get(nid))} ({deg})" for nid, deg in top)
                )
            src_rows, total = self.db.source_reliability(pid)
            if src_rows:
                out.append("")
                out.append("Fiabilité des sources :")
                for r in src_rows[:5]:
                    appears = int(r["appears"]) or 1
                    rate = 100.0 * int(r["sourced"]) / appears
                    out.append(
                        f"- {cell(r['name'])} : {rate:.0f} % "
                        f"({int(r['sourced'])}/{appears} preuves)"
                    )

        return "\n".join(out)

    def _on_project_selected(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        """Charge le projet sélectionné dans la vue à onglets."""
        if hasattr(self, "btn_delete_project"):
            self.btn_delete_project.setEnabled(current is not None)
        if current is None:
            self._active_project_id = None
            self.project_view.clear()
            self._update_status_summary()
            return
        pid = current.data(Qt.ItemDataRole.UserRole)
        self._active_project_id = pid
        self.project_view.load_project(pid)
        self.statusBar().showMessage(f"Projet « {current.text()} » ouvert.", 4000)
        self._update_status_summary()
        self.settings.setValue("last_project_id", int(pid))
        self.settings.setValue("last_db_path", str(self.db.db_path))

    def _on_project_activated(self, item: QListWidgetItem) -> None:
        """Réagit au double-clic (équivalent à la sélection)."""
        self.project_list.setCurrentItem(item)

    def _update_status_summary(self) -> None:
        """Met à jour le résumé permanent de la barre de statut."""
        pid = self._active_project_id
        if pid is None or self.db.get_project(pid) is None:
            self.status_summary.setText("Aucun projet")
            return
        project = self.db.get_project(pid)
        n_ev = len(self.db.list_evidence(pid))
        n_hyp = len(self.db.list_hypotheses(pid))
        n_pred_active = sum(
            1 for p in self.db.list_predictions(pid) if p["outcome"] is None
        )
        n_ent = len(self.db.list_entities(pid))
        self.status_summary.setText(
            f"Projet : {project['name']}  |  preuves : {n_ev}  |  "
            f"hypothèses : {n_hyp}  |  prédictions actives : {n_pred_active}  |  "
            f"entités : {n_ent}"
        )

    def _restore_session(self) -> None:
        """Restaure la dernière base et le dernier projet ouverts (QSettings)."""
        last_db = self.settings.value("last_db_path", "", type=str)
        if last_db and Path(last_db) != self.db.db_path and Path(last_db).exists():
            try:
                new_db = Database(Path(last_db))
                old_db = self.db
                self.db = new_db
                self.project_view.set_database(new_db)
                if old_db is not None:
                    old_db.close()
                self.refresh_projects()
            except Exception:  # noqa: BLE001 - base illisible : on garde la défaut
                pass

        last_pid = self.settings.value("last_project_id", 0, type=int)
        if not last_pid:
            return
        for row in range(self.project_list.count()):
            item = self.project_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == last_pid:
                self.project_list.setCurrentItem(item)
                break

    def show_about(self) -> None:
        """Affiche la boîte de dialogue « À propos »."""
        QMessageBox.about(
            self,
            f"À propos de {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            "<p>Poste d'Analyse Structurée &amp; Calibreur de Prédictions.</p>"
            "<p><b>Phase 8</b> — analyse assistée par LLM (FantasyAI Cloud) avec "
            "interface de révision humaine avant insertion, export Markdown "
            "compatible Yggdrasil (front matter YAML, historique de probabilité, "
            "score de tension), détection de doublons. Le LLM ne fait que "
            "<i>proposer</i> : tous les calculs restent locaux et déterministes.</p>"
            "<p>ACH, bayésien, prédictions (Brier), graphe de connaissances, NLP "
            "local, dashboard et exports (CSV, GraphML).</p>"
            "<p>Local-first · hors-ligne · déterministe.</p>",
        )

    # --- Vérification spaCy au démarrage -------------------------------------

    def _launch_spacy_check(self) -> None:
        """Lance la vérification du modèle spaCy dans un thread d'arrière-plan."""
        self.statusBar().showMessage("Vérification du modèle spaCy…")
        self._spacy_worker = SpacyModelWorker(SPACY_MODEL)
        self._spacy_worker.progress.connect(self.statusBar().showMessage)
        self._spacy_worker.finished_check.connect(self._on_spacy_finished)
        self._spacy_worker.start()

    def _on_spacy_finished(self, success: bool, message: str) -> None:
        """Affiche le résultat de la vérification spaCy (non bloquant)."""
        prefix = "Prêt — " if success else "Prêt — ⚠ "
        self.statusBar().showMessage(prefix + message, 6000 if success else 8000)


# =============================================================================
#  Point d'entrée
# =============================================================================

def main() -> int:
    """Point d'entrée de l'application."""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)

    db = Database(app_data_dir() / DB_FILENAME)
    window = MainWindow(db)
    # La base courante peut changer (Ouvrir un document) : on ferme celle en cours.
    app.aboutToQuit.connect(lambda: window.db.close())

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
