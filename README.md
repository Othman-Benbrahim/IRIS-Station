# IRIS-Station

> **Poste d'Analyse Structurée & Calibreur de Prédictions** — une application de bureau *local-first*, hors-ligne et déterministe pour raisonner avec méthode et mesurer honnêtement sa calibration.


IRIS-Station réunit deux pratiques qui se renforcent mutuellement :

- **L'analyse structurée** répond à *« que se passe-t-il maintenant, et quelle est l'hypothèse la plus solide ? »* — via l'**Analyse des Hypothèses Concurrentes (ACH)**, la mise à jour **bayésienne** et un **graphe de connaissances**.
- **La calibration de prédictions** répond à *« qu'est-ce qui va arriver, et est-ce que je m'améliore avec le temps ? »* — via le **score de Brier**, la **décomposition de Murphy** et les **diagrammes de fiabilité**.

Ce n'est pas deux outils dans une même fenêtre, mais **un seul outil à deux faces** qui partagent le même socle de données : une analyse produit des prédictions chiffrées ; ces prédictions se résolvent avec le temps ; le scoring mesure la qualité réelle du raisonnement ; et ce retour améliore les analyses suivantes.

---

## Sommaire

- [Caractéristiques principales](#-caractéristiques-principales)
- [Philosophie de conception](#-philosophie-de-conception)
- [Captures d'écran](#-captures-décran)
- [Pile technique](#️-pile-technique)
- [Installation](#-installation)
- [Prise en main rapide](#-prise-en-main-rapide)
- [Analyse assistée par IA (optionnelle)](#-analyse-assistée-par-ia-optionnelle)
- [Exports & format de données](#-exports--format-de-données)
- [Structure du projet](#️-structure-du-projet)
- [Feuille de route](#️-feuille-de-route)
- [Confidentialité](#-confidentialité)
- [Licence](#-licence)

---

## ✨ Caractéristiques principales

### 🔍 Analyse des Hypothèses Concurrentes (ACH)
- Matrice **preuves × hypothèses** avec scores de cohérence (`CC`, `C`, `N`, `I`, `II`).
- **Classement** des hypothèses par score d'incohérence (la moins incohérente est la plus probable).
- **Diagnosticité** de chaque preuve (capacité à discriminer entre hypothèses).
- **Analyse de sensibilité** : quelles preuves sont critiques pour le classement ?

### 🎲 Moteur bayésien
- Mise à jour en **espace logarithmique** (log-odds) pour la stabilité numérique.
- Estimation guidée des vraisemblances (échelle 1–5 → probabilités calibrées).
- Probabilités **a posteriori**, **facteurs de Bayes** par cellule, visualisation en barres.

### 📈 Calibreur de prédictions
- **Score de Brier** global, par catégorie et par période.
- **Décomposition de Murphy** : Fiabilité (REL) − Résolution (RES) + Incertitude (UNC).
- **Diagramme de fiabilité** (10 buckets, barres d'erreur, droite de régression).
- **Détection de biais** : sur/sous-confiance, dérive temporelle.
- Historique complet des mises à jour de probabilité.

### 🕸️ Graphe de connaissances
- Extraction d'entités via **spaCy** (NER) et de relations via règles (`Matcher`).
- Détection de **communautés** (Louvain), **centralité d'intermédiarité** (Brandes), **plus court chemin** (CTE récursives SQLite).
- Score de **fiabilité des sources** et règles de relations **externalisées** dans `patterns.json` (éditable sans toucher au code).

### 🧠 NLP local
- **Résumé extractif** par TextRank (PageRank sur similarité cosinus de phrases).
- **Mots-clés** par TF-IDF.
- Tout est **local, rapide et auditable** — aucun appel réseau pour le traitement.

### 🗺️ Tableau de bord & exports
- Onglet **« Vue d'ensemble »** : grille de cartes synthétiques (ACH, bayésien, prédictions, graphe) avec navigation au double-clic.
- Exports : **rapport Markdown** (compatible Yggdrasil, voir plus bas), **matrice ACH (CSV)**, **prédictions (CSV)**, **graphe (GraphML)** pour Gephi/Cytoscape.
- Restauration automatique du dernier projet, sauvegarde de copie horodatée, barre de statut permanente.

### 🤖 Analyse assistée par IA *(optionnelle)*
- Pré-extraction d'une analyse ACH complète à partir d'un texte brut (article, notes, rapport).
- **Interface de révision humaine obligatoire** avant toute insertion : on coche, édite ou supprime chaque proposition.
- **Insertion atomique** (transaction tout-ou-rien) et **détection de doublons** (preuves par similarité d'embeddings, entités par normalisation).
- ➡️ **L'IA ne fait que proposer ; tous les calculs restent locaux et déterministes.**

---

## 🧭 Philosophie de conception

IRIS-Station repose sur cinq principes non négociables :

1. **Local-first.** Tout fonctionne hors-ligne. La base **SQLite est le fichier utilisateur** — portable, sauvegardable, versionnable.
2. **Déterministe.** Pour les mêmes entrées, l'application produit les mêmes sorties. Aucune boîte noire dans le traitement.
3. **Auditable.** Chaque résultat se trace jusqu'à ses données d'origine ; chaque calcul est vérifiable à la main.
4. **Overridable.** L'IA *suggère* mais n'impose jamais. L'utilisateur garde le dernier mot sur tout résultat automatique.
5. **Sans télémétrie.** Aucune donnée ne quitte la machine sans action explicite de l'utilisateur.

> Le module LLM est **périphérique, pas central** : si le fournisseur est indisponible, l'outil continue de fonctionner intégralement (extraction, calcul, scoring, graphe, résumés). Seule l'analyse assistée et la rédaction de synthèses sont dégradées.

---

## 📸 Captures d'écran


- [Vue d'ensemble](docs/dashboard.png)
- [Matrice ACH](docs/matrice-ach.png)
- [Diagramme de fiabilité](docs/calibration.png)
- [Graphe de connaissances](docs/graphe.png)


---

## 🏗️ Pile technique

| Couche | Technologie | Rôle |
|---|---|---|
| Interface | **PySide6** (Qt pour Python) | Application de bureau native, widgets Qt (pas de navigateur) |
| Base de données | **SQLite** (`sqlite3`, stdlib) | Embarquée, un seul fichier, CTE récursives pour le graphe |
| NLP | **spaCy** (`fr_core_news_sm`) | Tokenisation, lemmatisation, NER, *rule-based matching* |
| Calcul | **NumPy**, **SciPy** | Algèbre, similarité cosinus, statistiques |
| Visualisation | **Matplotlib** | Diagrammes de fiabilité, graphiques |
| Graphe | **NetworkX** | Disposition du graphe (le calcul reste en Python pur / SQL) |
| Export | **PyYAML** | Front matter YAML de l'export Yggdrasil |
| Réseau (optionnel) | **QtNetwork** | Appels à l'API LLM (analyse assistée uniquement) |

L'application est conçue pour être empaquetable en exécutable autonome via **PyInstaller**.

---

## 📦 Installation

### Prérequis
- **Python 3.10 ou supérieur**
- Les dépendances système habituelles de Qt (généralement déjà présentes sous Windows/macOS ; sous Linux, installez les bibliothèques Qt de votre distribution si nécessaire).

### 1. Cloner le dépôt
```bash
git clone https://github.com/<votre-compte>/iris-station.git
cd iris-station
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle spaCy français
```bash
python -m spacy download fr_core_news_sm
```

### 5. Lancer l'application
```bash
python main.py
```

> Au premier lancement, l'application crée sa base SQLite dans le dossier de données utilisateur de votre système. Le fichier `patterns.json` (règles de relations du graphe) est également généré et reste librement éditable.

---

## 🚀 Prise en main rapide

1. **Créez un projet** (menu *Fichier › Nouveau projet*) en formulant la question que vous analysez.
2. **Ajoutez des hypothèses concurrentes** (H1, H2, …) et, optionnellement, leur probabilité *a priori*.
3. **Ajoutez des preuves** (E1, E2, …) avec leur source et leur crédibilité (1–5).
4. **Remplissez la matrice ACH** : pour chaque couple preuve × hypothèse, indiquez la cohérence (`CC` → `II`). Le **classement** et la **diagnosticité** se calculent instantanément.
5. **Intégrez des vraisemblances bayésiennes** pour obtenir les probabilités *a posteriori* et les facteurs de Bayes.
6. **Posez des prédictions** chiffrées (onglet *Prédictions*), avec échéance et catégorie. Résolvez-les le moment venu (vrai/faux) pour alimenter votre **score de Brier** et votre **diagramme de fiabilité**.
7. **Construisez le graphe de connaissances** (onglet *Graphe*) pour faire émerger entités, relations et communautés.
8. **Consultez la *Vue d'ensemble*** pour une synthèse en un coup d'œil, puis **exportez** votre projet.

---

## 🤖 Analyse assistée par IA (optionnelle)

IRIS-Station peut **pré-mâcher** l'analyse d'un texte brut grâce à un LLM, **sans jamais court-circuiter votre jugement**.

### Configuration
1. Menu **Configuration › API LLM…**
2. Renseignez votre **clé API** (fournisseur OpenAI-compatible — *FantasyAI Cloud* par défaut). La clé est stockée localement dans les réglages de l'application, **jamais dans la base de données**.
3. Choisissez un modèle (la liste est récupérée automatiquement) et testez la connexion.

### Utilisation
1. Bouton **« 🤖 Analyser »** ou menu *Fichier › Importer et analyser…*
2. Collez votre texte, choisissez un mode (*Analyse complète*, *Complétion*, *Exploration*).
3. L'IA propose preuves, hypothèses, scores ACH, vraisemblances, entités, relations, prédictions et nœuds de bifurcation.
4. Dans l'onglet **Révision**, vous **validez, éditez ou rejetez** chaque élément. Les doublons probables sont signalés (🔴 preuves quasi-identiques, 🟠 entités à fusionner).
5. **« Appliquer la sélection »** insère le tout en **une transaction atomique** (en cas d'erreur, rien n'est écrit).

> **Garde-fou central :** le LLM *propose*, vous *disposez*. Aucun calcul (ACH, bayésien, scoring, graphe, résumés) n'est délégué au modèle ; tout reste local et déterministe.

---

## 📤 Exports & format de données

- **La base SQLite est le format de fichier.** Elle est portable et lisible par n'importe quel outil compatible SQLite.
- **Rapport Markdown compatible Yggdrasil** — un en-tête **front matter YAML** (`# YGGDRASIL_IMPORT: v1`) précède le rapport classique. Il encode, avec des identifiants positionnels (H1/E1/BN1/HZ1…) :
  - hypothèses (priors/posteriors, horizons, parents) et leur **historique de probabilité** (`probability_history`) ;
  - nœuds de bifurcation et leur **score de tension** (`tension_score`, qui mesure à quel point les branches filles sont incertaines) ;
  - preuves (diagnosticité, scores ACH, vraisemblances bayésiennes) ;
  - entités, relations et prédictions.
- **Matrice ACH (CSV)**, **Prédictions (CSV)** (avec Brier individuel), **Graphe (GraphML)** pour Gephi/Cytoscape.

---

## 🗂️ Structure du projet

L'application est actuellement distribuée en **un seul fichier** pour simplifier l'exécution et l'empaquetage :

```
iris-station/
├── main.py            # Application complète (UI + noyau algorithmique + accès données)
├── requirements.txt   # Dépendances Python
├── patterns.json      # Règles de relations du graphe (généré, éditable)
└── README.md
```

> Le noyau algorithmique (ACH, bayésien, scoring, graphe, NLP) est composé de **fonctions pures** indépendantes de l'interface, donc directement testables en isolation. Une scission en modules (`core/`, `db/`, `ui/`) est une évolution naturelle envisagée.

---

## 🛣️ Feuille de route

Fonctionnalités déjà en place : ACH, moteur bayésien, calibreur de prédictions, graphe de connaissances, NLP local (TextRank/TF-IDF), tableau de bord, exports (Markdown/Yggdrasil, CSV, GraphML), analyse assistée par IA avec révision humaine.

Pistes futures envisagées :
- **Recherche sémantique** sur un corpus documentaire personnel (indexation PDF/Markdown/HTML, embeddings locaux, recherche hybride lexicale + sémantique).
- **Raisonnement avancé** : agrégation bayésienne temporelle, raisonnement contrefactuel, matrice de confusion hypothèses ↔ prédictions.
- **Visualisations avancées** : timeline des analyses, treemap de calibration par catégorie.
- **Distribution** : exécutable signé, documentation utilisateur, format de fichier documenté pour l'interopérabilité.

---

## 🔒 Confidentialité

- **Aucune télémétrie**, aucune collecte. Vos données restent sur votre machine.
- Le seul accès réseau possible est l'**analyse assistée par IA**, strictement déclenchée par vous, et qui n'envoie que le texte que vous choisissez de soumettre.
- La clé API est conservée dans les réglages locaux de l'application, **jamais en base de données**.

---

## 📄 Licence

 À définir. Ajoutez un fichier [LICENSE](https://github.com/Othman-Benbrahim/IRIS-Station/blob/main/LICENSE) à la racine. Une licence permissive comme **MIT** est un choix courant pour ce type de projet :

```
MIT License — Copyright (c) 2026- Ben Brahim Othman-
```
## ⬇️ Téléchargement

Un binaire Windows de démonstration est disponible dans les [Releases](https://github.com/Othman-Benbrahim/IRIS-Station/releases/tag/v0.1.0-demo).

> ⚠️ **Binaire non signé — Windows peut afficher un avertissement** (SmartScreen / « Windows a protégé votre ordinateur »). Cliquez sur **« Informations complémentaires » › « Exécuter quand même »**. Pour une installation sans avertissement, **buildez depuis les sources** (voir Installation).

<p align="center">
  <sub>IRIS-Station — penser avec méthode, mesurer avec honnêteté.</sub>
</p>
