# GUIDE DE CONFIGURATION - Version Dash

Document detaillant les etapes exactes pour deployer la version Dash sur GitHub.

---

## PREREQUIS

Avant de commencer, verifiez que vous avez :
- Git installe sur votre machine
- Python 3.9+ installe
- Acces au repository GitHub
- Les fichiers Dash fournis (app.py, requirements.txt, README.md)

---

## ETAPES A SUIVRE

### ETAPE 1 : Sauvegarder la version Streamlit actuelle

```bash
# Ouvrir le terminal et naviguer vers le projet
cd chemin/vers/Quant-Option-Portfolio

# Verifier que tout est commite
git status

# Si des fichiers ne sont pas commites
git add .
git commit -m "Save Streamlit version before Dash migration"

# Creer un tag pour marquer la version Streamlit
git tag -a v2.0-streamlit -m "Version Streamlit complete avec toutes les fonctionnalites"

# Pousser le tag sur GitHub
git push origin v2.0-streamlit
git push origin main
```

### ETAPE 2 : Creer la nouvelle branche Dash

```bash
# Creer et basculer sur la nouvelle branche
git checkout -b dash-version

# Verifier que vous etes sur la bonne branche
git branch
# Vous devez voir : * dash-version
```

### ETAPE 3 : Nettoyer le repertoire pour Dash

```bash
# Supprimer les fichiers Streamlit (on les garde sur main)
rm app.py
rm style.css 2>/dev/null  # Peut ne pas exister

# Garder les fichiers de documentation et notebooks
# Ne pas supprimer : README.md, LICENSE, notebooks, data files
```

### ETAPE 4 : Copier les fichiers Dash

Copiez les fichiers suivants dans le repertoire du projet :
- app.py (nouvelle version Dash)
- requirements.txt (mis a jour pour Dash)
- README.md (mis a jour)
- SETUP_GUIDE.md (ce fichier)

```bash
# Depuis le dossier ou vous avez telecharge les fichiers
cp chemin/vers/downloads/app.py ./app.py
cp chemin/vers/downloads/requirements.txt ./requirements.txt
cp chemin/vers/downloads/README.md ./README.md
cp chemin/vers/downloads/SETUP_GUIDE.md ./SETUP_GUIDE.md
```

### ETAPE 5 : Tester en local

```bash
# Creer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur macOS/Linux :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# Installer les dependances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

Ouvrez votre navigateur a l'adresse : http://localhost:8050

### ETAPE 6 : Verifier le fonctionnement

Checklist de verification :

- [ ] L'application se lance sans erreur
- [ ] La sidebar affiche les parametres
- [ ] Les donnees de marche se chargent (SPY, etc.)
- [ ] L'onglet Dashboard affiche les metriques
- [ ] L'onglet CRR Model affiche l'arbre binomial
- [ ] L'onglet Convergence affiche les graphiques
- [ ] L'onglet Vol Surface charge les donnees (cliquer sur le bouton)
- [ ] L'onglet Hedging lance la simulation
- [ ] L'onglet P&L Analysis affiche les heatmaps
- [ ] L'onglet Theory affiche les formules

### ETAPE 7 : Commit et Push sur GitHub

```bash
# Arreter le serveur (Ctrl+C dans le terminal)

# Ajouter tous les fichiers
git add .

# Creer le commit
git commit -m "Migrate to Dash framework

Major changes:
- Complete rewrite using Dash and Plotly
- Professional dark theme design
- Real-time market data integration via yfinance
- 7 interactive tabs
- Monte Carlo hedging simulation
- Greeks heatmaps
- 3D volatility surface
- Stress testing scenarios

Technical:
- Dash 2.14+ with Bootstrap components
- Callbacks for reactive updates
- Modular code structure
- Production-ready with Gunicorn support"

# Pousser la branche sur GitHub
git push -u origin dash-version
```

### ETAPE 8 : Creer une Pull Request (optionnel)

1. Allez sur https://github.com/AlexisAHG/Quant-Option-Portfolio
2. Vous verrez un bandeau "dash-version had recent pushes"
3. Cliquez sur "Compare & pull request" si vous voulez fusionner avec main
4. Ou gardez les deux branches separees

---

## STRUCTURE FINALE DU REPOSITORY

Apres toutes les etapes :

```
Quant-Option-Portfolio/
│
├── [branche: main] - Version Streamlit
│   ├── app.py (Streamlit)
│   ├── requirements.txt
│   └── ...
│
└── [branche: dash-version] - Version Dash
    ├── app.py (Dash)
    ├── requirements.txt
    ├── README.md
    ├── SETUP_GUIDE.md
    ├── LICENSE
    ├── Notebooks/
    │   ├── Stochastic_Volatility_Models.ipynb
    │   └── ...
    └── Data/
        ├── spx.csv
        └── vix_daily.csv
```

---

## COMMANDES GIT UTILES

### Basculer entre les versions

```bash
# Aller sur la version Streamlit
git checkout main

# Aller sur la version Dash
git checkout dash-version
```

### Voir l'historique

```bash
# Voir tous les commits
git log --oneline

# Voir les branches
git branch -a

# Voir les tags
git tag -l
```

### En cas de probleme

```bash
# Annuler les modifications non commitees
git checkout -- .

# Revenir a un commit precedent
git reset --hard HEAD~1

# Revenir a la version Streamlit taguee
git checkout v2.0-streamlit
```

---

## RESOLUTION DES PROBLEMES COURANTS

### Erreur : ModuleNotFoundError

```bash
# Verifier que l'environnement virtuel est active
which python  # Doit pointer vers venv/bin/python

# Reinstaller les dependances
pip install -r requirements.txt
```

### Erreur : Port 8050 deja utilise

```bash
# Trouver le processus qui utilise le port
lsof -i :8050

# Tuer le processus
kill -9 <PID>

# Ou changer le port dans app.py
app.run_server(debug=True, port=8051)
```

### Les donnees de marche ne se chargent pas

- Verifier la connexion internet
- Verifier que yfinance est installe : pip install yfinance
- Essayer un autre ticker (QQQ au lieu de SPY)

### Le graphique de volatility surface est vide

- Cliquer sur "Load Volatility Surface"
- Attendre le chargement (peut prendre quelques secondes)
- Essayer avec SPY ou QQQ (plus liquides)

---

## DEPLOIEMENT EN PRODUCTION (OPTIONNEL)

### Render.com (gratuit)

1. Creer un compte sur render.com
2. New > Web Service
3. Connecter le repository GitHub
4. Selectionner la branche dash-version
5. Build Command: pip install -r requirements.txt
6. Start Command: gunicorn app:server
7. Deployer

### Fichier Procfile (pour Heroku/Render)

Creer un fichier nomme `Procfile` (sans extension) :
```
web: gunicorn app:server
```

---

## CONTACT ET SUPPORT

En cas de probleme :
1. Verifier ce guide
2. Consulter la documentation Dash : https://dash.plotly.com/
3. Contacter l'equipe projet

---

Document cree le 18/01/2026
Projet Pi2 - ESILV
