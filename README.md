# Chatbot RAG — Intelligence artificielle générative

Projet réalisé dans le cadre du TP d'IA générative : un chatbot qui répond à des questions sur l'intelligence artificielle générative en recherchant des informations dans des pages Wikipédia françaises.

## Fonctionnalités

- Récupération de 12 pages Wikipédia portant sur l'IA générative.
- Découpage des documents en *chunks* de 1 000 caractères, avec un chevauchement de 200 caractères.
- Création d'embeddings avec OpenAI ou Mistral AI, puis stockage dans ChromaDB.
- Recherche des 4 passages les plus pertinents pour chaque question.
- Génération d'une réponse en français, accompagnée des sources Wikipédia.
- Interface de conversation Streamlit avec historique et zone repliable présentant les passages récupérés.

## Prérequis

- Python 3.12 installé sur l'ordinateur.
- Git (pour cloner le projet).
- Une clé API **OpenAI** ou **Mistral AI** valide. Les appels API peuvent être soumis à des quotas ou à une facturation.
- Une connexion Internet, notamment au premier lancement pour accéder à Wikipédia.

## 1. Récupérer le projet

Ouvrir un terminal et exécuter :

```bash
git clone https://github.com/alexandragithub1/TP-RAG-IA-Generative.git
cd TP-RAG-IA-Generative
```

## 2. Créer l'environnement Python

Sous **Windows (PowerShell)** :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation, autoriser temporairement les scripts dans le terminal courant, puis réessayer :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Sous **macOS / Linux** :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Configurer la clé API

Créer le fichier `.env` à partir du modèle fourni :

Sous Windows (PowerShell) :

```powershell
Copy-Item .env.template .env
```

Sous macOS / Linux :

```bash
cp .env.template .env
```

Ouvrir `.env` et renseigner **une seule** des deux clés :

```dotenv
OPENAI_API_KEY=votre_cle_openai
MISTRAL_API_KEY=
LLM_PROVIDER=openai
```

Ou, pour Mistral AI :

```dotenv
OPENAI_API_KEY=
MISTRAL_API_KEY=votre_cle_mistral
LLM_PROVIDER=mistral
```

**Sécurité :** ne jamais publier ni partager une clé API. Le fichier `.env` est exclu de Git grâce à `.gitignore`. Le fichier `.env.template` ne doit contenir aucune clé réelle.

## 4. Lancer le chatbot

### Avec l'interface web Streamlit (recommandé)

```powershell
python -m streamlit run app.py
```

Ouvrir l'adresse affichée dans le terminal, généralement **http://localhost:8501**.

### Dans le terminal

```powershell
python main.py
```

## 5. Exemple d'utilisation

Poser une question comme :

> Qui a créé LLaMA ?

Le système recherche les passages pertinents dans sa base documentaire, puis génère une réponse, par exemple :

> LLaMA a été développé par Meta.

Les liens vers les pages Wikipédia sélectionnées sont affichés sous la réponse. Dans Streamlit, la section **« Passages récupérés »** permet de consulter les extraits qui ont été transmis au modèle.

## Fonctionnement technique

Le pipeline RAG suit ces étapes :

1. **Chargement** des pages Wikipédia et de leurs métadonnées (titre et URL).
2. **Découpage** en chunks avec `RecursiveCharacterTextSplitter`.
3. **Vectorisation** avec le modèle d'embeddings choisi.
4. **Indexation** et persistance locale dans ChromaDB.
5. **Recherche** des 4 chunks les plus proches de la question.
6. **Génération** de la réponse par un modèle de langage, à partir du contexte retrouvé.

La base locale est enregistrée dans un dossier `chroma_db_openai/` ou `chroma_db_mistral/`. Elle est réutilisée lorsqu'elle existe, pour éviter de recalculer les embeddings de tous les documents au lancement suivant. Elle n'est pas publiée sur GitHub.

## En cas de problème

- **Erreur Wikipédia `429 Too Many Requests` :** le site limite temporairement le nombre de requêtes. Réessayer plus tard.
- **Erreur de clé API :** vérifier que le fichier `.env` existe, que la clé est valide et que `LLM_PROVIDER` correspond au fournisseur choisi.
- **Erreur liée au modèle ou au quota :** vérifier l'accès au modèle et les crédits ou limites du compte API. Le modèle peut être personnalisé avec `CHAT_MODEL` dans `.env`.
- **Absence de réponse pertinente :** vérifier les documents récupérés dans la zone « Passages récupérés » de l'interface Streamlit.

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| `main.py` | Chargement des documents, ChromaDB, recherche et génération RAG |
| `app.py` | Interface de chat Streamlit |
| `requirements.txt` | Dépendances Python |
| `.env.template` | Modèle de configuration sans secrets |
| `.gitignore` | Exclusion des secrets et fichiers générés |
| `README.md` | Installation et utilisation du projet |
