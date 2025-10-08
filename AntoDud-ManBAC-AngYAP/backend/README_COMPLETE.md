# Backend - Générateur d'Histoires Interactives

API REST construite avec FastAPI pour la génération d'histoires interactives avec IA.

## 🚀 Installation et Configuration

### 1. Prérequis
- Python 3.8+
- pip (gestionnaire de paquets Python)
- Au moins 8GB de RAM libre (pour les modèles IA)

### 2. Création de l'environnement virtuel
```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate
```

### 3. Installation des dépendances
```bash
pip install -r requirements.txt
```

**Note importante** : L'installation peut prendre du temps (10-30 minutes) car elle télécharge des modèles IA volumineux.

### 4. Configuration
Créez un fichier `.env` à la racine du backend :
```env
# Configuration de l'environnement
ENVIRONMENT=development
DEBUG=True

# CORS - origines autorisées (séparées par des virgules)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Configuration IA
HUGGINGFACE_TOKEN=your_token_here
MODEL_CACHE_DIR=./models

# Configuration des images
IMAGES_OUTPUT_DIR=./generated_images
MAX_IMAGE_SIZE=1024

# Configuration de la mémoire
MEMORY_STORAGE_DIR=./memory_storage
MAX_MEMORY_ENTRIES=1000
```

## 🏃‍♂️ Démarrage du serveur

### Première fois - Initialisation des modèles IA
Le premier démarrage prend plus de temps car les modèles IA sont téléchargés et initialisés :

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Vous verrez des messages comme :
```
🚀 Démarrage de l'application...
📝 Chargement du modèle de génération narrative...
🎨 Chargement du modèle de génération d'images...
✅ Modèle de génération narrative chargé
✅ Modèle de génération d'images chargé
📊 RAPPORT D'INITIALISATION DES SERVICES IA
✅ NARRATIVE    : ACTIF
✅ IMAGE        : ACTIF
✅ MEMORY       : ACTIF
✅ STORY        : ACTIF
🎉 Tous les services IA sont opérationnels!
```

### Démarrages ultérieurs
Les démarrages suivants sont plus rapides car les modèles sont mis en cache.

### Mode production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Important** : Utilisez un seul worker en production pour éviter de dupliquer les modèles IA en mémoire.

## 📖 Accès à la documentation

Une fois le serveur démarré, vous pouvez accéder à :
- **Documentation Swagger**: http://localhost:8000/docs
- **Documentation ReDoc**: http://localhost:8000/redoc
- **Santé de l'API**: http://localhost:8000/api/v1/health
- **État des services IA**: http://localhost:8000/api/v1/health/ai-services

## 🔧 Endpoints principaux

### Santé et monitoring
- `GET /api/v1/health` - Santé générale de l'API
- `GET /api/v1/health/ai-services` - État détaillé des services IA

### Histoires
- `POST /api/v1/stories` - Créer une nouvelle histoire
- `GET /api/v1/stories/{story_id}` - Récupérer une histoire
- `POST /api/v1/stories/{story_id}/action` - Effectuer une action dans l'histoire
- `DELETE /api/v1/stories/{story_id}` - Supprimer une histoire

### Images
- `POST /api/v1/images/generate` - Générer une image pour une scène

## 🧪 Tests

Pour lancer les tests de base :
```bash
python test_backend.py
```

Ce script teste :
- La disponibilité du serveur
- Les endpoints de santé
- L'état des services IA
- La création d'une histoire simple
- L'exécution d'une action dans l'histoire

## 🛠️ Dépannage

### Problèmes courants

1. **Erreur de mémoire insuffisante**
   - Fermez d'autres applications
   - Augmentez la mémoire virtuelle
   - Utilisez des modèles plus petits si disponibles

2. **Échec de chargement des modèles**
   - Vérifiez votre connexion internet
   - Vérifiez l'espace disque disponible (>10GB recommandés)
   - Consultez les logs pour identifier le modèle défaillant

3. **Erreur de token Hugging Face**
   - Créez un compte sur https://huggingface.co
   - Générez un token d'accès
   - Ajoutez-le dans votre fichier `.env`

### Modes dégradés

L'application peut fonctionner en mode dégradé si certains services IA échouent :
- **Narrative Service indisponible** : Histoires avec contenu statique
- **Image Service indisponible** : Pas de génération d'images
- **Memory Service toujours actif** : Ne nécessite pas de modèle externe

## 📁 Structure du projet

```
backend/
├── main.py                 # Point d'entrée de l'application
├── requirements.txt        # Dépendances Python
├── .env.example           # Modèle de configuration
├── test_backend.py        # Tests de base
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration de l'application
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py     # Modèles de données Pydantic
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py      # Endpoints de santé
│   │   ├── story.py       # Endpoints des histoires
│   │   └── image.py       # Endpoints des images
│   └── services/
│       ├── __init__.py
│       ├── ai_startup.py          # Gestionnaire des services IA
│       ├── story_service.py       # Service principal des histoires
│       ├── narrative_service.py   # Service de génération de texte
│       ├── image_service.py       # Service de génération d'images
│       └── memory_service.py      # Service de mémoire intelligente
```

## 🎯 Fonctionnalités IA implémentées

### 1. Génération narrative (narrative_service.py)
- Modèles de texte : Qwen/Qwen2.5-3B-Instruct ou similaires
- Prompts spécialisés par genre d'histoire
- Génération contextuelle basée sur la mémoire
- Fallback vers contenu statique en cas d'échec

### 2. Génération d'images (image_service.py)
- Modèle : Stable Diffusion XL
- Styles adaptatifs selon le genre
- Cache des images générées
- Optimisation des prompts visuels

### 3. Mémoire intelligente (memory_service.py)
- Extraction automatique d'entités
- Tracking des personnages et lieux
- Gestion de la cohérence narrative
- Système de scoring d'importance

### 4. Orchestration (story_service.py)
- Coordination de tous les services IA
- Gestion des erreurs gracieuse
- Fallbacks automatiques
- Sauvegarde JSON des histoires

## 🚀 Prochaines étapes

Une fois le backend fonctionnel :

1. **Testez l'API** avec `python test_backend.py`
2. **Développez le frontend** React/Next.js
3. **Intégrez les services** via les endpoints REST
4. **Déployez en production** avec les optimisations appropriées

Le backend est maintenant complet avec toutes les fonctionnalités IA implémentées ! 🎉