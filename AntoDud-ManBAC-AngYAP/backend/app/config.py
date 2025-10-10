"""
Configuration centralisée de l'application

Ce fichier contient tous les paramètres configurables de l'application :
- Variables d'environnement
- Configuration des modèles IA
- Paramètres de stockage
- Configuration réseau et sécurité

Utilise Pydantic Settings pour la validation automatique des types
et le chargement depuis les variables d'environnement et fichiers .env
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Déterminer le dossier racine du backend (où se trouve ce fichier config.py)
BACKEND_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = BACKEND_ROOT / "data"


class Settings(BaseSettings):
    """
    Classe de configuration principale utilisant Pydantic Settings
    
    Les valeurs peuvent être surchargées par :
    1. Variables d'environnement (ex: export DEBUG=false)
    2. Fichier .env dans le répertoire racine
    3. Valeurs par défaut définies ici
    """
    # Environment
    ENV: str = "development"
    DEBUG: bool = True

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # AI Models
    TEXT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    TEXT_MODEL_DEVICE: str = "auto"  # "auto" sélectionne CUDA/MPS si disponible
    IMAGE_MODEL_NAME: str = "stabilityai/sdxl-turbo"
    IMAGE_MODEL_DEVICE: str = "auto"  # "auto" sélectionne CUDA/MPS si disponible
    MODEL_CACHE_PATH: str = str(DATA_ROOT / "models")

    # Dev server reload configuration
    RELOAD_EXCLUDE_PATTERNS: str = "data/**"

    # External APIs
    OPENAI_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""

    # Story Configuration
    MAX_STORY_LENGTH: int = 10000
    MAX_CONTEXT_LENGTH: int = 4000
    DEFAULT_GENRE: str = "fantasy"

    # File Storage
    STORIES_PATH: str = str(DATA_ROOT / "stories")
    IMAGES_PATH: str = str(DATA_ROOT / "images")

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    class Config:
        env_file = ".env"


settings = Settings()
