"""
Point d'entrée principal de l'API Interactive Story Generator

Ce fichier configure et lance l'application FastAPI avec :
- Configuration CORS pour le frontend
- Inclusion des routeurs API
- Middleware de logging et gestion d'erreurs
- Documentation automatique Swagger/ReDoc
- Initialisation automatique des services IA
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.routers import story, health, image
from app.services.ai_startup import initialize_ai_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application
    
    Cette fonction s'exécute au démarrage et à l'arrêt de l'application
    pour initialiser les services IA et nettoyer les ressources.
    """
    # Démarrage : initialisation des services IA
    print("🚀 Démarrage de l'application...")
    await initialize_ai_services()
    
    yield  # L'application fonctionne ici
    
    # Arrêt : nettoyage des ressources
    print("🔄 Arrêt de l'application...")

# Création de l'application FastAPI avec métadonnées pour la documentation
app = FastAPI(
    title="Interactive Story Generator API",
    description="""
    API pour la génération d'histoires interactives avec images.
    
    Fonctionnalités principales :
    - Création d'histoires dans différents genres
    - Progression narrative basée sur les actions du joueur
    - Génération d'images pour chaque scène
    - Système de mémoire pour la cohérence narrative
    - Sauvegarde et reprise des histoires
    """,
    version="1.0.0",
    docs_url="/docs",  # Documentation Swagger
    redoc_url="/redoc",  # Documentation ReDoc alternative
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des routeurs API
# Chaque routeur gère un domaine fonctionnel spécifique
app.include_router(health.router, prefix="/api/v1")
app.include_router(story.router, prefix="/api/v1")
app.include_router(image.router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Endpoint racine pour vérifier que l'API fonctionne
    Accessible à GET /
    """
    return {
        "message": "Interactive Story Generator API is running",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    """
    Point d'entrée pour lancer l'application en développement
    En production, utiliser un serveur ASGI comme uvicorn, gunicorn, etc.
    """
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )