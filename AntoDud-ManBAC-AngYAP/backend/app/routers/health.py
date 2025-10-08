"""
Routeur pour les endpoints de santé et monitoring

Ce module fournit des endpoints pour :
- Vérifier l'état de l'API
- Monitoring des services
- Informations de version et configuration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
import time

from app.services.ai_startup import get_ai_services_status
from datetime import datetime

from app.models.schemas import HealthResponse

# Création du routeur avec tags pour la documentation
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de vérification de santé de l'API
    
    Retourne l'état actuel des services et composants :
    - Statut général de l'API
    - Timestamp de la vérification
    - Version de l'application
    - État des modèles IA (à implémenter)
    
    Returns:
        HealthResponse: Informations de santé de l'API
    """
    # Récupération de l'état actuel des services IA
    ai_services_status = get_ai_services_status()
    
    models_status = {
        "text_model": ai_services_status.get("narrative", False),
        "image_model": ai_services_status.get("image", False),
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=models_status
    )


@router.get("/health/ai-services")
async def ai_services_health():
    """
    État spécifique des services IA
    
    Retourne l'état détaillé de tous les services IA :
    - Narrative Service (génération de texte)
    - Image Service (génération d'images)
    - Memory Service (gestion de la mémoire)
    - Story Service (orchestration)
    """
    services_status = get_ai_services_status()
    
    active_services = sum(services_status.values())
    total_services = len(services_status)
    
    if active_services == total_services:
        overall_status = "operational"
    elif active_services > 0:
        overall_status = "partial"
    else:
        overall_status = "unavailable"
    
    return {
        "overall_status": overall_status,
        "services": services_status,
        "summary": f"{active_services}/{total_services} services active",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }