"""
Routeur API pour les opérations sur les images

Ce module contient les endpoints REST pour :
- Générer des images standalone pour des scènes
- Récupérer des images existantes
- Gérer le cache d'images

Architecture REST :
- POST /images/generate : Générer une image standalone
- GET /images/{story_id}/{scene_number} : Récupérer une image de scène
"""

from fastapi import APIRouter, HTTPException, Path, Depends
from typing import Optional

from app.models.schemas import ImageGenerationRequest, ImageResponse
from app.services.image_service import ImageService


# Création du routeur avec préfixe et tags pour la documentation
router = APIRouter(prefix="/images", tags=["images"])


# Instance du service (Dependency Injection pattern)
def get_image_service() -> ImageService:
    """
    Factory function pour l'injection de dépendance du ImageService
    Permet de faciliter les tests et la maintenance
    """
    return ImageService()


@router.post("/generate", response_model=ImageResponse, status_code=201)
async def generate_image_standalone(
    request: ImageGenerationRequest,
    image_service: ImageService = Depends(get_image_service)
) -> ImageResponse:
    """
    Génère une image de manière autonome
    
    Ce endpoint permet de générer des images indépendamment
    du flux principal des histoires, utile pour :
    - Tests de génération d'images
    - Création d'illustrations custom
    - Preview de styles visuels
    
    Args:
        request: Données de génération (prompt, style, IDs)
        image_service: Service injecté pour la génération d'images
        
    Returns:
        ImageResponse: Métadonnées de l'image générée
        
    Raises:
        HTTPException 400: Si les données de génération sont invalides
        HTTPException 500: Si erreur lors de la génération
        HTTPException 503: Si le service d'images n'est pas disponible
    """
    try:
        # Vérification que le service est disponible
        if not image_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Service de génération d'images indisponible"
            )
        
        # Génération de l'image
        image_response = await image_service.generate_image_standalone(request)
        
        if not image_response:
            raise HTTPException(
                status_code=500,
                detail="Échec de la génération d'image"
            )
        
        return image_response
        
    except HTTPException:
        # Re-raise des HTTPException pour préserver les codes d'erreur
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération d'image: {str(e)}"
        )


@router.get("/{story_id}/scene/{scene_number}")
async def get_scene_image(
    story_id: str = Path(..., description="ID unique de l'histoire"),
    scene_number: int = Path(..., description="Numéro de la scène"),
    image_service: ImageService = Depends(get_image_service)
) -> dict:
    """
    Récupère les métadonnées d'une image de scène
    
    Args:
        story_id: Identifiant unique de l'histoire
        scene_number: Numéro de la scène
        image_service: Service injecté pour la gestion des images
        
    Returns:
        dict: Informations sur l'image de la scène
        
    Raises:
        HTTPException 404: Si l'image n'existe pas
        HTTPException 500: Si erreur lors de la récupération
    """
    try:
        # TODO: Implémenter la récupération d'images existantes
        # Cette fonctionnalité sera ajoutée dans une phase ultérieure
        
        return {
            "message": "Récupération d'images existantes pas encore implémentée",
            "story_id": story_id,
            "scene_number": scene_number
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération: {str(e)}"
        )


@router.get("/health")
async def image_service_health(
    image_service: ImageService = Depends(get_image_service)
) -> dict:
    """
    Vérifie l'état du service de génération d'images
    
    Args:
        image_service: Service injecté pour la génération d'images
        
    Returns:
        dict: État du service d'images
    """
    return {
        "service": "image_generation",
        "status": "ready" if image_service.is_model_loaded() else "loading",
        "model_loaded": image_service.is_model_loaded()
    }