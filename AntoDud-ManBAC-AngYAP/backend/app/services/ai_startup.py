"""
Script de démarrage pour l'initialisation des services IA

Ce module gère le chargement et l'initialisation de tous les services IA
au démarrage de l'application. Il permet de :
- Charger les modèles de manière asynchrone
- Gérer les erreurs de chargement gracieusement
- Fournir un feedback sur l'état des services
- Permettre un démarrage même si certains modèles échouent

Utilisation :
- Appelé automatiquement au démarrage de FastAPI
- Peut être appelé manuellement pour recharger les modèles
"""

import asyncio
import logging
from typing import Dict

from app.services.story_service import StoryService
from app.services.narrative_service import NarrativeService
from app.services.image_service import ImageService
from app.services.memory_service import MemoryService


# Configuration du logging pour le démarrage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIServicesManager:
    """
    Gestionnaire centralisé des services IA
    
    Cette classe coordonne le chargement et l'état de tous les services IA :
    - Narrative Service (génération de texte)
    - Image Service (génération d'images)
    - Memory Service (gestion de la mémoire)
    - Story Service (orchestration générale)
    """
    
    def __init__(self):
        """
        Initialise le gestionnaire de services IA
        """
        self.services = {
            "narrative": NarrativeService(),
            "image": ImageService(),
            "memory": MemoryService(),
            "story": StoryService()
        }
        
        self.services_status: Dict[str, bool] = {
            "narrative": False,
            "image": False,
            "memory": True,  # MemoryService ne nécessite pas de chargement
            "story": False
        }
        
        self.initialization_completed = False
    
    async def initialize_all_services(self) -> Dict[str, bool]:
        """
        Initialise tous les services IA de manière asynchrone
        
        Cette méthode :
        1. Lance le chargement des modèles en parallèle quand possible
        2. Gère les erreurs individuellement par service
        3. Continue même si certains services échouent
        4. Retourne un rapport d'état détaillé
        
        Returns:
            Dict[str, bool]: État de chargement de chaque service
        """
        logger.info("🚀 Démarrage de l'initialisation des services IA...")
        
        # Chargement des services qui nécessitent des modèles
        tasks = []
        
        # Narrative Service (modèle de texte)
        tasks.append(self._load_narrative_service())
        
        # Image Service (modèle d'images)
        tasks.append(self._load_image_service())
        
        # Story Service (orchestration - nécessite les autres services)
        # On l'initialise après les autres
        
        # Exécution en parallèle des chargements
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traitement des résultats
        narrative_ok = not isinstance(results[0], Exception) and results[0]
        image_ok = not isinstance(results[1], Exception) and results[1]
        self.services_status["narrative"] = narrative_ok
        self.services_status["image"] = image_ok
        
        # Initialisation du Story Service une fois les autres chargés
        try:
            story_loaded = await self.services["story"].load_ai_models()
            self.services_status["story"] = story_loaded
        except Exception as e:
            logger.error(f"Erreur chargement Story Service: {str(e)}")
            self.services_status["story"] = False
        
        # Rapport final
        self.initialization_completed = True
        self._log_initialization_report()
        
        return self.services_status
    
    async def _load_narrative_service(self) -> bool:
        """
        Charge le service de génération narrative
        
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            logger.info("📝 Chargement du modèle de génération narrative...")
            success = await self.services["narrative"].load_model()
            
            if success:
                logger.info("✅ Modèle de génération narrative chargé")
            else:
                logger.warning("⚠️ Échec modèle narratif - mode fallback activé")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement narratif: {str(e)}")
            return False
    
    async def _load_image_service(self) -> bool:
        """
        Charge le service de génération d'images
        
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            logger.info("🎨 Chargement du modèle de génération d'images...")
            success = await self.services["image"].load_model()
            
            if success:
                logger.info("✅ Modèle de génération d'images chargé")
            else:
                logger.warning("⚠️ Échec modèle images - génération désactivée")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement images: {str(e)}")
            return False
    
    def _log_initialization_report(self) -> None:
        """
        Affiche un rapport détaillé de l'initialisation
        """
        logger.info("\n" + "="*60)
        logger.info("📊 RAPPORT D'INITIALISATION DES SERVICES IA")
        logger.info("="*60)
        
        for service_name, status in self.services_status.items():
            status_icon = "✅" if status else "❌"
            status_text = "ACTIF" if status else "INACTIF"
            service_display = service_name.upper()
            logger.info(f"{status_icon} {service_display:<12} : {status_text}")
        
        # Résumé global
        active_count = sum(self.services_status.values())
        total_count = len(self.services_status)
        
        logger.info("-"*60)
        logger.info(f"📈 Services actifs: {active_count}/{total_count}")
        
        if active_count == total_count:
            logger.info("🎉 Tous les services IA sont opérationnels!")
        elif active_count > 0:
            logger.info("⚡ Application prête avec services partiels")
        else:
            logger.warning("⚠️ Aucun service IA disponible - mode dégradé")
        
        logger.info("="*60 + "\n")
    
    def get_services_status(self) -> Dict[str, bool]:
        """
        Retourne l'état actuel de tous les services
        
        Returns:
            Dict[str, bool]: État de chaque service
        """
        return self.services_status.copy()
    
    def is_service_ready(self, service_name: str) -> bool:
        """
        Vérifie si un service spécifique est prêt
        
        Args:
            service_name: Nom du service à vérifier
            
        Returns:
            bool: True si le service est prêt
        """
        return self.services_status.get(service_name, False)
    
    def get_ready_services_count(self) -> tuple[int, int]:
        """
        Retourne le nombre de services prêts / total
        
        Returns:
            tuple[int, int]: (services_prêts, total_services)
        """
        ready = sum(self.services_status.values())
        total = len(self.services_status)
        return ready, total


# Instance globale du gestionnaire de services
ai_services_manager = AIServicesManager()


async def initialize_ai_services() -> Dict[str, bool]:
    """
    Fonction publique pour initialiser tous les services IA
    
    Cette fonction est appelée au démarrage de l'application FastAPI
    pour charger tous les modèles nécessaires.
    
    Returns:
        Dict[str, bool]: État de chargement de chaque service
    """
    return await ai_services_manager.initialize_all_services()


def get_ai_services_status() -> Dict[str, bool]:
    """
    Retourne l'état actuel des services IA
    
    Returns:
        Dict[str, bool]: État de chaque service
    """
    return ai_services_manager.get_services_status()


def is_ai_service_ready(service_name: str) -> bool:
    """
    Vérifie si un service IA spécifique est prêt
    
    Args:
        service_name: Nom du service ("narrative", "image", "memory", "story")
        
    Returns:
        bool: True si le service est opérationnel
    """
    return ai_services_manager.is_service_ready(service_name)