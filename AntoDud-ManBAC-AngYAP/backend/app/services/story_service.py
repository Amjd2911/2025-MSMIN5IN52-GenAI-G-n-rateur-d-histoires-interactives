"""
Service principal pour la gestion des histoires

Ce service contient toute la logique métier pour :
- Créer de nouvelles histoires
- Gérer la progression des histoires
- Maintenir la cohérence narrative
- Orchestrer les appels aux services IA

Architecture du service :
- Méthodes publiques : API pour les contrôleurs
- Méthodes privées : Logique interne et utilitaires
- Gestion d'erreurs : Try/catch avec logging approprié
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import asyncio

from app.models.schemas import (
    Story, Scene, StoryMemory, UserAction, Character, Location,
    StoryGenre, StoryState, StoryCreateRequest, StoryContinueRequest
)
from app.config import settings
from app.services.narrative_service import NarrativeService
from app.services.image_service import ImageService
from app.services.memory_service import MemoryService


class StoryService:
    """
    Service principal pour la gestion des histoires interactives
    
    Ce service orchestre toute la logique métier liée aux histoires :
    - Création et initialisation des histoires
    - Gestion de la progression narrative
    - Maintien de la mémoire contextuelle
    - Coordination avec les services IA (texte et image)
    """
    
    def __init__(self):
        """
        Initialise le service avec les dépendances nécessaires
        """
        # Répertoires de stockage (créés automatiquement si inexistants)
        self.stories_path = settings.STORIES_PATH
        self.ensure_directories_exist()
        
        # Cache en mémoire pour les histoires actives (évite les I/O répétés)
        self._story_cache: Dict[str, Story] = {}
        
        # Templates de prompts par genre (utilisés pour l'initialisation)
        self._genre_templates = self._load_genre_templates()
        
        # Services IA intégrés (Phase 2 & 3)
        self.narrative_service = NarrativeService()
        self.image_service = ImageService()
        self.memory_service = MemoryService()
        
        # Flag pour savoir si les modèles sont chargés
        self._models_loaded = False
    
    def ensure_directories_exist(self) -> None:
        """
        Crée les répertoires de stockage s'ils n'existent pas
        Méthode utilitaire appelée à l'initialisation
        """
        os.makedirs(self.stories_path, exist_ok=True)
        os.makedirs(os.path.join(self.stories_path, "active"), exist_ok=True)
        os.makedirs(os.path.join(self.stories_path, "archived"), exist_ok=True)
    
    def _load_genre_templates(self) -> Dict[StoryGenre, Dict[str, str]]:
        """
        Charge les templates de prompts spécifiques à chaque genre
        
        Returns:
            Dict contenant les prompts d'initialisation par genre
        """
        # Pour l'instant, templates simples - à terme, charger depuis des fichiers
        return {
            StoryGenre.FANTASY: {
                "system_prompt": "Tu es un narrateur expert en fantasy. Crée des histoires riches en magie, créatures fantastiques et aventures épiques.",
                "init_prompt": "Commence une nouvelle aventure fantasy. Le joueur incarne un héros dans un monde magique.",
                "style": "fantasy art, detailed, magical atmosphere"
            },
            StoryGenre.SCIENCE_FICTION: {
                "system_prompt": "Tu es un narrateur spécialisé en science-fiction. Explore les technologies futuristes et les concepts scientifiques.",
                "init_prompt": "Démarre une histoire de science-fiction dans un futur technologique avancé.",
                "style": "sci-fi art, futuristic, cyberpunk aesthetic"
            },
            StoryGenre.HORROR: {
                "system_prompt": "Tu es un maître de l'horreur. Crée une atmosphère oppressante et des situations terrifiantes.",
                "init_prompt": "Plonge le joueur dans une histoire d'horreur pleine de suspense et de frissons.",
                "style": "dark horror art, ominous atmosphere, gothic"
            },
            StoryGenre.MYSTERY: {
                "system_prompt": "Tu es un narrateur de mystères. Tisses des intrigues complexes avec des indices et des révélations.",
                "init_prompt": "Lance une enquête mystérieuse que le joueur devra résoudre étape par étape.",
                "style": "noir mystery art, detective aesthetic, moody lighting"
            },
            StoryGenre.ADVENTURE: {
                "system_prompt": "Tu es un guide d'aventures. Crée des quêtes palpitantes et des défis excitants.",
                "init_prompt": "Embarque le joueur dans une aventure pleine d'action et de découvertes.",
                "style": "adventure art, dynamic action, exploration theme"
            },
            StoryGenre.ROMANCE: {
                "system_prompt": "Tu es un conteur romantique. Développe des relations touchantes et des histoires d'amour.",
                "init_prompt": "Raconte une belle histoire d'amour avec ses joies et ses obstacles.",
                "style": "romantic art, soft lighting, emotional atmosphere"
            }
        }
    
    async def create_story(self, request: StoryCreateRequest) -> Story:
        """
        Crée une nouvelle histoire avec le genre spécifié
        
        Process:
        1. Générer un ID unique pour l'histoire
        2. Initialiser la mémoire contextuelle
        3. Créer la première scène avec le contexte initial
        4. Sauvegarder l'histoire
        5. Ajouter au cache
        
        Args:
            request: Requête contenant le genre et éventuellement un prompt initial
            
        Returns:
            Story: L'histoire créée et initialisée
            
        Raises:
            Exception: En cas d'erreur lors de la création
        """
        try:
            # Génération d'un ID unique pour l'histoire
            story_id = str(uuid.uuid4())
            
            # Initialisation de la mémoire contextuelle vide
            initial_memory = StoryMemory(
                story_id=story_id,
                current_summary="Histoire en cours d'initialisation...",
                key_events=[],
                characters={},
                locations={},
                world_state={},
                plot_threads=[],
                user_actions_history=[]
            )
            
            # Création de l'objet Story principal
            story = Story(
                story_id=story_id,
                genre=request.genre,
                initial_prompt=request.initial_prompt,
                state=StoryState.CREATED,
                current_scene_number=0,
                total_scenes=0,
                scenes=[],
                memory=initial_memory
            )
            
            # Génération de la première scène d'introduction
            # Cette scène établit le contexte sans action utilisateur
            intro_scene = await self._generate_intro_scene(story)
            story.scenes.append(intro_scene)
            story.total_scenes = 1
            story.current_scene_number = 1
            story.state = StoryState.IN_PROGRESS
            
            # Mise à jour de la mémoire avec les éléments de la première scène
            await self._update_memory_from_scene(story, intro_scene)
            
            # Sauvegarde persistante
            await self._save_story(story)
            
            # Ajout au cache pour accès rapide
            self._story_cache[story_id] = story
            
            return story
            
        except Exception as e:
            # Log de l'erreur pour debugging
            print(f"Erreur lors de la création de l'histoire: {str(e)}")
            raise Exception(f"Impossible de créer l'histoire: {str(e)}")
    
    async def continue_story(self, story_id: str, request: StoryContinueRequest) -> Story:
        """
        Continue une histoire existante avec l'action du joueur
        
        Process:
        1. Charger l'histoire depuis le cache ou le stockage
        2. Valider que l'histoire peut continuer
        3. Enregistrer l'action du joueur
        4. Générer la réponse narrative
        5. Créer la nouvelle scène
        6. Mettre à jour la mémoire
        7. Sauvegarder les changements
        
        Args:
            story_id: ID unique de l'histoire
            request: Requête contenant l'action du joueur
            
        Returns:
            Story: L'histoire mise à jour avec la nouvelle scène
            
        Raises:
            Exception: Si l'histoire n'existe pas ou ne peut pas continuer
        """
        try:
            # Chargement de l'histoire (cache ou stockage)
            story = await self._load_story(story_id)
            
            if not story:
                raise Exception(f"Histoire {story_id} introuvable")
            
            if story.state not in [StoryState.IN_PROGRESS, StoryState.PAUSED]:
                raise Exception(f"L'histoire est dans l'état {story.state} et ne peut pas continuer")
            
            # Création de l'objet UserAction pour traçabilité
            user_action = UserAction(
                action_text=request.user_action,
                scene_number=story.current_scene_number + 1,
                timestamp=datetime.now()
            )
            
            # Génération de la nouvelle scène basée sur l'action
            new_scene = await self._generate_story_scene(story, user_action)
            
            # Mise à jour de l'histoire
            story.scenes.append(new_scene)
            story.total_scenes += 1
            story.current_scene_number += 1
            story.last_activity = datetime.now()
            story.updated_at = datetime.now()
            
            # Ajout de l'action à l'historique de la mémoire
            story.memory.user_actions_history.append(user_action)
            
            # Mise à jour de la mémoire contextuelle
            await self._update_memory_from_scene(story, new_scene)
            
            # Sauvegarde des changements
            await self._save_story(story)
            
            # Mise à jour du cache
            self._story_cache[story_id] = story
            
            return story
            
        except Exception as e:
            print(f"Erreur lors de la continuation de l'histoire {story_id}: {str(e)}")
            raise Exception(f"Impossible de continuer l'histoire: {str(e)}")
    
    async def get_story(self, story_id: str) -> Optional[Story]:
        """
        Récupère une histoire par son ID
        
        Args:
            story_id: ID unique de l'histoire
            
        Returns:
            Story ou None si introuvable
        """
        return await self._load_story(story_id)
    
    async def load_ai_models(self) -> bool:
        """
        Charge tous les modèles IA nécessaires
        
        Cette méthode doit être appelée au démarrage de l'application
        pour initialiser les services d'IA
        
        Returns:
            bool: True si tous les modèles sont chargés avec succès
        """
        try:
            # Chargement du modèle de génération de texte
            text_loaded = await self.narrative_service.load_model()
            
            # Chargement du modèle de génération d'images
            image_loaded = await self.image_service.load_model()
            
            self._models_loaded = text_loaded or image_loaded
            return self._models_loaded
            
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {str(e)}")
            return False

    async def _generate_intro_scene(self, story: Story) -> Scene:
        """
        Génère la scène d'introduction d'une histoire avec IA intégrée
        
        Cette méthode utilise maintenant les services IA pour :
        - Générer un texte narratif cohérent avec le genre
        - Proposer des actions adaptées au contexte
        - Créer une image illustrant la scène d'ouverture
        
        Args:
            story: L'histoire pour laquelle générer l'intro
            
        Returns:
            Scene: La scène d'introduction générée avec IA
        """
        try:
            # Génération du contenu narratif avec le NarrativeService
            if self.narrative_service.is_model_loaded():
                narrative_text, suggested_actions = await self.narrative_service.generate_intro_scene(story)
            else:
                # Fallback si l'IA n'est pas disponible
                narrative_text, suggested_actions = self._generate_fallback_intro(story)
            
            # Création de la scène de base
            scene = Scene(
                scene_number=1,
                story_id=story.story_id,
                narrative_text=narrative_text,
                suggested_actions=suggested_actions
            )
            
            # Génération de l'image de la scène
            if self.image_service.is_model_loaded():
                image_response = await self.image_service.generate_scene_image(
                    story_id=story.story_id,
                    scene_number=1,
                    narrative_text=narrative_text,
                    genre=story.genre
                )
                
                if image_response:
                    scene.image_url = image_response.image_url
                    scene.image_prompt = image_response.prompt_used
                    scene.image_generation_time = image_response.generation_time
            
            return scene
            
        except Exception as e:
            print(f"Erreur lors de la génération d'intro IA: {str(e)}")
            return self._generate_fallback_intro_scene(story)
    
    def _generate_fallback_intro(self, story: Story) -> Tuple[str, List[str]]:
        """
        Génère une introduction de secours si l'IA n'est pas disponible
        
        Args:
            story: L'histoire pour laquelle générer l'intro
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        genre_template = self._genre_templates.get(story.genre)
        
        narrative_text = f"Bienvenue dans cette aventure {story.genre.value}! " + \
                        "Vous vous réveillez dans un monde mystérieux. " + \
                        "Que souhaitez-vous faire ?"
        
        if story.initial_prompt:
            narrative_text += f" {story.initial_prompt}"
        
        suggested_actions = [
            "Explorer les environs",
            "Chercher des indices sur votre situation",
            "Appeler à l'aide"
        ]
        
        return narrative_text, suggested_actions
    
    def _generate_fallback_intro_scene(self, story: Story) -> Scene:
        """
        Génère une scène d'introduction de secours
        
        Args:
            story: L'histoire pour laquelle générer l'intro
            
        Returns:
            Scene: La scène d'introduction de base
        """
        narrative_text, suggested_actions = self._generate_fallback_intro(story)
        
        return Scene(
            scene_number=1,
            story_id=story.story_id,
            narrative_text=narrative_text,
            suggested_actions=suggested_actions,
            image_prompt=f"Scene introduction for {story.genre.value} story"
        )
    
    async def _generate_story_scene(self, story: Story, user_action: UserAction) -> Scene:
        """
        Génère une nouvelle scène basée sur l'action du joueur avec IA intégrée
        
        Cette méthode utilise maintenant tous les services IA :
        - NarrativeService pour générer le texte cohérent
        - ImageService pour créer l'illustration
        - MemoryService pour maintenir la cohérence (sera intégré après)
        
        Args:
            story: L'histoire en cours
            user_action: L'action effectuée par le joueur
            
        Returns:
            Scene: La nouvelle scène générée avec IA
        """
        try:
            scene_number = story.current_scene_number + 1
            
            # Génération du contenu narratif avec le NarrativeService
            if self.narrative_service.is_model_loaded():
                narrative_text, suggested_actions = await self.narrative_service.generate_scene_continuation(
                    story, user_action
                )
            else:
                # Fallback si l'IA n'est pas disponible
                narrative_text, suggested_actions = self._generate_fallback_continuation(
                    user_action
                )
            
            # Création de la scène de base
            scene = Scene(
                scene_number=scene_number,
                story_id=story.story_id,
                narrative_text=narrative_text,
                user_action=user_action,
                suggested_actions=suggested_actions
            )
            
            # Génération de l'image de la scène
            if self.image_service.is_model_loaded():
                image_response = await self.image_service.generate_scene_image(
                    story_id=story.story_id,
                    scene_number=scene_number,
                    narrative_text=narrative_text,
                    genre=story.genre
                )
                
                if image_response:
                    scene.image_url = image_response.image_url
                    scene.image_prompt = image_response.prompt_used
                    scene.image_generation_time = image_response.generation_time
            
            return scene
            
        except Exception as e:
            print(f"Erreur lors de la génération de scène IA: {str(e)}")
            return self._generate_fallback_scene(story, user_action)
    
    def _generate_fallback_continuation(self, user_action: UserAction) -> Tuple[str, List[str]]:
        """
        Génère une continuation de secours si l'IA n'est pas disponible
        
        Args:
            user_action: Action du joueur
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        narrative_text = f"Vous décidez de {user_action.action_text.lower()}. " + \
                        "L'histoire continue de se développer de manière inattendue..."
        
        suggested_actions = [
            "Continuer dans cette direction",
            "Changer d'approche",
            "Réfléchir à la situation"
        ]
        
        return narrative_text, suggested_actions
    
    def _generate_fallback_scene(self, story: Story, user_action: UserAction) -> Scene:
        """
        Génère une scène de secours
        
        Args:
            story: L'histoire en cours
            user_action: Action du joueur
            
        Returns:
            Scene: Scène de base
        """
        narrative_text, suggested_actions = self._generate_fallback_continuation(user_action)
        
        return Scene(
            scene_number=story.current_scene_number + 1,
            story_id=story.story_id,
            narrative_text=narrative_text,
            user_action=user_action,
            suggested_actions=suggested_actions,
            image_prompt=f"Scene {story.current_scene_number + 1} for {story.genre.value} story"
        )
    
    async def _update_memory_from_scene(self, story: Story, scene: Scene) -> None:
        """
        Met à jour la mémoire contextuelle avec le MemoryService intelligent
        
        Cette méthode utilise maintenant le MemoryService (Phase 3) pour :
        - Analyser intelligemment le contenu de la scène
        - Extraire automatiquement les entités (personnages, lieux, objets)
        - Maintenir la cohérence narrative globale
        - Optimiser la mémoire pour les performances
        
        Args:
            story: L'histoire à mettre à jour
            scene: La scène à analyser
        """
        try:
            # Utilisation du MemoryService intelligent pour l'analyse
            story.memory = await self.memory_service.analyze_and_update_memory(
                story, scene
            )
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour mémoire IA: {str(e)}")
            # Fallback vers la méthode basique
            self._update_memory_basic(story, scene)
    
    def _update_memory_basic(self, story: Story, scene: Scene) -> None:
        """
        Mise à jour basique de la mémoire (fallback)
        
        Args:
            story: L'histoire à mettre à jour
            scene: La scène à analyser
        """
        # Mise à jour basique comme avant
        story.memory.key_events.append(
            f"Scène {scene.scene_number}: {scene.narrative_text[:100]}..."
        )
        
        # Mise à jour du résumé (version simplifiée)
        if len(story.scenes) == 1:
            story.memory.current_summary = scene.narrative_text[:200] + "..."
        else:
            story.memory.current_summary += f" Puis: {scene.narrative_text[:100]}..."
        
        story.memory.last_updated = datetime.now()
        story.memory.memory_version += 1
    
    async def _save_story(self, story: Story) -> None:
        """
        Sauvegarde une histoire sur le disque
        
        Args:
            story: L'histoire à sauvegarder
        """
        file_path = os.path.join(self.stories_path, "active", f"{story.story_id}.json")
        
        # Conversion en dictionnaire pour sérialisation JSON
        story_data = story.model_dump()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, ensure_ascii=False, indent=2, default=str)
    
    async def _load_story(self, story_id: str) -> Optional[Story]:
        """
        Charge une histoire depuis le cache ou le disque
        
        Args:
            story_id: ID de l'histoire à charger
            
        Returns:
            Story ou None si introuvable
        """
        # Vérification du cache d'abord
        if story_id in self._story_cache:
            return self._story_cache[story_id]
        
        # Chargement depuis le disque
        file_path = os.path.join(self.stories_path, "active", f"{story_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            
            # Reconstruction de l'objet Story depuis le JSON
            story = Story(**story_data)
            
            # Ajout au cache
            self._story_cache[story_id] = story
            
            return story
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'histoire {story_id}: {str(e)}")
            return None