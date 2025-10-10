"""
Service de génération d'images pour les scènes narratives

Ce service gère l'intégration avec les modèles de génération d'images pour :
- Créer des illustrations automatiques pour chaque scène
- Adapter le style visuel selon le genre d'histoire
- Optimiser les prompts visuels depuis le texte narratif
- Gérer le cache et stockage des images générées

Architecture :
- Support pour Stable Diffusion XL (local)
- Support pour APIs externes (OpenAI DALL-E, Midjourney)
- Système de prompts visuels optimisés
- Cache intelligent pour éviter les régénérations
"""

import torch
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from PIL import Image
import requests
import base64
import io
import os
import hashlib
from typing import Optional, Dict, Tuple
import asyncio
import logging
from datetime import datetime

from app.models.schemas import StoryGenre, ImageGenerationRequest, ImageResponse
from app.config import settings


class ImageService:
    """
    Service principal pour la génération d'images narratives
    
    Ce service encapsule toute la logique de génération visuelle :
    - Chargement et gestion des modèles de diffusion
    - Construction de prompts visuels optimisés
    - Génération d'images avec contrôle de qualité
    - Gestion du cache et du stockage
    """
    
    def __init__(self):
        """
        Initialise le service avec les modèles de génération d'images
        """
        # Configuration du logging
        self.logger = logging.getLogger(__name__)

        # Répertoire cache des modèles
        self.model_cache_dir = settings.MODEL_CACHE_PATH
        os.makedirs(self.model_cache_dir, exist_ok=True)

        self._patch_clip_offload_kwarg()

        # Variables pour le pipeline Stable Diffusion
        self.pipeline = None
        self.device, self.device_label = self._select_device(settings.IMAGE_MODEL_DEVICE)
        self.logger.info("Image generation model will use device: %s", self.device_label)
        if self.device.type == "cuda":
            try:
                device_name = torch.cuda.get_device_name(self.device.index or 0)
                self.logger.info("CUDA device detected: %s", device_name)
            except Exception:
                pass
        self.model_loaded = False
        
        # Configuration de génération par défaut
        self.generation_config = {
            "num_inference_steps": 30,  # Nombre d'étapes de débruitage
            "guidance_scale": 7.5,      # Force du guidage par le prompt
            "width": 1024,              # Largeur de l'image
            "height": 1024,             # Hauteur de l'image
            "num_images_per_prompt": 1, # Nombre d'images à générer
        }
        
        # Templates de style par genre
        self.genre_style_templates = self._load_genre_styles()
        
        # Cache des images générées (hash prompt -> chemin fichier)
        self._image_cache: Dict[str, str] = {}
        
        # Répertoire de stockage des images
        self.images_path = settings.IMAGES_PATH
        self._ensure_images_directory()
    
    def _load_genre_styles(self) -> Dict[StoryGenre, Dict[str, str]]:
        """
        Charge les templates de style visuel par genre
        
        Ces templates définissent l'esthétique visuelle de chaque genre
        avec des modificateurs de prompt et des styles artistiques
        
        Returns:
            Dict avec les styles visuels par genre
        """
        return {
            StoryGenre.FANTASY: {
                "style_suffix": ", fantasy art, magical atmosphere, detailed, epic, cinematic lighting, vibrant colors",
                "negative_prompt": "modern items, technology, cars, planes, smartphones, contemporary clothing",
                "artist_style": "in the style of Frank Frazetta and Boris Vallejo"
            },
            
            StoryGenre.SCIENCE_FICTION: {
                "style_suffix": ", sci-fi art, futuristic, cyberpunk aesthetic, neon lights, high-tech, detailed",
                "negative_prompt": "medieval, fantasy elements, magic, dragons, swords",
                "artist_style": "in the style of Syd Mead and H.R. Giger"
            },
            
            StoryGenre.HORROR: {
                "style_suffix": ", dark horror art, ominous atmosphere, gothic, eerie, shadows, dramatic lighting",
                "negative_prompt": "bright colors, cheerful, cartoonish, cute, happy",
                "artist_style": "in the style of H.P. Lovecraft illustrations and gothic art"
            },
            
            StoryGenre.MYSTERY: {
                "style_suffix": ", noir mystery art, detective aesthetic, moody lighting, film noir, atmospheric",
                "negative_prompt": "bright colors, fantasy elements, futuristic items",
                "artist_style": "in the style of film noir and detective novel covers"
            },
            
            StoryGenre.ADVENTURE: {
                "style_suffix": ", adventure art, dynamic action, exploration theme, cinematic, detailed landscape",
                "negative_prompt": "static pose, boring composition, low quality",
                "artist_style": "in the style of adventure movie posters and National Geographic"
            },
            
            StoryGenre.ROMANCE: {
                "style_suffix": ", romantic art, soft lighting, emotional atmosphere, warm colors, intimate",
                "negative_prompt": "dark themes, violence, horror elements, cold colors",
                "artist_style": "in the style of romantic novel covers and impressionist paintings"
            }
        }
    
    def _ensure_images_directory(self) -> None:
        """
        Crée le répertoire de stockage des images s'il n'existe pas
        """
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(os.path.join(self.images_path, "generated"), exist_ok=True)
        os.makedirs(os.path.join(self.images_path, "cache"), exist_ok=True)
    
    async def load_model(self) -> bool:
        """
        Charge le pipeline Stable Diffusion XL en mémoire
        
        Cette méthode initialise le modèle de génération d'images :
        - Chargement du pipeline depuis Hugging Face
        - Configuration du device (CPU/GPU)
        - Optimisations mémoire pour l'inférence
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            self.logger.info(f"Chargement du modèle d'images {settings.IMAGE_MODEL_NAME}...")
            
            # Chargement du pipeline Stable Diffusion XL
            pipeline_kwargs = {
                "torch_dtype": self._get_model_dtype(),
                "use_safetensors": True,
                "low_cpu_mem_usage": False,
                "device_map": None,
            }

            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                settings.IMAGE_MODEL_NAME,
                cache_dir=self.model_cache_dir,
                **pipeline_kwargs
            )
            
            # Déplacement vers le device approprié
            try:
                self.pipeline.to(self.device)
            except RuntimeError as runtime_error:
                if self.device.type == "cuda":
                    self.logger.warning(
                        "Impossible de charger le pipeline sur CUDA (%s). Repli sur CPU.",
                        runtime_error
                    )
                    self.device, self.device_label = torch.device("cpu"), "cpu"
                    self.pipeline.to(torch_dtype=torch.float32)
                    self.pipeline.to(self.device)
                    self.logger.info("Image pipeline fallback device: %s", self.device_label)
                elif self.device.type == "mps":
                    self.logger.warning(
                        "Impossible de charger le pipeline sur MPS (%s). Repli sur CPU.",
                        runtime_error
                    )
                    self.device, self.device_label = torch.device("cpu"), "cpu"
                    self.pipeline.to(torch_dtype=torch.float32)
                    self.pipeline.to(self.device)
                    self.logger.info("Image pipeline fallback device: %s", self.device_label)
                else:
                    raise
            
            # Optimisations mémoire si GPU
            if self.device.type == "cuda":
                self.pipeline.enable_memory_efficient_attention()
                self.pipeline.enable_vae_slicing()
            
            self.model_loaded = True
            self.logger.info("Modèle d'images chargé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle d'images: {str(e)}")
            self.model_loaded = False
            return False

    def _patch_clip_offload_kwarg(self) -> None:
        """Empêche les kwargs inconnus de casser l'init des modèles CLIP."""
        if getattr(self, "_clip_patched", False):
            return

        def _wrap_init(cls):
            original_init = cls.__init__
            if getattr(original_init, "_offload_patch", False):
                return

            def patched_init(self, config, *args, **kwargs):
                kwargs.pop("offload_state_dict", None)
                return original_init(self, config, *args, **kwargs)

            patched_init._offload_patch = True
            cls.__init__ = patched_init

        _wrap_init(CLIPTextModel)
        _wrap_init(CLIPTextModelWithProjection)
        self._clip_patched = True

    def _select_device(self, configured_device: Optional[str]) -> Tuple[torch.device, str]:
        """Sélectionne le device optimal pour le pipeline d'images."""
        available_cuda = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        available_mps = bool(mps_backend and mps_backend.is_available())

        normalized = (configured_device or "auto").strip().lower()

        if normalized.startswith("cuda"):
            if available_cuda:
                device_str = normalized if ":" in normalized else "cuda"
                return torch.device(device_str), device_str
            self.logger.warning("CUDA demandé pour les images mais indisponible, utilisation du CPU.")
            return torch.device("cpu"), "cpu"

        if normalized == "mps":
            if available_mps:
                return torch.device("mps"), "mps"
            self.logger.warning("MPS demandé pour les images mais indisponible, utilisation du CPU.")
            return torch.device("cpu"), "cpu"

        if normalized == "cpu":
            return torch.device("cpu"), "cpu"

        if normalized not in {"auto", ""}:
            self.logger.warning(
                "Device '%s' inconnu pour le pipeline images, détection automatique activée.",
                configured_device
            )

        if available_cuda:
            return torch.device("cuda"), "cuda"
        if available_mps:
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    def _get_model_dtype(self) -> torch.dtype:
        """Détermine le dtype optimal en fonction du device."""
        if self.device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
    
    async def generate_scene_image(
        self, 
        story_id: str,
        scene_number: int,
        narrative_text: str, 
        genre: StoryGenre
    ) -> Optional[ImageResponse]:
        """
        Génère une image pour une scène narrative spécifique
        
        Cette méthode :
        1. Extrait les éléments visuels du texte narratif
        2. Construit un prompt optimisé selon le genre
        3. Vérifie le cache pour éviter les régénérations
        4. Génère l'image avec le style approprié
        5. Sauvegarde et retourne les métadonnées
        
        Args:
            story_id: ID de l'histoire
            scene_number: Numéro de la scène
            narrative_text: Texte narratif de la scène
            genre: Genre de l'histoire pour le style
            
        Returns:
            ImageResponse: Métadonnées de l'image générée ou None si échec
        """
        try:
            # Construction du prompt visuel optimisé
            visual_prompt = await self._build_visual_prompt(narrative_text, genre)
            
            # Vérification du cache
            cache_key = self._get_cache_key(visual_prompt, genre)
            cached_image = self._get_cached_image(cache_key)
            
            if cached_image:
                self.logger.info(f"Image trouvée en cache pour la scène {scene_number}")
                return ImageResponse(
                    image_url=cached_image,
                    prompt_used=visual_prompt,
                    generation_time=0.0,
                    story_id=story_id
                )
            
            # Génération de l'image
            start_time = datetime.now()
            
            if self.model_loaded:
                image_path = await self._generate_with_local_model(
                    visual_prompt, genre, story_id, scene_number
                )
            else:
                # Fallback vers API externe si disponible
                image_path = await self._generate_with_external_api(
                    visual_prompt, genre, story_id, scene_number
                )
            
            if not image_path:
                return None
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Mise en cache
            self._cache_image(cache_key, image_path)
            
            return ImageResponse(
                image_url=image_path,
                prompt_used=visual_prompt,
                generation_time=generation_time,
                story_id=story_id
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'image: {str(e)}")
            return None
    
    async def _build_visual_prompt(self, narrative_text: str, genre: StoryGenre) -> str:
        """
        Construit un prompt visuel optimisé depuis le texte narratif
        
        Cette méthode analyse le texte narratif pour extraire :
        - Les éléments visuels principaux (personnages, objets, décors)
        - L'ambiance et l'atmosphère
        - Les actions importantes à représenter
        
        Args:
            narrative_text: Texte narratif à analyser
            genre: Genre pour adapter le style
            
        Returns:
            str: Prompt visuel optimisé
        """
        # Extraction des mots-clés visuels importants
        visual_keywords = self._extract_visual_keywords(narrative_text)
        
        # Construction du prompt de base
        base_prompt = self._build_base_visual_description(narrative_text, visual_keywords)
        
        # Ajout du style selon le genre
        genre_style = self.genre_style_templates[genre]
        
        # Prompt final optimisé
        final_prompt = f"{base_prompt}{genre_style['style_suffix']}, {genre_style['artist_style']}"
        
        return final_prompt
    
    def _extract_visual_keywords(self, text: str) -> list[str]:
        """
        Extrait les mots-clés visuels importants du texte narratif
        
        Args:
            text: Texte narratif à analyser
            
        Returns:
            List[str]: Mots-clés visuels extraits
        """
        # Dictionnaire de mots-clés visuels par catégorie
        visual_categories = {
            "characters": ["personnage", "héros", "homme", "femme", "enfant", "guerrier", "mage", "roi", "reine"],
            "creatures": ["dragon", "loup", "cheval", "oiseau", "monstre", "démon", "ange", "esprit"],
            "objects": ["épée", "livre", "cristal", "trésor", "clé", "carte", "potion", "armure"],
            "places": ["forêt", "château", "montagne", "rivière", "village", "ville", "cave", "temple"],
            "atmosphere": ["sombre", "lumineux", "mystérieux", "paisible", "dangereux", "magique", "ancien"]
        }
        
        found_keywords = []
        text_lower = text.lower()
        
        # Recherche des mots-clés dans le texte
        for category, keywords in visual_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        return found_keywords[:10]  # Limiter à 10 mots-clés maximum
    
    def _build_base_visual_description(self, text: str, keywords: list[str]) -> str:
        """
        Construit la description visuelle de base
        
        Args:
            text: Texte narratif complet
            keywords: Mots-clés visuels extraits
            
        Returns:
            str: Description visuelle de base
        """
        # Si le texte est court, l'utiliser directement comme base
        if len(text) < 200:
            base = text[:150]
        else:
            # Prendre la première partie du texte pour le contexte principal
            sentences = text.split('.')
            base = '. '.join(sentences[:2])
        
        # Ajouter les mots-clés importants si pertinents
        if keywords:
            key_elements = ', '.join(keywords[:5])
            base = f"{base}, featuring {key_elements}"
        
        return base
    
    async def _generate_with_local_model(
        self, 
        prompt: str, 
        genre: StoryGenre, 
        story_id: str, 
        scene_number: int
    ) -> Optional[str]:
        """
        Génère une image avec le modèle local (Stable Diffusion XL)
        
        Args:
            prompt: Prompt visuel optimisé
            genre: Genre pour les paramètres de style
            story_id: ID de l'histoire
            scene_number: Numéro de la scène
            
        Returns:
            str: Chemin vers l'image générée ou None si échec
        """
        try:
            genre_style = self.genre_style_templates[genre]
            
            # Génération avec le pipeline
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=genre_style["negative_prompt"],
                    **self.generation_config
                )
            
            # Récupération de la première image
            image = result.images[0]
            
            # Sauvegarde de l'image
            filename = f"{story_id}_scene_{scene_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image_path = os.path.join(self.images_path, "generated", filename)
            
            image.save(image_path, format="PNG", quality=95)
            
            # Retour du chemin relatif pour l'URL
            return f"/images/generated/{filename}"
            
        except Exception as e:
            self.logger.error(f"Erreur génération locale: {str(e)}")
            return None
    
    async def _generate_with_external_api(
        self, 
        prompt: str, 
        genre: StoryGenre, 
        story_id: str, 
        scene_number: int
    ) -> Optional[str]:
        """
        Génère une image avec une API externe (OpenAI DALL-E par exemple)
        
        Args:
            prompt: Prompt visuel optimisé
            genre: Genre pour les paramètres de style
            story_id: ID de l'histoire
            scene_number: Numéro de la scène
            
        Returns:
            str: Chemin vers l'image générée ou None si échec
        """
        try:
            # Vérification de la disponibilité de l'API OpenAI
            if not settings.OPENAI_API_KEY:
                self.logger.warning("Clé API OpenAI non configurée")
                return None
            
            # TODO: Implémenter l'appel à l'API OpenAI DALL-E
            # Pour l'instant, retourner None (sera implémenté selon les besoins)
            self.logger.info("Génération via API externe non encore implémentée")
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur génération API externe: {str(e)}")
            return None
    
    def _get_cache_key(self, prompt: str, genre: StoryGenre) -> str:
        """
        Génère une clé de cache unique pour un prompt et genre donnés
        
        Args:
            prompt: Prompt visuel
            genre: Genre de l'histoire
            
        Returns:
            str: Clé de cache (hash MD5)
        """
        cache_string = f"{prompt}_{genre.value}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_image(self, cache_key: str) -> Optional[str]:
        """
        Récupère une image depuis le cache si elle existe
        
        Args:
            cache_key: Clé de cache à rechercher
            
        Returns:
            str: Chemin vers l'image en cache ou None
        """
        if cache_key in self._image_cache:
            cached_path = self._image_cache[cache_key]
            # Vérifier que le fichier existe toujours
            if os.path.exists(os.path.join(self.images_path, "cache", cached_path)):
                return f"/images/cache/{cached_path}"
        
        return None
    
    def _cache_image(self, cache_key: str, image_path: str) -> None:
        """
        Met une image en cache pour réutilisation future
        
        Args:
            cache_key: Clé de cache unique
            image_path: Chemin vers l'image à mettre en cache
        """
        try:
            # Copie de l'image vers le répertoire de cache
            source_path = os.path.join(self.images_path, image_path.lstrip('/images/'))
            cache_filename = f"cached_{cache_key}.png"
            cache_path = os.path.join(self.images_path, "cache", cache_filename)
            
            if os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, cache_path)
                self._image_cache[cache_key] = cache_filename
                
        except Exception as e:
            self.logger.error(f"Erreur mise en cache: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """
        Vérifie si le modèle est chargé et prêt
        
        Returns:
            bool: True si le modèle est prêt à l'usage
        """
        return self.model_loaded and self.pipeline is not None
    
    async def generate_image_standalone(self, request: ImageGenerationRequest) -> Optional[ImageResponse]:
        """
        Génère une image de manière autonome (endpoint dédié)
        
        Args:
            request: Requête de génération d'image
            
        Returns:
            ImageResponse: Réponse avec l'image générée
        """
        try:
            # Utiliser le prompt directement tel que fourni
            visual_prompt = f"{request.prompt}, {request.style}"
            
            # Détection du genre depuis le style ou utiliser adventure par défaut
            genre = StoryGenre.ADVENTURE  # Défaut générique
            
            # Génération de l'image
            start_time = datetime.now()
            
            if self.model_loaded:
                image_path = await self._generate_with_local_model(
                    visual_prompt, genre, request.story_id, request.scene_number
                )
            else:
                image_path = await self._generate_with_external_api(
                    visual_prompt, genre, request.story_id, request.scene_number
                )
            
            if not image_path:
                return None
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return ImageResponse(
                image_url=image_path,
                prompt_used=visual_prompt,
                generation_time=generation_time,
                story_id=request.story_id
            )
            
        except Exception as e:
            self.logger.error(f"Erreur génération standalone: {str(e)}")
            return None