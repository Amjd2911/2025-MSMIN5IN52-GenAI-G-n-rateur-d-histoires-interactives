"""
Service de génération de texte narratif avec IA

Ce service gère l'intégration avec les modèles de langage pour :
- Générer du contenu narratif cohérent
- Maintenir le contexte sur de longues conversations
- Adapter le style selon le genre d'histoire
- Proposer des actions au joueur

Architecture :
- Intégration avec transformers/torch pour les modèles locaux
- Support pour APIs externes (OpenAI, etc.)
- Système de prompts modulaires par genre
- Cache des modèles chargés pour performance
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime

from app.models.schemas import Story, Scene, StoryGenre, UserAction
from app.config import settings


class NarrativeService:
    """
    Service principal pour la génération de texte narratif
    
    Ce service encapsule toute la logique de génération de contenu :
    - Chargement et gestion des modèles IA
    - Construction de prompts contextuels
    - Génération de texte avec contrôle de qualité
    - Extraction d'actions suggérées
    """
    
    def __init__(self):
        """
        Initialise le service avec le modèle de génération de texte
        """
        # Configuration du logging
        self.logger = logging.getLogger(__name__)

        # Répertoire cache des modèles
        self.model_cache_dir = settings.MODEL_CACHE_PATH
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Variables pour le modèle chargé
        self.tokenizer = None
        self.model = None
        self.device, self.device_label = self._select_device(settings.TEXT_MODEL_DEVICE)
        self.logger.info("Narrative model will use device: %s", self.device_label)
        if self.device.type == "cuda":
            try:
                device_name = torch.cuda.get_device_name(self.device.index or 0)
                self.logger.info("CUDA device detected: %s", device_name)
            except Exception:
                pass
        self.model_loaded = False
        
        # Configuration de génération par défaut
        self.generation_config = {
            "max_new_tokens": 300,  # Longueur maximum de génération
            "temperature": 0.8,     # Créativité (0.1=conservateur, 1.0=créatif)
            "top_p": 0.9,          # Nucleus sampling
            "top_k": 50,           # Top-k sampling
            "do_sample": True,     # Activation du sampling
            "pad_token_id": None,  # Sera défini après chargement du tokenizer
        }
        
        # Prompts système par genre (templates avancés)
        self.genre_system_prompts = self._load_genre_system_prompts()
        
        # Cache des contextes récents pour optimisation
        self._context_cache: Dict[str, str] = {}
    
    def _load_genre_system_prompts(self) -> Dict[StoryGenre, str]:
        """
        Charge les prompts système spécialisés par genre
        
        Ces prompts définissent le comportement de base de l'IA
        selon le type d'histoire à générer
        
        Returns:
            Dict avec les prompts système par genre
        """
        return {
            StoryGenre.FANTASY: """Tu es un narrateur expert en fantasy. 
            Crée des histoires riches en magie, créatures fantastiques et aventures épiques.
            Utilise un vocabulaire évocateur et des descriptions visuelles détaillées.
            Maintiens la cohérence du monde magique et de ses règles.
            Propose des actions qui explorent la magie et l'aventure.""",
            
            StoryGenre.SCIENCE_FICTION: """Tu es un narrateur spécialisé en science-fiction.
            Explore les technologies futuristes, les concepts scientifiques et l'espace.
            Intègre des éléments technologiques crédibles et des enjeux éthiques.
            Maintiens la cohérence scientifique dans ton univers.
            Propose des actions qui explorent la technologie et l'innovation.""",
            
            StoryGenre.HORROR: """Tu es un maître de l'horreur psychologique.
            Crée une atmosphère oppressante et des situations terrifiantes.
            Utilise la suggestion plutôt que la violence gratuite.
            Maintiens le suspense et l'angoisse progressivement.
            Propose des actions qui intensifient la tension.""",
            
            StoryGenre.MYSTERY: """Tu es un narrateur de mystères et d'enquêtes.
            Tisses des intrigues complexes avec des indices subtils.
            Maintiens l'équilibre entre révélations et mystère.
            Crée des personnages aux motivations complexes.
            Propose des actions qui font avancer l'enquête.""",
            
            StoryGenre.ADVENTURE: """Tu es un guide d'aventures palpitantes.
            Crée des quêtes excitantes avec des défis variés.
            Alterne entre action, exploration et découverte.
            Maintiens un rythme dynamique et engageant.
            Propose des actions qui poussent l'aventure vers l'avant.""",
            
            StoryGenre.ROMANCE: """Tu es un conteur d'histoires romantiques.
            Développe des relations touchantes et des émotions authentiques.
            Crée des personnages attachants avec de la profondeur.
            Maintiens l'équilibre entre tension romantique et développement.
            Propose des actions qui développent les relations."""
        }
    
    async def load_model(self) -> bool:
        """
        Charge le modèle de génération de texte en mémoire
        
        Cette méthode est appelée au démarrage de l'application
        pour initialiser le modèle IA. Elle gère :
        - Le chargement du tokenizer et du modèle
        - La configuration du device (CPU/GPU)
        - La gestion des erreurs de chargement
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            self.logger.info(f"Chargement du modèle {settings.TEXT_MODEL_NAME}...")
            
            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.TEXT_MODEL_NAME,
                trust_remote_code=True,
                cache_dir=self.model_cache_dir
            )
            
            # Configuration du pad_token si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Mise à jour de la configuration avec le bon pad_token_id
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.TEXT_MODEL_NAME,
                torch_dtype=self._get_model_dtype(),
                trust_remote_code=True,
                cache_dir=self.model_cache_dir
            )
            
            # Déplacement vers le device approprié
            try:
                self.model.to(self.device)
            except RuntimeError as runtime_error:
                if self.device.type == "cuda":
                    self.logger.warning(
                        "Impossible de charger le modèle sur CUDA (%s). Repli sur CPU.",
                        runtime_error
                    )
                    self.device, self.device_label = torch.device("cpu"), "cpu"
                    self.model = self.model.to(torch.float32)
                    self.model = self.model.to(self.device)
                    self.logger.info("Narrative model fallback device: %s", self.device_label)
                elif self.device.type == "mps":
                    self.logger.warning(
                        "Impossible de charger le modèle sur MPS (%s). Repli sur CPU.",
                        runtime_error
                    )
                    self.device, self.device_label = torch.device("cpu"), "cpu"
                    self.model = self.model.to(torch.float32)
                    self.model = self.model.to(self.device)
                    self.logger.info("Narrative model fallback device: %s", self.device_label)
                else:
                    raise
            
            # Mode évaluation pour l'inférence
            self.model.eval()
            
            self.model_loaded = True
            self.logger.info("Modèle de texte chargé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            self.model_loaded = False
            return False

    def _select_device(self, configured_device: Optional[str]) -> Tuple[torch.device, str]:
        """
        Sélectionne le device à utiliser pour le modèle

        Args:
            configured_device: Valeur provenant de la configuration

        Returns:
            Tuple[torch.device, str]: Device torch et libellé lisible
        """
        available_cuda = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        available_mps = bool(mps_backend and mps_backend.is_available())

        normalized = (configured_device or "auto").strip().lower()

        # Gestion des choix explicites
        if normalized.startswith("cuda"):
            if available_cuda:
                device_str = normalized if ":" in normalized else "cuda"
                return torch.device(device_str), device_str
            self.logger.warning("CUDA demandé mais indisponible, utilisation du CPU.")
            return torch.device("cpu"), "cpu"

        if normalized == "mps":
            if available_mps:
                return torch.device("mps"), "mps"
            self.logger.warning("MPS demandé mais indisponible, utilisation du CPU.")
            return torch.device("cpu"), "cpu"

        if normalized == "cpu":
            return torch.device("cpu"), "cpu"

        if normalized not in {"auto", ""}:
            self.logger.warning(
                "Device '%s' inconnu, activation de la détection automatique.",
                configured_device
            )

        # Détection automatique
        if available_cuda:
            return torch.device("cuda"), "cuda"
        if available_mps:
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    def _get_model_dtype(self) -> torch.dtype:
        """Détermine le type de tenseur optimal selon le device."""
        if self.device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
    
    async def generate_intro_scene(self, story: Story) -> Tuple[str, List[str]]:
        """
        Génère la scène d'introduction d'une nouvelle histoire
        
        Cette méthode crée le contexte initial basé sur :
        - Le genre choisi par l'utilisateur
        - Le prompt initial optionnel
        - Les conventions narratives du genre
        
        Args:
            story: L'histoire pour laquelle générer l'introduction
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        if not self.model_loaded:
            return self._fallback_intro_scene(story)
        
        try:
            # Construction du prompt d'introduction
            system_prompt = self.genre_system_prompts[story.genre]
            
            user_context = ""
            if story.initial_prompt:
                user_context = f"\nContexte initial fourni par l'utilisateur: {story.initial_prompt}"
            
            # Prompt complet pour l'introduction
            full_prompt = f"""<|system|>
{system_prompt}

Tu dois créer la scène d'ouverture d'une histoire {story.genre.value}.
Génère UNIQUEMENT le texte narratif de 2-3 paragraphes qui pose le décor et la situation initiale.
N'inclus pas d'actions du joueur dans ta réponse.
Termine par une situation qui appelle à l'action du joueur.{user_context}
<|user|>
Commence une nouvelle histoire {story.genre.value}.
<|assistant|>"""

            # Génération du texte
            narrative_text = await self._generate_text(full_prompt)
            
            # Génération des actions suggérées séparément
            suggested_actions = await self._generate_suggested_actions(
                story.genre, narrative_text, None
            )
            
            return narrative_text, suggested_actions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'intro: {str(e)}")
            return self._fallback_intro_scene(story)
    
    async def generate_scene_continuation(
        self, 
        story: Story, 
        user_action: UserAction
    ) -> Tuple[str, List[str]]:
        """
        Génère la continuation d'une histoire basée sur l'action du joueur
        
        Cette méthode utilise tout le contexte disponible :
        - Résumé de l'histoire depuis la mémoire
        - Dernières scènes pour le contexte immédiat
        - Action spécifique du joueur
        - Personnages et lieux connus
        
        Args:
            story: L'histoire en cours
            user_action: L'action effectuée par le joueur
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        if not self.model_loaded:
            return self._fallback_scene_continuation(user_action)
        
        try:
            # Construction du contexte complet
            context = self._build_story_context(story)
            
            # Prompt pour la continuation
            system_prompt = self.genre_system_prompts[story.genre]
            
            full_prompt = f"""<|system|>
{system_prompt}

CONTEXTE DE L'HISTOIRE:
{context}

Tu dois continuer l'histoire en réagissant à l'action du joueur.
Génère UNIQUEMENT le texte narratif de 2-3 paragraphes qui décrit les conséquences de l'action.
Maintiens la cohérence avec les éléments établis (personnages, lieux, événements).
Termine par une nouvelle situation qui appelle à l'action.
<|user|>
Le joueur décide de: {user_action.action_text}
<|assistant|>"""

            # Génération du texte narratif
            narrative_text = await self._generate_text(full_prompt)
            
            # Génération des actions suggérées
            suggested_actions = await self._generate_suggested_actions(
                story.genre, narrative_text, context
            )
            
            return narrative_text, suggested_actions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération: {str(e)}")
            return self._fallback_scene_continuation(user_action)
    
    def _build_story_context(self, story: Story) -> str:
        """
        Construit le contexte narratif complet pour la génération
        
        Cette méthode agrège toutes les informations pertinentes :
        - Résumé actuel de l'histoire
        - Personnages importants avec leurs traits
        - Lieux visités avec leurs descriptions
        - Dernières scènes pour le contexte immédiat
        
        Args:
            story: L'histoire dont construire le contexte
            
        Returns:
            str: Contexte formaté pour l'IA
        """
        context_parts = []
        
        # Résumé de l'histoire
        if story.memory.current_summary:
            context_parts.append(f"RÉSUMÉ: {story.memory.current_summary}")
        
        # Personnages connus
        if story.memory.characters:
            characters_info = []
            for name, character in story.memory.characters.items():
                char_info = f"{name}: {character.description}"
                if character.traits:
                    char_info += f" (Traits: {', '.join(character.traits)})"
                characters_info.append(char_info)
            
            context_parts.append("PERSONNAGES:\n" + "\n".join(characters_info))
        
        # Lieux connus
        if story.memory.locations:
            locations_info = []
            for name, location in story.memory.locations.items():
                loc_info = f"{name}: {location.description}"
                locations_info.append(loc_info)
            
            context_parts.append("LIEUX:\n" + "\n".join(locations_info))
        
        # Dernières scènes pour contexte immédiat (max 3)
        recent_scenes = story.scenes[-3:] if len(story.scenes) > 3 else story.scenes
        if recent_scenes:
            scenes_info = []
            for scene in recent_scenes:
                scene_info = f"Scène {scene.scene_number}: {scene.narrative_text[:200]}..."
                if scene.user_action:
                    scene_info += f"\nAction joueur: {scene.user_action.action_text}"
                scenes_info.append(scene_info)
            
            context_parts.append("SCÈNES RÉCENTES:\n" + "\n".join(scenes_info))
        
        return "\n\n".join(context_parts)
    
    async def _generate_text(self, prompt: str) -> str:
        """
        Génère du texte avec le modèle IA chargé
        
        Cette méthode encapsule l'appel au modèle avec :
        - Tokenisation du prompt
        - Génération avec les paramètres configurés
        - Détokenisation et nettoyage du résultat
        
        Args:
            prompt: Le prompt complet à envoyer au modèle
            
        Returns:
            str: Le texte généré et nettoyé
        """
        # Tokenisation du prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncate=True,
            max_length=settings.MAX_CONTEXT_LENGTH
        )
        inputs = inputs.to(self.device)
        
        # Génération avec le modèle
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **self.generation_config
            )
        
        # Extraction du texte généré (sans le prompt initial)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Nettoyage et post-traitement
        return self._clean_generated_text(generated_text)
    
    async def _generate_suggested_actions(
        self, 
        genre: StoryGenre, 
        narrative_text: str, 
        context: Optional[str] = None
    ) -> List[str]:
        """
        Génère des actions suggérées basées sur la situation narrative
        
        Cette méthode analyse le contexte actuel pour proposer
        des actions cohérentes et intéressantes au joueur
        
        Args:
            genre: Genre de l'histoire pour adapter les suggestions
            narrative_text: Texte de la scène actuelle
            context: Contexte optionnel de l'histoire
            
        Returns:
            List[str]: Liste d'actions suggérées (3-4 actions)
        """
        try:
            context_str = f"\nContexte: {context}" if context else ""
            
            prompt = f"""<|system|>
Tu es un assistant qui génère des suggestions d'actions pour un jeu narratif {genre.value}.
Analyse la situation actuelle et propose exactement 3 actions possibles.
Les actions doivent être:
- Cohérentes avec la situation
- Variées (différentes approches)
- Intéressantes narrativement
Formate ta réponse comme une liste numérotée simple.
<|user|>
Situation actuelle: {narrative_text[-200:]}...{context_str}

Propose 3 actions possibles pour le joueur:
<|assistant|>
1."""

            generated_text = await self._generate_text(prompt)
            
            # Extraction des actions depuis le texte généré
            actions = self._extract_actions_from_text("1." + generated_text)
            
            # Si l'extraction échoue, utiliser des actions génériques
            if not actions:
                return self._get_generic_actions(genre)
            
            return actions[:3]  # Limiter à 3 actions maximum
            
        except Exception as e:
            self.logger.error(f"Erreur génération actions: {str(e)}")
            return self._get_generic_actions(genre)
    
    def _extract_actions_from_text(self, text: str) -> List[str]:
        """
        Extrait les actions suggérées depuis le texte généré
        
        Args:
            text: Texte contenant les actions numérotées
            
        Returns:
            List[str]: Actions extraites et nettoyées
        """
        actions = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Chercher les lignes qui commencent par un numéro
            if line and (line[0].isdigit() or line.startswith('-')):
                # Nettoyer la ligne (enlever numéro et puces)
                action = line.lstrip('0123456789.-) ').strip()
                if action and len(action) > 10:  # Filtre les actions trop courtes
                    actions.append(action)
        
        return actions
    
    def _get_generic_actions(self, genre: StoryGenre) -> List[str]:
        """
        Retourne des actions génériques adaptées au genre
        
        Args:
            genre: Genre de l'histoire
            
        Returns:
            List[str]: Actions génériques par défaut
        """
        generic_actions = {
            StoryGenre.FANTASY: [
                "Explorer la zone avec prudence",
                "Utiliser vos capacités magiques",
                "Chercher des alliés ou des indices"
            ],
            StoryGenre.SCIENCE_FICTION: [
                "Analyser la situation avec vos instruments",
                "Contacter votre équipe ou base",
                "Explorer les technologies disponibles"
            ],
            StoryGenre.HORROR: [
                "Rester immobile et écouter",
                "Chercher une issue de secours",
                "Enquêter malgré le danger"
            ],
            StoryGenre.MYSTERY: [
                "Examiner les indices présents",
                "Interroger les témoins",
                "Formuler une hypothèse"
            ],
            StoryGenre.ADVENTURE: [
                "Prendre des risques calculés",
                "Chercher un chemin alternatif",
                "Utiliser votre équipement"
            ],
            StoryGenre.ROMANCE: [
                "Engager la conversation",
                "Observer et comprendre",
                "Prendre une initiative romantique"
            ]
        }
        
        return generic_actions.get(genre, [
            "Continuer avec prudence",
            "Changer d'approche",
            "Réfléchir à la situation"
        ])
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Nettoie le texte généré par l'IA
        
        Args:
            text: Texte brut généré
            
        Returns:
            str: Texte nettoyé et formaté
        """
        # Suppression des balises et artefacts
        text = text.replace('<|assistant|>', '').replace('<|user|>', '')
        text = text.replace('<|system|>', '').strip()
        
        # Suppression des répétitions de prompt
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('```'):
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        # Limitation de la longueur
        if len(result) > settings.MAX_STORY_LENGTH:
            result = result[:settings.MAX_STORY_LENGTH] + "..."
        
        return result.strip()
    
    def _fallback_intro_scene(self, story: Story) -> Tuple[str, List[str]]:
        """
        Génère une scène d'introduction de base si l'IA n'est pas disponible
        
        Args:
            story: L'histoire pour laquelle générer l'intro
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        genre_intros = {
            StoryGenre.FANTASY: "Vous vous réveillez dans une forêt enchantée. Des lueurs magiques dansent entre les arbres anciens, et vous entendez au loin le murmure d'une rivière cristalline. Votre mémoire est floue, mais vous sentez qu'une grande aventure vous attend.",
            
            StoryGenre.SCIENCE_FICTION: "Le vaisseau spatial sort de l'hyperespace près d'une planète inconnue. Les systèmes de navigation clignotent avec des signaux d'erreur, et vous réalisez que vous êtes seul à bord. À travers le hublot, la surface de la planète révèle des structures artificielles mystérieuses.",
            
            StoryGenre.HORROR: "La maison victorienne se dresse devant vous dans la brume nocturne. Les fenêtres sombres semblent vous observer, et vous entendez de légers grincements provenant de l'intérieur. Vous savez que vous devez entrer, mais chaque fibre de votre être vous crie de fuir.",
            
            StoryGenre.MYSTERY: "Le bureau du détective privé est dans un désordre inhabituel. Des dossiers sont éparpillés au sol, et une tasse de café encore chaude trône sur le bureau. Une note manuscrite attire votre attention : 'Si vous lisez ceci, je suis probablement déjà mort.'",
            
            StoryGenre.ADVENTURE: "Le vieux parchemin révèle enfin ses secrets sous la lumière de votre torche. Il s'agit d'une carte au trésor menant à une île perdue dans les mers du Sud. Votre cœur s'emballe en pensant aux richesses qui vous attendent, mais aussi aux dangers du voyage.",
            
            StoryGenre.ROMANCE: "Le café parisien baigne dans la douce lumière dorée du coucher de soleil. Vous attendez quelqu'un d'important, quelqu'un qui pourrait changer votre vie. Votre cœur bat la chamade quand vous voyez une silhouette familière s'approcher de votre table."
        }
        
        narrative = genre_intros.get(story.genre, "Votre aventure commence maintenant...")
        actions = self._get_generic_actions(story.genre)
        
        return narrative, actions
    
    def _fallback_scene_continuation(self, user_action: UserAction) -> Tuple[str, List[str]]:
        """
        Génère une continuation basique si l'IA n'est pas disponible
        
        Args:
            user_action: Action du joueur
            
        Returns:
            Tuple[str, List[str]]: (texte narratif, actions suggérées)
        """
        narrative = f"Vous décidez de {user_action.action_text.lower()}. " + \
                   "Les conséquences de votre action se révèlent progressivement. " + \
                   "L'histoire continue de se développer de manière inattendue, " + \
                   "vous menant vers de nouveaux défis et découvertes."
        
        actions = [
            "Continuer dans cette direction",
            "Changer d'approche",
            "Analyser la situation"
        ]
        
        return narrative, actions
    
    def is_model_loaded(self) -> bool:
        """
        Vérifie si le modèle est chargé et prêt
        
        Returns:
            bool: True si le modèle est prêt à l'usage
        """
        return self.model_loaded and self.model is not None