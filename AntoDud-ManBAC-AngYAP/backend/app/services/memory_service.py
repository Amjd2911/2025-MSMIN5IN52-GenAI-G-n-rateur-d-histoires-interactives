"""
Service de mémoire contextuelle intelligente

Ce service gère la mémoire narrative pour maintenir la cohérence :
- Analyse automatique des textes pour extraire les entités
- Mise à jour intelligente du contexte global
- Résumé adaptatif des événements importants
- Gestion des relations entre personnages et lieux

Architecture :
- Analyseur de texte pour extraction d'entités
- Système de scoring pour prioriser les informations
- Compression intelligente de l'historique
- Détection de contradictions narratives
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from app.models.schemas import (
    Story, Scene, StoryMemory, Character, Location, UserAction,
    StoryGenre
)
from app.config import settings


class MemoryService:
    """
    Service intelligent pour la gestion de la mémoire narrative
    
    Ce service maintient la cohérence de l'histoire en :
    - Analysant chaque nouvelle scène pour extraire les informations
    - Mettant à jour les entités (personnages, lieux, objets)
    - Gérant l'historique des événements avec priorités
    - Détectant et résolvant les incohérences potentielles
    """
    
    def __init__(self):
        """
        Initialise le service de mémoire avec les analyseurs de texte
        """
        # Patterns regex pour l'extraction d'entités
        self.entity_patterns = self._compile_entity_patterns()
        
        # Dictionnaires de mots-clés par catégorie
        self.character_indicators = self._load_character_indicators()
        self.location_indicators = self._load_location_indicators()
        self.action_indicators = self._load_action_indicators()
        
        # Configuration de la mémoire
        self.max_events_in_memory = 50  # Nombre max d'événements stockés
        self.summary_threshold = 10     # Seuil pour déclencher un résumé
        self.importance_decay = 0.9     # Facteur de décroissance de l'importance
    
    def _compile_entity_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile les patterns regex pour l'extraction d'entités
        
        Returns:
            Dict contenant les patterns compilés par type d'entité
        """
        return {
            # Noms propres (personnages et lieux)
            "proper_names": re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            
            # Personnages (pronoms et désignations)
            "character_refs": re.compile(r'\b(?:il|elle|lui|elle|le|la|vous|je|tu|nous|ils|elles)\b', re.IGNORECASE),
            
            # Lieux et directions
            "locations": re.compile(r'\b(?:dans|à|vers|sur|sous|devant|derrière|près de|loin de)\s+(?:la?|les?|une?|des?)\s+([a-zàâäéèêëïîôöùûüÿç]+)', re.IGNORECASE),
            
            # Actions importantes
            "actions": re.compile(r'\b(?:décide|choisit|prend|trouve|découvre|rencontre|combat|parle|ouvre|ferme|entre|sort)\b', re.IGNORECASE),
            
            # Objets importants
            "objects": re.compile(r'\b(?:une?|le|la|des?)\s+([a-zàâäéèêëïîôöùûüÿç]+(?:\s+[a-zàâäéèêëïîôöùûüÿç]+)?)\b', re.IGNORECASE)
        }
    
    def _load_character_indicators(self) -> Set[str]:
        """
        Charge les indicateurs de personnages (titres, rôles, etc.)
        
        Returns:
            Set des mots-clés indiquant des personnages
        """
        return {
            # Titres et rôles
            "roi", "reine", "prince", "princesse", "seigneur", "dame",
            "chevalier", "guerrier", "mage", "sorcier", "prêtre",
            "marchand", "garde", "capitaine", "général", "baron",
            
            # Descriptions physiques
            "homme", "femme", "enfant", "vieillard", "jeune",
            "grand", "petit", "blond", "brun", "barbu",
            
            # Relations
            "ami", "ennemi", "allié", "père", "mère", "frère", "sœur",
            "fils", "fille", "époux", "épouse", "maître", "serviteur"
        }
    
    def _load_location_indicators(self) -> Set[str]:
        """
        Charge les indicateurs de lieux
        
        Returns:
            Set des mots-clés indiquant des lieux
        """
        return {
            # Bâtiments
            "château", "maison", "cabane", "tour", "temple", "église",
            "auberge", "taverne", "marché", "palais", "fort",
            
            # Environnements naturels
            "forêt", "montagne", "rivière", "lac", "mer", "océan",
            "plaine", "colline", "vallée", "désert", "jungle",
            
            # Lieux urbains
            "ville", "village", "cité", "quartier", "rue", "place",
            "port", "pont", "route", "chemin", "sentier",
            
            # Lieux spéciaux
            "cave", "grotte", "donjon", "crypte", "laboratoire",
            "bibliothèque", "armurerie", "écurie", "jardin"
        }
    
    def _load_action_indicators(self) -> Set[str]:
        """
        Charge les indicateurs d'actions importantes
        
        Returns:
            Set des verbes d'action significatifs
        """
        return {
            # Actions de mouvement
            "aller", "venir", "partir", "arriver", "entrer", "sortir",
            "monter", "descendre", "traverser", "suivre", "fuir",
            
            # Actions d'interaction
            "parler", "dire", "demander", "répondre", "écouter",
            "rencontrer", "saluer", "présenter", "convaincre",
            
            # Actions de manipulation
            "prendre", "donner", "ouvrir", "fermer", "casser",
            "construire", "réparer", "détruire", "créer",
            
            # Actions de combat
            "attaquer", "défendre", "combattre", "frapper", "tuer",
            "blesser", "soigner", "protéger", "sauver",
            
            # Actions de découverte
            "trouver", "découvrir", "chercher", "explorer", "examiner",
            "observer", "voir", "entendre", "sentir", "toucher"
        }
    
    async def analyze_and_update_memory(
        self, 
        story: Story, 
        new_scene: Scene
    ) -> StoryMemory:
        """
        Analyse une nouvelle scène et met à jour la mémoire narrative
        
        Cette méthode centrale :
        1. Extrait les entités de la nouvelle scène
        2. Met à jour les personnages et lieux connus
        3. Ajoute les événements importants
        4. Recalcule le résumé si nécessaire
        5. Détecte les incohérences potentielles
        
        Args:
            story: L'histoire complète
            new_scene: La nouvelle scène à analyser
            
        Returns:
            StoryMemory: Mémoire mise à jour
        """
        memory = story.memory
        
        # 1. Extraction des entités de la nouvelle scène
        entities = await self._extract_entities_from_scene(new_scene, story.genre)
        
        # 2. Mise à jour des personnages
        memory = await self._update_characters(memory, entities, new_scene)
        
        # 3. Mise à jour des lieux
        memory = await self._update_locations(memory, entities, new_scene)
        
        # 4. Ajout des événements importants
        memory = await self._add_important_events(memory, new_scene, entities)
        
        # 5. Mise à jour du résumé global
        memory = await self._update_global_summary(memory, story.scenes)
        
        # 6. Nettoyage et optimisation de la mémoire
        memory = await self._optimize_memory(memory)
        
        # 7. Mise à jour des métadonnées
        memory.last_updated = datetime.now()
        memory.memory_version += 1
        
        return memory
    
    async def _extract_entities_from_scene(
        self, 
        scene: Scene, 
        genre: StoryGenre
    ) -> Dict[str, List[str]]:
        """
        Extrait les entités importantes d'une scène
        
        Args:
            scene: Scène à analyser
            genre: Genre pour adapter l'extraction
            
        Returns:
            Dict contenant les entités extraites par catégorie
        """
        text = scene.narrative_text
        entities = {
            "characters": [],
            "locations": [],
            "objects": [],
            "actions": [],
            "proper_names": []
        }
        
        # Extraction des noms propres
        proper_names = self.entity_patterns["proper_names"].findall(text)
        entities["proper_names"] = list(set(proper_names))
        
        # Classification des noms propres en personnages ou lieux
        for name in proper_names:
            context = self._get_word_context(text, name, window=10)
            if self._is_likely_character(context):
                entities["characters"].append(name)
            elif self._is_likely_location(context):
                entities["locations"].append(name)
        
        # Extraction des lieux mentionnés
        location_matches = self.entity_patterns["locations"].findall(text)
        entities["locations"].extend([loc.capitalize() for loc in location_matches])
        
        # Extraction des actions importantes
        action_matches = self.entity_patterns["actions"].findall(text)
        entities["actions"] = list(set(action_matches))
        
        # Extraction des objets mentionnés
        object_matches = self.entity_patterns["objects"].findall(text)
        # Filtrer les objets communs peu importants
        important_objects = [
            obj for obj in object_matches 
            if self._is_important_object(obj, genre)
        ]
        entities["objects"] = important_objects
        
        return entities
    
    def _get_word_context(self, text: str, word: str, window: int = 10) -> str:
        """
        Récupère le contexte autour d'un mot dans le texte
        
        Args:
            text: Texte complet
            word: Mot dont récupérer le contexte
            window: Nombre de mots avant et après
            
        Returns:
            str: Contexte autour du mot
        """
        words = text.split()
        contexts = []
        
        for i, w in enumerate(words):
            if word.lower() in w.lower():
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context = ' '.join(words[start:end])
                contexts.append(context)
        
        return ' '.join(contexts)
    
    def _is_likely_character(self, context: str) -> bool:
        """
        Détermine si un nom propre est probablement un personnage
        
        Args:
            context: Contexte autour du nom
            
        Returns:
            bool: True si probablement un personnage
        """
        context_lower = context.lower()
        character_score = sum(
            1 for indicator in self.character_indicators
            if indicator in context_lower
        )
        
        # Vérifier la présence de verbes d'action personnelle
        personal_actions = ["dit", "répond", "sourit", "regarde", "pense", "décide"]
        action_score = sum(
            1 for action in personal_actions
            if action in context_lower
        )
        
        return character_score > 0 or action_score > 0
    
    def _is_likely_location(self, context: str) -> bool:
        """
        Détermine si un nom propre est probablement un lieu
        
        Args:
            context: Contexte autour du nom
            
        Returns:
            bool: True si probablement un lieu
        """
        context_lower = context.lower()
        location_score = sum(
            1 for indicator in self.location_indicators
            if indicator in context_lower
        )
        
        # Vérifier les prépositions de lieu
        location_prepositions = ["dans", "à", "vers", "sur", "sous", "près de"]
        prep_score = sum(
            1 for prep in location_prepositions
            if prep in context_lower
        )
        
        return location_score > 0 or prep_score > 0
    
    def _is_important_object(self, obj: str, genre: StoryGenre) -> bool:
        """
        Détermine si un objet est important selon le genre
        
        Args:
            obj: Objet à évaluer
            genre: Genre de l'histoire
            
        Returns:
            bool: True si l'objet est important
        """
        # Objets généralement peu importants
        common_objects = {
            "table", "chaise", "porte", "fenêtre", "mur", "sol",
            "air", "eau", "temps", "moment", "chose", "fois"
        }
        
        if obj.lower() in common_objects:
            return False
        
        # Objets importants par genre
        genre_objects = {
            StoryGenre.FANTASY: {
                "épée", "magie", "sortilège", "cristal", "trésor",
                "potion", "parchemin", "baguette", "grimoire"
            },
            StoryGenre.SCIENCE_FICTION: {
                "vaisseau", "robot", "laser", "ordinateur", "scanner",
                "communicateur", "réacteur", "hologramme"
            },
            StoryGenre.HORROR: {
                "sang", "couteau", "cercueil", "miroir", "pendule",
                "portrait", "journal", "clé", "livre"
            },
            StoryGenre.MYSTERY: {
                "indice", "preuve", "témoin", "suspect", "mobile",
                "alibi", "arme", "lettre", "photo"
            }
        }
        
        important_for_genre = genre_objects.get(genre, set())
        return obj.lower() in important_for_genre or len(obj) > 4
    
    async def _update_characters(
        self, 
        memory: StoryMemory, 
        entities: Dict[str, List[str]], 
        scene: Scene
    ) -> StoryMemory:
        """
        Met à jour les personnages connus dans la mémoire
        
        Args:
            memory: Mémoire actuelle
            entities: Entités extraites de la scène
            scene: Scène analysée
            
        Returns:
            StoryMemory: Mémoire avec personnages mis à jour
        """
        for char_name in entities["characters"]:
            if char_name not in memory.characters:
                # Nouveau personnage découvert
                character = Character(
                    name=char_name,
                    description=f"Personnage mentionné pour la première fois dans la scène {scene.scene_number}",
                    role="Inconnu",
                    traits=[],
                    relationships={},
                    first_appearance_scene=scene.scene_number
                )
                memory.characters[char_name] = character
            else:
                # Personnage existant - possibilité de mise à jour
                # TODO: Analyser le contexte pour enrichir la description
                pass
        
        return memory
    
    async def _update_locations(
        self, 
        memory: StoryMemory, 
        entities: Dict[str, List[str]], 
        scene: Scene
    ) -> StoryMemory:
        """
        Met à jour les lieux connus dans la mémoire
        
        Args:
            memory: Mémoire actuelle
            entities: Entités extraites de la scène
            scene: Scène analysée
            
        Returns:
            StoryMemory: Mémoire avec lieux mis à jour
        """
        for loc_name in entities["locations"]:
            if loc_name not in memory.locations:
                # Nouveau lieu découvert
                location = Location(
                    name=loc_name,
                    description=f"Lieu mentionné pour la première fois dans la scène {scene.scene_number}",
                    atmosphere="Indéterminée",
                    key_features=[],
                    first_visit_scene=scene.scene_number
                )
                memory.locations[loc_name] = location
        
        return memory
    
    async def _add_important_events(
        self, 
        memory: StoryMemory, 
        scene: Scene, 
        entities: Dict[str, List[str]]
    ) -> StoryMemory:
        """
        Ajoute les événements importants à l'historique
        
        Args:
            memory: Mémoire actuelle
            scene: Scène analysée
            entities: Entités extraites
            
        Returns:
            StoryMemory: Mémoire avec événements ajoutés
        """
        # Création d'un résumé de l'événement
        event_summary = self._create_event_summary(scene, entities)
        
        # Calcul de l'importance de l'événement
        importance = self._calculate_event_importance(scene, entities)
        
        # Ajout à l'historique avec timestamp et importance
        event_entry = f"[Scène {scene.scene_number}] {event_summary} (Importance: {importance})"
        memory.key_events.append(event_entry)
        
        # Limitation du nombre d'événements stockés
        if len(memory.key_events) > self.max_events_in_memory:
            # Garder les événements les plus importants
            memory.key_events = self._keep_most_important_events(memory.key_events)
        
        return memory
    
    def _create_event_summary(
        self, 
        scene: Scene, 
        entities: Dict[str, List[str]]
    ) -> str:
        """
        Crée un résumé concis de l'événement de la scène
        
        Args:
            scene: Scène à résumer
            entities: Entités de la scène
            
        Returns:
            str: Résumé de l'événement
        """
        # Utiliser la première phrase comme base
        first_sentence = scene.narrative_text.split('.')[0]
        
        # Ajouter les entités importantes
        important_entities = []
        if entities["characters"]:
            important_entities.extend(entities["characters"][:2])
        if entities["locations"]:
            important_entities.extend(entities["locations"][:1])
        
        if important_entities:
            entities_str = ", ".join(important_entities)
            summary = f"{first_sentence} (Implique: {entities_str})"
        else:
            summary = first_sentence
        
        # Limiter la longueur
        return summary[:200] + "..." if len(summary) > 200 else summary
    
    def _calculate_event_importance(
        self, 
        scene: Scene, 
        entities: Dict[str, List[str]]
    ) -> int:
        """
        Calcule l'importance d'un événement (score 1-10)
        
        Args:
            scene: Scène à évaluer
            entities: Entités de la scène
            
        Returns:
            int: Score d'importance (1-10)
        """
        importance = 5  # Score de base
        
        # Bonus pour nouveaux personnages
        importance += len(entities["characters"]) * 2
        
        # Bonus pour nouveaux lieux
        importance += len(entities["locations"])
        
        # Bonus pour actions importantes
        important_actions = ["combat", "mort", "découverte", "révélation"]
        text_lower = scene.narrative_text.lower()
        for action in important_actions:
            if action in text_lower:
                importance += 3
        
        # Bonus pour action du joueur
        if scene.user_action:
            importance += 2
        
        # Limiter entre 1 et 10
        return max(1, min(10, importance))
    
    def _keep_most_important_events(self, events: List[str]) -> List[str]:
        """
        Garde les événements les plus importants quand la limite est atteinte
        
        Args:
            events: Liste des événements
            
        Returns:
            List[str]: Événements les plus importants
        """
        # Extraire les scores d'importance et trier
        scored_events = []
        for event in events:
            # Extraire le score depuis le format "(Importance: X)"
            import re
            match = re.search(r'\(Importance: (\d+)\)', event)
            importance = int(match.group(1)) if match else 5
            scored_events.append((importance, event))
        
        # Trier par importance décroissante
        scored_events.sort(key=lambda x: x[0], reverse=True)
        
        # Garder les plus importants
        keep_count = self.max_events_in_memory // 2
        return [event for _, event in scored_events[:keep_count]]
    
    async def _update_global_summary(
        self, 
        memory: StoryMemory, 
        all_scenes: List[Scene]
    ) -> StoryMemory:
        """
        Met à jour le résumé global de l'histoire
        
        Args:
            memory: Mémoire actuelle
            all_scenes: Toutes les scènes de l'histoire
            
        Returns:
            StoryMemory: Mémoire avec résumé mis à jour
        """
        # Déclencher un nouveau résumé si nécessaire
        if len(all_scenes) % self.summary_threshold == 0:
            memory.current_summary = await self._generate_comprehensive_summary(
                memory, all_scenes
            )
        else:
            # Mise à jour incrémentale
            latest_scene = all_scenes[-1]
            memory.current_summary += f" {latest_scene.narrative_text[:100]}..."
        
        # Limiter la longueur du résumé
        if len(memory.current_summary) > settings.MAX_CONTEXT_LENGTH:
            memory.current_summary = await self._compress_summary(memory.current_summary)
        
        return memory
    
    async def _generate_comprehensive_summary(
        self, 
        memory: StoryMemory, 
        all_scenes: List[Scene]
    ) -> str:
        """
        Génère un résumé complet de l'histoire
        
        Args:
            memory: Mémoire actuelle
            all_scenes: Toutes les scènes
            
        Returns:
            str: Résumé complet de l'histoire
        """
        # Pour l'instant, résumé basique basé sur les événements clés
        # TODO: Utiliser l'IA pour générer un résumé plus sophistiqué
        
        summary_parts = []
        
        # Résumé des personnages
        if memory.characters:
            char_names = list(memory.characters.keys())[:5]  # Top 5
            summary_parts.append(f"Personnages principaux: {', '.join(char_names)}")
        
        # Résumé des lieux
        if memory.locations:
            loc_names = list(memory.locations.keys())[:3]  # Top 3
            summary_parts.append(f"Lieux visités: {', '.join(loc_names)}")
        
        # Événements récents les plus importants
        recent_events = memory.key_events[-5:]  # 5 derniers événements
        if recent_events:
            events_text = ". ".join([
                event.split('] ')[1].split(' (Importance:')[0] 
                for event in recent_events
            ])
            summary_parts.append(f"Événements récents: {events_text}")
        
        return ". ".join(summary_parts)
    
    async def _compress_summary(self, summary: str) -> str:
        """
        Compresse un résumé trop long
        
        Args:
            summary: Résumé à comprimer
            
        Returns:
            str: Résumé compressé
        """
        # Compression basique - garder le début et la fin
        max_length = settings.MAX_CONTEXT_LENGTH // 2
        
        if len(summary) <= max_length:
            return summary
        
        # Garder le premier tiers et le dernier tiers
        part_length = max_length // 3
        start_part = summary[:part_length]
        end_part = summary[-part_length:]
        
        return f"{start_part}... [résumé compressé] ...{end_part}"
    
    async def _optimize_memory(self, memory: StoryMemory) -> StoryMemory:
        """
        Optimise la mémoire en supprimant les informations redondantes
        
        Args:
            memory: Mémoire à optimiser
            
        Returns:
            StoryMemory: Mémoire optimisée
        """
        # Supprimer les personnages sans information utile
        chars_to_remove = []
        for name, character in memory.characters.items():
            if (character.description == f"Personnage mentionné pour la première fois dans la scène {character.first_appearance_scene}" 
                and not character.traits and not character.relationships):
                chars_to_remove.append(name)
        
        for name in chars_to_remove:
            del memory.characters[name]
        
        # Pareil pour les lieux
        locs_to_remove = []
        for name, location in memory.locations.items():
            if (location.description == f"Lieu mentionné pour la première fois dans la scène {location.first_visit_scene}"
                and not location.key_features):
                locs_to_remove.append(name)
        
        for name in locs_to_remove:
            del memory.locations[name]
        
        return memory
    
    async def detect_inconsistencies(self, story: Story) -> List[str]:
        """
        Détecte les incohérences potentielles dans l'histoire
        
        Args:
            story: Histoire à analyser
            
        Returns:
            List[str]: Liste des incohérences détectées
        """
        inconsistencies = []
        
        # Vérifier les contradictions de personnages
        # TODO: Implémenter la détection d'incohérences
        
        return inconsistencies