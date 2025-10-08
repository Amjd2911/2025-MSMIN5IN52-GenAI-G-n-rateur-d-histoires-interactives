#!/usr/bin/env python3
"""
Script de test pour le backend du générateur d'histoires interactives

Ce script effectue des tests de base pour vérifier :
- Le démarrage de l'application
- La réponse des endpoints principaux
- L'état des services IA
- La génération d'une histoire simple

Usage:
    python test_backend.py
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any


class BackendTester:
    """
    Testeur pour l'API du générateur d'histoires
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialise le testeur
        
        Args:
            base_url: URL de base de l'API
        """
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        """Contexte d'entrée asynchrone"""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Contexte de sortie asynchrone"""
        if self.client:
            await self.client.aclose()
    
    async def test_server_running(self) -> bool:
        """
        Teste si le serveur répond
        
        Returns:
            bool: True si le serveur répond
        """
        try:
            response = await self.client.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Serveur non accessible: {e}")
            return False
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """
        Teste l'endpoint de santé
        
        Returns:
            Dict contenant les informations de santé
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_ai_services_status(self) -> Dict[str, Any]:
        """
        Teste l'état des services IA
        
        Returns:
            Dict contenant l'état des services IA
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/health/ai-services"
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_create_story(self) -> Dict[str, Any]:
        """
        Teste la création d'une histoire
        
        Returns:
            Dict contenant la réponse de création d'histoire
        """
        story_data = {
            "title": "Test Adventure",
            "genre": "adventure",
            "initial_context": "You are a brave explorer entering a mysterious cave."
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/stories",
                json=story_data
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Status code: {response.status_code}",
                    "response": response.text
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def test_story_action(self, story_id: str) -> Dict[str, Any]:
        """
        Teste une action sur une histoire
        
        Args:
            story_id: ID de l'histoire
            
        Returns:
            Dict contenant la réponse de l'action
        """
        action_data = {
            "action": "look around carefully"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/stories/{story_id}/action",
                json=action_data
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Status code: {response.status_code}",
                    "response": response.text
                }
        except Exception as e:
            return {"error": str(e)}
    
    def print_result(self, test_name: str, result: Dict[str, Any], 
                    success_key: str = None):
        """
        Affiche le résultat d'un test
        
        Args:
            test_name: Nom du test
            result: Résultat du test
            success_key: Clé indiquant le succès
        """
        print(f"\n📋 {test_name}")
        print("-" * 50)
        
        if "error" in result:
            print(f"❌ ÉCHEC: {result['error']}")
        else:
            if success_key and success_key in result:
                print(f"✅ SUCCÈS: {result[success_key]}")
            else:
                print("✅ SUCCÈS")
            
            # Affichage des informations importantes
            if "status" in result:
                print(f"   Status: {result['status']}")
            if "overall_status" in result:
                print(f"   État global: {result['overall_status']}")
            if "services" in result:
                print("   Services IA:")
                for service, status in result["services"].items():
                    status_icon = "✅" if status else "❌"
                    print(f"     {status_icon} {service}")
            if "story_id" in result:
                print(f"   ID Histoire: {result['story_id']}")
            if "current_scene" in result:
                scene = result["current_scene"]
                print(f"   Scène: {scene.get('description', 'N/A')[:100]}...")
    
    async def run_all_tests(self):
        """
        Lance tous les tests
        """
        print("🚀 DÉBUT DES TESTS DU BACKEND")
        print("=" * 60)
        
        # Test 1: Serveur accessible
        server_ok = await self.test_server_running()
        if not server_ok:
            print("❌ ARRÊT: Serveur non accessible")
            return
        print("✅ Serveur accessible")
        
        # Test 2: Health check
        health_result = await self.test_health_endpoint()
        self.print_result("Health Check", health_result, "status")
        
        # Test 3: État des services IA
        ai_status = await self.test_ai_services_status()
        self.print_result("Services IA", ai_status, "overall_status")
        
        # Test 4: Création d'histoire
        story_result = await self.test_create_story()
        self.print_result("Création d'Histoire", story_result, "story_id")
        
        # Test 5: Action sur l'histoire (si création réussie)
        if "story_id" in story_result:
            story_id = story_result["story_id"]
            action_result = await self.test_story_action(story_id)
            self.print_result("Action sur Histoire", action_result)
        
        print("\n" + "=" * 60)
        print("🏁 TESTS TERMINÉS")


async def main():
    """
    Fonction principale de test
    """
    print("Assurez-vous que le serveur FastAPI est démarré sur localhost:8000")
    print("Commande: uvicorn main:app --reload")
    print()
    
    # Attente pour laisser le temps de démarrer le serveur
    await asyncio.sleep(2)
    
    async with BackendTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())