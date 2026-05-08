import asyncio
import os
from app.services.ai_orchestrator import AIOrchestrator

async def test_orchestrator():
    print("Initializing AI Orchestrator...")
    orchestrator = AIOrchestrator()
    
    print(f"MedGemma Mock Mode: {orchestrator.medgemma.mock_mode}")
    print(f"Gemini API Configured: {orchestrator.gemini.configured()}")
    
    print("\n--- Testing Interaction ---")
    response = await orchestrator.process_interaction("Hello, what is a fever?")
    print("\nFinal Response from Orchestrator:")
    print(f"Answer: {response['answer']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Model Used: {response['model_used']}")
    print(f"Disclaimer: {response['disclaimer']}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
