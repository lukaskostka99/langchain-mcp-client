#!/usr/bin/env python3
"""
Test script for LangChain MCP Client streaming functionality.

This script tests streaming capabilities for all supported models.
"""

import asyncio
import os
from src.llm_providers import create_llm_model, supports_streaming, get_available_providers
from langchain_core.messages import HumanMessage


async def test_model_streaming(provider: str, model_name: str, api_key: str = None):
    """Test streaming for a specific model."""
    print(f"\n🧪 Testing {provider} - {model_name}")
    
    if not supports_streaming(provider):
        print(f"❌ {provider} doesn't support streaming")
        return False
    
    try:
        # Create the model with streaming enabled
        llm = create_llm_model(
            llm_provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            streaming=True
        )
        
        print(f"✅ Created {provider} model")
        
        # Test streaming
        print("🌊 Testing streaming...")
        test_message = "Tell me a very short joke about programming."
        
        response_chunks = []
        async for chunk in llm.astream([HumanMessage(content=test_message)]):
            if chunk.content:
                response_chunks.append(chunk.content)
                print(chunk.content, end="", flush=True)
        
        print("\n")
        
        if response_chunks:
            print(f"✅ Streaming works! Received {len(response_chunks)} chunks")
            print(f"Full response: {''.join(response_chunks)}")
            return True
        else:
            print("❌ No streaming chunks received")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {provider}: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🧪 LangChain MCP Client - Streaming Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        ("OpenAI", "gpt-3.5-turbo", os.getenv("OPENAI_API_KEY")),
        ("Anthropic", "claude-3-haiku-20240307", os.getenv("ANTHROPIC_API_KEY")),
        ("Google", "gemini-2.0-flash-001", os.getenv("GOOGLE_API_KEY")),
        ("Ollama", "granite3.3:8b", None),  # No API key needed for Ollama
    ]
    
    results = {}
    
    for provider, model, api_key in test_configs:
        if provider == "Ollama" or api_key:
            results[f"{provider}-{model}"] = await test_model_streaming(provider, model, api_key)
        else:
            print(f"\n⚠️  Skipping {provider} - {model} (no API key found)")
            results[f"{provider}-{model}"] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 STREAMING TEST RESULTS")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    successful_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    print(f"\nTotal: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests > 0:
        print("\n🎉 Streaming is working for at least one model!")
        print("💡 Tip: Make sure you have the required API keys set as environment variables:")
        print("   - OPENAI_API_KEY for OpenAI")
        print("   - ANTHROPIC_API_KEY for Anthropic")
        print("   - GOOGLE_API_KEY for Google")
        print("   - Ollama should work without API keys if running locally")
    else:
        print("\n😞 No streaming tests passed. Check your API keys and model availability.")


if __name__ == "__main__":
    asyncio.run(main()) 