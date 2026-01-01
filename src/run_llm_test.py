"""
LLM Setup Test Runner
Phase 5.1: Validate LLM configuration

Run: python src/run_llm_test.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_wrapper import LLMWrapper, get_llm
from config.llm_config import SYSTEM_PROMPTS


def main():
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 5.1: LLM SETUP TEST")
    print("=" * 70)
    
    # Initialize LLM
    print("\n📦 Initializing LLM Wrapper...")
    llm = get_llm()
    print(f"   {llm}")
    
    # Health check
    print("\n🏥 Health Check...")
    health = llm.health_check()
    
    print(f"\n   Ollama:")
    if health["ollama"]["available"]:
        print(f"      ✅ Available")
        print(f"      📋 Models: {', '.join(health['ollama']['models'])}")
    else:
        print(f"      ❌ Not Available")
        if "error" in health["ollama"]:
            print(f"      ⚠️ Error: {health['ollama']['error']}")
    
    print(f"\n   Groq:")
    if health["groq"]["available"]:
        print(f"      ✅ Available")
    else:
        print(f"      ❌ Not Available")
        if "error" in health["groq"]:
            print(f"      ⚠️ Error: {health['groq']['error']}")
    
    # Test queries
    if health["ollama"]["available"] or health["groq"]["available"]:
        print("\n" + "=" * 70)
        print("🧪 TESTING LLM RESPONSES")
        print("=" * 70)
        
        test_queries = [
            {
                "prompt": "What is Source Data Verification (SDV) in clinical trials?",
                "system": "default",
                "description": "Basic clinical trial question"
            },
            {
                "prompt": "A site has 50 open queries. What should the CRA do first?",
                "system": "resolver",
                "description": "Resolver agent test"
            },
            {
                "prompt": "Generate a brief reminder email for missing signatures.",
                "system": "communicator",
                "description": "Communicator agent test"
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{'─' * 70}")
            print(f"Test {i}: {test['description']}")
            print(f"System Prompt: {test['system']}")
            print(f"Query: {test['prompt'][:60]}...")
            print("─" * 70)
            
            response = llm.generate(
                prompt=test["prompt"],
                system_prompt=test["system"]
            )
            
            if response.success:
                print(f"\n✅ Response ({response.provider}/{response.model}):")
                print(f"   Latency: {response.latency_ms:.0f}ms")
                print(f"   Tokens: {response.total_tokens}")
                print(f"\n{response.content[:500]}...")
            else:
                print(f"\n❌ Error: {response.error}")
        
        # Test JSON generation
        print(f"\n{'─' * 70}")
        print("Test 4: JSON Generation")
        print("─" * 70)
        
        json_response = llm.generate_json(
            prompt="""Analyze this clinical trial issue:
            - Site: JP-101
            - Issue: 25 open queries
            - Days outstanding: 14
            
            Return JSON with: priority, root_cause, recommendation"""
        )
        
        if "error" not in json_response:
            print(f"\n✅ JSON Response:")
            import json
            print(json.dumps(json_response, indent=2))
        else:
            print(f"\n⚠️ JSON Parse Issue:")
            print(f"   Error: {json_response.get('error')}")
            print(f"   Raw: {json_response.get('raw', '')[:200]}...")
        
        # Print stats
        print("\n" + "=" * 70)
        print("📊 USAGE STATISTICS")
        print("=" * 70)
        stats = llm.get_stats()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Ollama Calls: {stats['ollama_calls']}")
        print(f"   Groq Calls: {stats['groq_calls']}")
        print(f"   Cache Hits: {stats['cache_hits']}")
        print(f"   Total Tokens: {stats['total_tokens']}")
        print(f"   Errors: {stats['errors']}")
    
    else:
        print("\n" + "=" * 70)
        print("⚠️ NO LLM PROVIDERS AVAILABLE")
        print("=" * 70)
        print("""
To fix this:

1. For Ollama (recommended):
   - Download from: https://ollama.ai/download
   - Install and run: ollama pull mistral
   - Verify: ollama list

2. For Groq (backup):
   - Sign up at: https://console.groq.com/
   - Get API key
   - Set environment variable: 
     $env:GROQ_API_KEY = "your-key-here"
""")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 5.1 SETUP COMPLETE")
    print("=" * 70)
    print("""
Files created:
├── config/llm_config.py        - LLM configuration
├── src/agents/__init__.py      - Agents module
├── src/agents/llm_wrapper.py   - LLM wrapper class
└── src/run_llm_test.py         - Test runner

Next steps:
1. Ensure Ollama is running: ollama serve
2. Or set GROQ_API_KEY environment variable
3. Run: python src/run_llm_test.py
""")


if __name__ == "__main__":
    main()