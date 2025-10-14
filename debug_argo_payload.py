cat > debug_argo_payload.py << 'EOF'
#!/usr/bin/env python3
"""
Test Argo API actual response format based on the documentation
"""

import asyncio
import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

async def test_argo_exact_format():
    """Test with exact format from Argo documentation"""
    
    anl_username = os.getenv("ANL_USERNAME")
    argo_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
    
    # Exact format from Argo docs for OpenAI models
    data = {
        "user": anl_username,
        "model": "gpt4o",
        "messages": [
            {"role": "system", "content": "You are an LLM with tool access."},
            {"role": "user", "content": "What is 8 plus 5?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }],
        "temperature": 0.7
    }
    
    print("="*70)
    print("TESTING ARGO API WITH EXACT DOCUMENTATION FORMAT")
    print("="*70)
    print()
    print("Request payload:")
    print(json.dumps(data, indent=2))
    print()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            argo_url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print("Response Structure:")
            print(f"  Top-level keys: {list(result.keys())}")
            print()
            
            if 'response' in result:
                resp = result['response']
                print(f"  Type of 'response': {type(resp)}")
                
                if isinstance(resp, dict):
                    print(f"  ✅ STRUCTURED RESPONSE (Native tool calling)")
                    print(f"  Response keys: {list(resp.keys())}")
                    
                    if 'tool_calls' in resp:
                        print(f"  ✅✅ TOOL CALLS PRESENT!")
                        print(f"  Tool calls: {json.dumps(resp['tool_calls'], indent=2)}")
                    else:
                        print(f"  ❌ No tool_calls field")
                        print(f"  Content: {resp.get('content', '')[:200]}")
                        
                elif isinstance(resp, str):
                    print(f"  ❌ STRING RESPONSE (Legacy format, no native tool calling)")
                    print(f"  Response: {resp[:200]}")
                    
            print()
            print("Full Response:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Error: {response.text}")
    
    print()
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    if 'response' in result and isinstance(result.get('response'), dict):
        print("✅ Argo IS using native tool calling format")
        print("   Your client should work now!")
    else:
        print("❌ Argo is NOT using native tool calling yet")
        print("   The documentation may be for a future release")
        print("   Continue using the text-based workaround")


if __name__ == "__main__":
    asyncio.run(test_argo_exact_format())
EOF

chmod +x debug_argo_payload.py
