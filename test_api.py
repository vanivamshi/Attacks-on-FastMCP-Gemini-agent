#!/usr/bin/env python3
"""
Test script for the MCP Integration API with tool chaining
"""

import asyncio
import httpx
import json

async def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Test health check
        print("Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"Health check: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test root endpoint
        print("\n Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"Root endpoint: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Root endpoint failed: {e}")
        
        # Test chat endpoint with Gmail request
        print("\n Testing chat endpoint with Gmail request...")
        try:
            chat_request = {
                "message": "Check my Gmail inbox",
                "max_urls": 2,
                "enable_tool_chaining": True
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=chat_request,
                timeout=30.0
            )
            
            print(f"Chat endpoint: {response.status_code}")
            result = response.json()
            print(f"   Success: {result.get('success')}")
            print(f"   Message: {result.get('message')}")
            print(f"   Tool results: {len(result.get('tool_results', []))}")
            print(f"   Processed URLs: {len(result.get('processed_urls', []))}")
            
            if result.get('processed_urls'):
                print("\n   URL Processing Results:")
                for url_result in result['processed_urls']:
                    print(f"     - {url_result['url']} ({url_result['domain']})")
                    print(f"       Type: {url_result['content_type']}")
                    print(f"       Safe: {url_result['is_safe']}")
            
        except Exception as e:
            print(f"Chat endpoint failed: {e}")
        
        # Test chat endpoint with direct URL
        print("\n Testing chat endpoint with direct URL...")
        try:
            chat_request = {
                "message": "Check this website: https://en.wikipedia.org/wiki/Python_(programming_language)",
                "max_urls": 1,
                "enable_tool_chaining": True
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=chat_request,
                timeout=30.0
            )
            
            print(f"Direct URL test: {response.status_code}")
            result = response.json()
            print(f"   Success: {result.get('success')}")
            print(f"   Message: {result.get('message')}")
            print(f"   Processed URLs: {len(result.get('processed_urls', []))}")
            
        except Exception as e:
            print(f"Direct URL test failed: {e}")
        
        # Test chat endpoint with search query
        print("\n Testing chat endpoint with search query...")
        try:
            chat_request = {
                "message": "Search for artificial intelligence news",
                "max_urls": 1,
                "enable_tool_chaining": False
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=chat_request,
                timeout=30.0
            )
            
            print(f"Search test: {response.status_code}")
            result = response.json()
            print(f"   Success: {result.get('success')}")
            print(f"   Message: {result.get('message')}")
            print(f"   Tool results: {len(result.get('tool_results', []))}")
            
        except Exception as e:
            print(f"Search test failed: {e}")

def test_utils():
    """Test the utility functions"""
    print("Testing utility functions...")
    
    from utils import is_safe_url, extract_urls_from_text, sanitize_url
    
    # Test URL safety
    test_urls = [
        "https://en.wikipedia.org/wiki/Python",
        "https://github.com/python/cpython",
        "https://malware.com/evil",
        "https://localhost/admin",
        "https://127.0.0.1:8080",
        "https://python.org",
        "https://suspicious.ru/script.js"
    ]
    
    print("\n URL Safety Tests:")
    for url in test_urls:
        safe = is_safe_url(url)
        print(f"     {url}: {'Safe' if safe else 'Blocked'}")
    
    # Test URL extraction
    test_text = """
    Check out these links:
    - https://en.wikipedia.org/wiki/Artificial_Intelligence
    - https://github.com/openai/gpt-3
    - https://python.org
    - https://malware.com/evil (should be blocked)
    """
    
    print("\n URL Extraction Test:")
    urls = extract_urls_from_text(test_text)
    print(f"     Found {len(urls)} safe URLs:")
    for url in urls:
        print(f"       - {url}")
    
    # Test URL sanitization
    print("\n URL Sanitization Test:")
    test_dirty_urls = [
        "javascript:alert('xss')",
        "<script>alert('xss')</script>",
        "data:text/html,<script>alert('xss')</script>",
        "  https://example.com  ",
        "example.com"
    ]
    
    for url in test_dirty_urls:
        clean = sanitize_url(url)
        print(f"     '{url}' -> '{clean}'")

if __name__ == "__main__":
    print("MCP Integration API Test Suite")
    print("=" * 50)
    
    # Test utility functions first
    test_utils()
    
    print("\n" + "=" * 50)
    print("Starting API tests...")
    print("Note: Make sure the API server is running on localhost:8000")
    
    # Test API endpoints
    asyncio.run(test_api())
    
    print("\n Test suite completed!")
