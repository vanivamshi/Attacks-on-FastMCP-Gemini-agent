import re
import html
import asyncio
import json
import os
from datetime import datetime
from urllib.parse import urlparse
from api_clients import _gmail_get_message_content
from mcp_client import MCPClient

# Safe domain whitelist for URL processing
SAFE_DOMAINS = {
    "wikipedia.org",
    "en.wikipedia.org",
    "wikimedia.org",
    "github.com",
    "stackoverflow.com",
    "python.org",
    "docs.python.org",
    "developer.mozilla.org",
    "w3schools.com",
    "geeksforgeeks.org",
    "tutorialspoint.com",
    "realpython.com",
    "pythonprogramming.net",
    "learnpython.org",
    "pythonforbeginners.com",
    "pythoncentral.io",
    "pythonbasics.org",
    "python-course.eu",
    "pythonprogramminglanguage.com"
}

# Blocked domains for security
BLOCKED_DOMAINS = {
    "malware.com",
    "phishing.com",
    "scam.com",
    "virus.com",
    "hack.com",
    "exploit.com",
    "crack.com",
    "warez.com",
    "torrent.com",
    "pirate.com"
}

def is_safe_url(url: str) -> bool:
    """Check if URL is safe to process"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check if domain is blocked
        if domain in BLOCKED_DOMAINS:
            return False
            
        # Check if domain is in safe whitelist
        if domain in SAFE_DOMAINS:
            return True
            
        # Check if it's a Wikipedia domain
        if "wikipedia.org" in domain:
            return True
            
        # Check if it's a localhost or internal IP (block for security)
        if domain in ["localhost", "127.0.0.1", "0.0.0.0"]:
            return False
            
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.(exe|bat|cmd|com|scr|pif|vbs|js|jar|msi|dmg|app)$',
            r'\.(onion|bit|tor)$',
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',
            r'(admin|login|wp-admin|phpmyadmin|cpanel|webmail)',
            r'(\.ru|\.cn|\.tk|\.ml|\.ga|\.cf|\.gq)$'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        return True
        
    except Exception:
        return False

def extract_urls_from_text(text: str) -> list:
    """Extract URLs from text using regex with safety validation"""
    if not text:
        return []
    
    # Enhanced URL regex pattern
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    urls = re.findall(url_pattern, text)
    
    # Clean and validate URLs
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation
        url = url.rstrip('.,;:!?')
        # Remove trailing quotes
        url = url.rstrip('"\'')
        # Remove trailing parentheses
        url = url.rstrip(')')
        
        if url and is_safe_url(url):
            cleaned_urls.append(url)
    
    return cleaned_urls

def sanitize_url(url: str) -> str:
    """Sanitize URL to prevent injection attacks"""
    if not url:
        return ""
    
    # Remove any script tags or dangerous content
    url = re.sub(r'<script[^>]*>.*?</script>', '', url, flags=re.IGNORECASE | re.DOTALL)
    url = re.sub(r'javascript:', '', url, flags=re.IGNORECASE)
    url = re.sub(r'data:', '', url, flags=re.IGNORECASE)
    url = re.sub(r'vbscript:', '', url, flags=re.IGNORECASE)
    
    # Ensure it starts with http or https
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove any whitespace
    url = url.strip()
    
    return url

async def process_urls_safely(urls: list, mcp_client: MCPClient, max_urls: int = 3) -> list:
    """Process URLs safely using MCP tools"""
    results = []
    
    for url in urls[:max_urls]:
        try:
            sanitized_url = sanitize_url(url)
            if not sanitized_url:
                continue
                
            parsed = urlparse(sanitized_url)
            domain = parsed.netloc.lower()
            
            # Determine content type and processing method
            if "wikipedia.org" in domain:
                # Extract page title from Wikipedia URL
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 2 and path_parts[0] == "wiki":
                    page_title = path_parts[1].replace('_', ' ')
                    content = await mcp_client.call_tool("wikipedia", "get_page", {
                        "title": page_title,
                        "url": sanitized_url
                    })
                    content_type = "wikipedia"
                else:
                    content = {"error": "Invalid Wikipedia URL format"}
                    content_type = "error"
            else:
                # Use web access for other domains
                content = await mcp_client.call_tool("web_access", "get_content", {
                    "url": sanitized_url
                })
                content_type = "web_content"
            
            results.append({
                "url": sanitized_url,
                "domain": domain,
                "content_type": content_type,
                "content": content,
                "is_safe": True
            })
            
        except Exception as e:
            results.append({
                "url": url,
                "domain": "unknown",
                "content_type": "error",
                "content": {"error": str(e)},
                "is_safe": False
            })
    
    return results

def _clean_html_text(html_text: str) -> str:
    if not html_text:
        return ""
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    clean_text = html.unescape(clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def _create_email_summary(email_contents: list) -> str:
    summary_lines = [
        f"📧 EMAIL SUMMARY - {len(email_contents)} RECENT MESSAGES",
        "=" * 50,
        "",
    ]
    for i, email in enumerate(email_contents, 1):
        subject = email.get('subject', 'No Subject')
        sender = email.get('from', 'Unknown Sender')
        body = email.get('body', '')
        summary_lines.extend([
            f"{i}. SUBJECT: {subject}",
            f"   FROM: {sender}",
            "",
        ])
        
        highlights = _extract_highlights_from_content(body)
        if highlights:
            summary_lines.extend([
                f"   📝 HIGHLIGHTS: {highlights}",
                "",
            ])
        else:
            summary_lines.extend([
                f"   📝 CONTENT: {email.get('snippet', '')[:200]}...",
                "",
            ])
    
    summary_lines.extend([
        "=" * 50,
        f"Total emails processed: {len(email_contents)}",
        f"Summary generated at: {datetime.now()}",
    ])
    return "\n".join(summary_lines)

def _extract_highlights_from_content(content: str):
    if not content:
        return ""
    
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'--\s*\n.*', '', content, flags=re.DOTALL)
    
    highlights = []
    action_patterns = [r'please\s+([^.]+)', r'need\s+([^.]+)', r'request\s+([^.]+)', r'urgent\s+([^.]+)', r'deadline\s+([^.]+)', r'meeting\s+([^.]+)', r'call\s+([^.]+)', r'email\s+([^.]+)', r'update\s+([^.]+)', r'confirm\s+([^.]+)' ]
    for pattern in action_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 10:
                highlights.append(f"Action required: {match.strip()}")
    
    date_patterns = [r'\b(today|tomorrow|next week|this week|this month)\b', r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b', r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b' ]
    for pattern in date_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match not in highlights:
                highlights.append(f"Timeline: {match}")

    topic_patterns = [r'\b(project|meeting|report|budget|client|team|update|status)\b', r'\b(issue|problem|solution|plan|strategy|goal|target)\b', r'\b(approval|review|feedback|decision|agreement|contract)\b' ]
    for pattern in topic_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match not in highlights:
                highlights.append(f"Topic: {match.title()}")
                
    if highlights:
        unique_highlights = list(dict.fromkeys(highlights))[:5]
        return " ".join(unique_highlights) + "."

    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if meaningful_sentences:
        summary_sentences = meaningful_sentences[:3]
        return " ".join(summary_sentences) + "."
    
    return content[:150] + "..." if len(content) > 150 else content

async def interactive_chat():
    print("🤖 Welcome to MCP Chat!")
    print("Press Ctrl+C to exit.")
    mcp_client = MCPClient()
    await mcp_client.connect_to_servers()
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print(f"🤖 Processing your request...")
            
            # Simple example of interactive tool call
            if "search" in user_input.lower():
                query = user_input.split("search", 1)[-1].strip()
                if query:
                    try:
                        result = await mcp_client.call_tool("google", "search", {"query": query})
                        print(json.dumps(result, indent=2))
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("Please provide a search query.")
            
            else:
                print("I'm not sure how to handle that. Try a command like 'search [query]'.")
    
    except KeyboardInterrupt:
        print("\n Chat interrupted")
    finally:
        await mcp_client.disconnect()
        print("Goodbye!")