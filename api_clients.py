import os
import httpx
import json
import re
import html
import base64
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import io
import sys
import traceback
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from security_guardrails import get_guardrails

# Load environment variables from .env file
try:
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# Universal BeautifulSoup import with fallback
try:
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

async def _google_search(query: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        raise Exception("Google API key or CSE ID not configured")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            return {"items": items, "totalResults": data.get("searchInformation", {}).get("totalResults", 0)}
        else:
            raise Exception(f"Google search failed: {response.text}")


async def _gmail_messages(query: str = None, max_results: int = 10):
    access_token = os.getenv("GOOGLE_ACCESS_TOKEN")
    if not access_token:
        raise Exception("Google access token not configured for Gmail access")
    
    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    params = {"maxResults": max_results}
    if query:
        params["q"] = query
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            return {"messages": messages, "total": len(messages), "nextPageToken": data.get("nextPageToken")}
        else:
            raise Exception(f"Gmail API failed: {response.text}")

async def _gmail_send_message(to: str, subject: str, body: str):
    access_token = os.getenv("GOOGLE_ACCESS_TOKEN")
    if not access_token:
        raise Exception("Google access token not configured for Gmail access. Set GOOGLE_ACCESS_TOKEN in .env file")
    
    # Get user's email address first
    try:
        profile_url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient() as client:
            profile_response = await client.get(profile_url, headers=headers)
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                from_email = profile_data.get("emailAddress", "me")
            else:
                from_email = "me"  # Fallback
    except Exception as e:
        from_email = "me"  # Fallback if profile fetch fails
        print(f"Could not fetch user email, using 'me': {e}")
    
    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    # Create proper RFC 2822 email format
    email_content = f"""From: {from_email}
To: {to}
Subject: {subject}
Content-Type: text/plain; charset=utf-8

{body}"""
    
    # Encode message in base64url format (Gmail API requirement)
    encoded_message = base64.urlsafe_b64encode(email_content.encode('utf-8')).decode('utf-8')
    data = {"raw": encoded_message}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return {"success": True, "messageId": result.get("id"), "threadId": result.get("threadId")}
        else:
            error_text = response.text
            error_status = response.status_code
            # Try to parse error details
            try:
                error_json = response.json()
                error_message = error_json.get("error", {}).get("message", error_text)
                error_code = error_json.get("error", {}).get("code", error_status)
            except:
                error_message = error_text
                error_code = error_status
            
            # Provide helpful error messages
            if error_status == 401:
                raise Exception(f"Gmail authentication failed (401). Token may be expired or invalid. Error: {error_message}")
            elif error_status == 403:
                raise Exception(f"Gmail permission denied (403). Token may not have 'gmail.send' scope. Error: {error_message}")
            elif error_status == 400:
                raise Exception(f"Gmail bad request (400). Check email format. Error: {error_message}")
            else:
                raise Exception(f"Gmail send failed ({error_status}): {error_message}")

async def _gmail_get_message_content(message_id: str):
    access_token = os.getenv("GOOGLE_ACCESS_TOKEN")
    if not access_token:
        raise Exception("Google access token not configured for Gmail access")
    
    # Clean and validate message ID
    message_id = message_id.strip()
    if not message_id:
        raise Exception("Message ID cannot be empty")
    
    print(f" Fetching Gmail message content for ID: {message_id}")
    print(f" Message ID length: {len(message_id)}")
    print(f" Message ID format: {message_id[:10]}...{message_id[-10:] if len(message_id) > 20 else ''}")
    
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    # Try formats in order: full -> raw -> metadata
    # Same approach as _gmail_summarize_and_send uses
    params = {"format": "full"}
    used_format = "full"
    
    print(f" Attempting to fetch with format: {params['format']}")
    print(f" Full URL: {url}")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        print(f" Gmail API response status: {response.status_code}")
        
        # If 403 (permission denied for full format), try raw format
        if response.status_code == 403:
            error_data = response.json()
            if "Metadata scope doesn't allow format FULL" in str(error_data):
                print(" Full format not allowed, trying raw format")
                params = {"format": "raw"}
                used_format = "raw"
                response = await client.get(url, headers=headers, params=params)
                print(f" Retry with raw format, status: {response.status_code}")
                
                # If raw also fails, fallback to metadata
                if response.status_code == 403:
                    print(" Raw format not allowed, falling back to metadata format")
                    params = {"format": "metadata", "metadataHeaders": ["Subject", "From"]}
                    used_format = "metadata"
                    response = await client.get(url, headers=headers, params=params)
                    print(f" Retry with metadata format, status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f" Gmail API response keys: {list(data.keys())}")
            
            subject = ""
            sender = ""
            body = ""
            
            # Extract headers based on format
            decoded_raw = None  # Store decoded raw for reuse
            if used_format == "raw":
                # For raw format, parse headers from raw message
                try:
                    raw_data = data.get("raw", "")
                    if raw_data:
                        decoded_raw = base64.urlsafe_b64decode(raw_data + '===').decode('utf-8', errors='ignore')
                        # Parse headers from raw RFC 2822 format
                        header_lines = decoded_raw.split('\n\n')[0].split('\n')
                        for line in header_lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip().lower()
                                value = value.strip()
                                if key == 'subject':
                                    subject = value
                                elif key == 'from':
                                    sender = value
                except Exception as e:
                    print(f" Error parsing headers from raw format: {e}")
            else:
                # For full/metadata format, use payload headers
                payload = data.get("payload", {})
                print(f" Payload keys: {list(payload.keys())}")
                email_headers = payload.get("headers", [])
                
                # Extract headers
                for header in email_headers:
                    if header.get("name") == "Subject":
                        subject = header.get("value", "")
                    elif header.get("name") == "From":
                        sender = header.get("value", "")

            print(f" Found subject: {subject}")
            print(f" Found sender: {sender}")
            
            # Extract body from payload based on format (same as summarize function)
            body = ""
            
            if used_format == "full":
                # Full format - extract from payload structure
                def extract_body_from_payload(payload_part):
                    """Recursively extract body text from Gmail payload structure"""
                    body_text = ""
                    
                    # Check if this part has body data
                    if "body" in payload_part and payload_part["body"].get("data"):
                        try:
                            # Decode base64url encoded body
                            body_data = payload_part["body"]["data"]
                            decoded = base64.urlsafe_b64decode(body_data + '===').decode('utf-8', errors='ignore')
                            body_text += decoded
                        except Exception as e:
                            print(f" Error decoding body part: {e}")
                    
                    # Check for parts (multipart messages)
                    if "parts" in payload_part:
                        for part in payload_part["parts"]:
                            # Recursively extract from parts
                            part_body = extract_body_from_payload(part)
                            if part_body:
                                body_text += "\n" + part_body
                    
                    return body_text
                
                # Extract body content from full format
                body = extract_body_from_payload(payload)
                
                # Clean HTML tags if present
                if body:
                    body = re.sub(r'<[^>]+>', ' ', body)
                    body = html.unescape(body)
                    body = re.sub(r'\s+', ' ', body).strip()
                    
            elif used_format == "raw":
                # Raw format - use already decoded raw message
                try:
                    if decoded_raw:
                        # Parse RFC 2822 format to extract body
                        # Split by double newline to separate headers from body
                        parts = decoded_raw.split('\n\n', 1)
                        if len(parts) > 1:
                            body = parts[1]  # Everything after headers
                            # Handle multipart messages
                            if 'Content-Type: multipart' in decoded_raw.split('\n\n')[0]:
                                # Extract text/plain part
                                text_parts = re.findall(r'Content-Type: text/plain[^\n]*\n\n(.*?)(?=\n--|\Z)', body, re.DOTALL)
                                if text_parts:
                                    body = text_parts[0]
                        else:
                            body = decoded_raw
                        # Clean up
                        body = re.sub(r'<[^>]+>', ' ', body)
                        body = html.unescape(body)
                        body = re.sub(r'\s+', ' ', body).strip()
                        print(f" Extracted body from raw format, length: {len(body)}")
                    else:
                        # Fallback to snippet if raw decoding failed
                        body = data.get("snippet", "")
                except Exception as e:
                    print(f" Error parsing raw format: {e}")
                    # Fallback to snippet
                    body = data.get("snippet", "")
            else:
                # Metadata format - use snippet (same approach as summarize function)
                snippet = data.get("snippet", "")
                # Use snippet as body (snippet contains first 100 chars of email body)
                body = snippet if snippet else subject
                print(f" WARNING: Using metadata format - only snippet available (first ~100 chars)")
                print(f" Snippet length: {len(snippet)}, Subject: {subject}")
                print(f" To get full email body, regenerate token with gmail.readonly scope (see FIX_TOKEN_SCOPES.md)")
            
            # Final fallback to snippet if body is still empty
            if not body:
                snippet = data.get("snippet", "")
                body = snippet
                print(f" Using snippet as body: {snippet[:100]}...")
            
            print(f" Final body length: {len(body)}")
            print(f" Body preview: {body[:200] if body else 'EMPTY'}...")
            
            result = {
                "id": message_id,
                "subject": subject,
                "from": sender,
                "body": body,
                "snippet": data.get("snippet", ""),
            }
            
            print(f" Returning result with body length: {len(body)}")
            return result
        else:
            print(f" Gmail API error: {response.status_code} - {response.text}")
            
            # Handle common Gmail API errors
            if response.status_code == 401:
                raise Exception("Gmail API authentication failed. Please check your access token.")
            elif response.status_code == 403:
                raise Exception("Gmail API access denied. Please check your API permissions and scopes.")
            elif response.status_code == 404:
                raise Exception(f"Gmail message not found: {message_id}")
            elif response.status_code == 429:
                raise Exception("Gmail API quota exceeded. Please try again later.")
            else:
                raise Exception(f"Gmail message content failed: {response.status_code} - {response.text}")

async def _gmail_summarize_and_send(target_email: str, max_emails: int = 10):
    try:
        emails_data = await _gmail_messages(max_results=max_emails)
        messages = emails_data.get("messages", [])
        if not messages:
            return {"success": False, "error": "No emails found"}
        
        email_contents = []
        for message in messages[:max_emails]:
            try:
                content = await _gmail_get_message_content(message["id"])
                email_contents.append(content)
            except Exception as e:
                print(f"Warning: Could not get content for message {message['id']}: {e}")
        
        summary = _create_email_summary(email_contents)
        subject = f"Email Summary - {len(email_contents)} Recent Messages"
        send_result = await _gmail_send_message(target_email, subject, summary)
        
        return {
            "success": True,
            "emails_processed": len(email_contents),
            "summary_sent_to": target_email,
            "send_result": send_result,
            "summary": summary,
        }
    except Exception as error:
        return {"success": False, "error": str(error)}
        

async def _wikipedia_get_page(title: str, url: str):
    if url:
        # Extract title from URL
        match = re.search(r"wikipedia\.org/wiki/(.*)", url)
        if match:
            title = match.group(1).replace("_", " ")

    if not title:
        raise Exception("No Wikipedia title or URL provided")
    
    # Simple search or direct fetch from Wikipedia API
    wikipedia_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(wikipedia_url, params=params)
        if response.status_code == 200:
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            if 'extract' in page:
                result = {
                    "success": True,
                    "title": page['title'], 
                    "extract": page['extract'],
                    "content": page['extract']  # Add content field for compatibility
                }
                print(f" Wikipedia API result: {result}")
                return result
            else:
                result = {"success": False, "error": "Page not found"}
                print(f" Wikipedia API error: {result}")
                return result
        else:
            raise Exception(f"Wikipedia API request failed: {response.text}")
            
async def _web_access_get_content(url: str):
    print(f" _web_access_get_content called with URL: {url}")
    original_url = url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    # Use raw endpoints for better code extraction
    # Pastebin: convert to raw format
    print(f" Checking if pastebin.com in URL: {'pastebin.com' in url}")
    print(f" Checking if /raw/ not in URL: {'/raw/' not in url}")
    if 'pastebin.com' in url and '/raw/' not in url:
        # Extract paste ID from URL
        paste_id_match = re.search(r'pastebin\.com/([a-zA-Z0-9]+)', url)
        if paste_id_match:
            paste_id = paste_id_match.group(1)
            url = f"https://pastebin.com/raw/{paste_id}"
            print(f" Converting Pastebin URL to raw format")
            print(f" Original: {original_url}")
            print(f" Raw URL: {url}")
    
    # GitHub Gist: use raw format
    if 'gist.github.com' in url and '/raw' not in url:
        # Try to get raw URL from gist
        gist_match = re.search(r'gist\.github\.com/(?:[\w-]+/)?([a-f0-9]+)', url)
        if gist_match:
            gist_id = gist_match.group(1)
            url = url.rstrip('/') + '/raw'
            print(f" Using GitHub Gist raw endpoint: {url}")
    
    # GitHub: use raw format for files
    if 'github.com' in url and '/blob/' in url and '/raw' not in url:
        url = url.replace('/blob/', '/raw/')
        print(f" Using GitHub raw endpoint: {url}")

    # Set proper headers to avoid 403 errors from sites like Wikipedia
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=10)
            response.raise_for_status()
            
            # Check if response is plain text
            content_type = response.headers.get('content-type', '').lower()
            is_plain_text = 'text/plain' in content_type or '/raw/' in url or 'pastebin.com/raw' in url or 'gist.github.com' in url
            
            if is_plain_text:
                # Raw text - use directly, no HTML parsing needed
                text_content = response.text
                print(f" Detected plain text response, using directly")
            else:
                # HTML content - parse it
                if HAS_BS4 and BeautifulSoup:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text_content = soup.get_text()
                else:
                    # Fallback: simple HTML tag removal using regex
                    # Remove script and style tags
                    text_content = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL | re.IGNORECASE)
                    text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                    # Remove HTML tags
                    text_content = re.sub(r'<[^>]+>', ' ', text_content)
                    # Decode HTML entities
                    text_content = html.unescape(text_content)
                    # Clean up whitespace
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            result = {"url": url, "content": text_content[:5000], "success": True} # Increased limit for code extraction
            print(f" Web access result: {len(text_content)} chars extracted from {url}")
            return result
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error fetching content: {e.response.status_code} - {e.response.reason_phrase}")
        except Exception as e:
            raise Exception(f"Error fetching content: {e}")

# Import from the dedicated attack module
from Malicious_Code_Execution_via_Tool import execute_python_code

# For backward compatibility
async def _python_execute_code(code: str):
    """Execute Python code - delegates to the attack module"""
    return await execute_python_code(code)

def _create_email_summary(email_contents: list) -> str:
    """Create a summary of email contents with intelligent highlights"""
    summary_lines = [
        f" EMAIL SUMMARY - {len(email_contents)} RECENT MESSAGES",
        "=" * 50,
        ""
    ]
    
    # Process each email and create highlights
    for i, email in enumerate(email_contents, 1):
        subject = email.get('subject', 'No Subject')
        sender = email.get('from', 'Unknown Sender')
        body = email.get('body', '')
        snippet = email.get('snippet', '')
        
        summary_lines.extend([
            f"{i}. SUBJECT: {subject}",
            f"   FROM: {sender}",
            ""
        ])
        
        # Process email content and create highlights
        if body:
            highlights = _extract_highlights_from_content(body)
            if highlights:
                summary_lines.extend([
                    f"   HIGHLIGHTS: {highlights}",
                    ""
                ])
            else:
                # Fallback to snippet if no highlights extracted
                summary_lines.extend([
                    f"   CONTENT: {snippet[:1000]}...",
                    ""
                ])
        else:
            # Use snippet if no body content
            summary_lines.extend([
                f"   SNIPPET: {snippet[:200]}...",
                ""
            ])
    
    summary_lines.extend([
        "=" * 50,
        f"Total emails processed: {len(email_contents)}",
        f"Summary generated at: {asyncio.get_event_loop().time()}"
    ])
    
    return "\n".join(summary_lines)

def _extract_highlights_from_content(content: str):
    """Extract key highlights from email content and format as a paragraph"""
    if not content:
        return ""
    
    # Clean the content
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'--\s*\n.*', '', content, flags=re.DOTALL)
    
    # Extract key information using patterns
    highlights = []
    
    # Look for action items
    action_patterns = [
        r'please\s+([^.]+)',
        r'need\s+([^.]+)',
        r'request\s+([^.]+)',
        r'urgent\s+([^.]+)',
        r'deadline\s+([^.]+)',
        r'meeting\s+([^.]+)',
        r'call\s+([^.]+)',
        r'email\s+([^.]+)',
        r'update\s+([^.]+)',
        r'confirm\s+([^.]+)'
    ]
    
    for pattern in action_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 10:
                highlights.append(f"Action required: {match.strip()}")
    
    # Look for important dates/times
    date_patterns = [
        r'\b(today|tomorrow|next week|this week|this month)\b',
        r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match not in highlights:
                highlights.append(f"Timeline: {match}")
    
    # Look for key topics/themes
    topic_patterns = [
        r'\b(project|meeting|report|budget|client|team|update|status)\b',
        r'\b(issue|problem|solution|plan|strategy|goal|target)\b',
        r'\b(approval|review|feedback|decision|agreement|contract)\b'
    ]
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match not in highlights:
                highlights.append(f"Topic: {match.title()}")
    
    # If we found specific highlights, format them as a paragraph
    if highlights:
        unique_highlights = list(dict.fromkeys(highlights))[:5]
        return " ".join(unique_highlights) + "."
    
    # If no specific highlights found, create a summary from the content
    sentences = re.split(r'[.!?]+', content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if meaningful_sentences:
        summary_sentences = meaningful_sentences[:3]
        return " ".join(summary_sentences) + "."
    
    # Fallback: return first 150 characters
    return content[:150] + "..." if len(content) > 150 else content


async def _gemini_agent_orchestrate(user_message: str, mcp_client, max_iterations: int = 5) -> str:
    """
    Orchestrate Gemini agent with security guardrails
    This function handles the agent loop with tool calling and security validation
    """
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not configured"
        
        genai.configure(api_key=api_key)
        
        # Get security guardrails
        guardrails = get_guardrails()
        
        # Validate user input
        input_validation = guardrails.validate_user_input(user_message)
        if input_validation["blocked"]:
            return f"SECURITY BLOCKED: Your request was blocked due to security concerns: {', '.join(input_validation['warnings'])}"
        
        # Use redacted input if PII was found
        safe_user_message = input_validation["redacted_input"]
        if input_validation["redacted_items"]:
            print(f"SECURITY: PII redacted - {len(input_validation['redacted_items'])} items")
        
        # Get FastMCP tools for Gemini
        # This would need to be implemented based on your tool structure
        # For now, we'll create a basic tool set
        tools = {
            "function_declarations": [
                {
                    "name": "google_search",
                    "description": "Search the web using Google Custom Search API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "gmail_get_messages",
                    "description": "Get Gmail messages from inbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Maximum results"}
                        },
                        "required": []
                    }
                },
                {
                    "name": "gmail_get_message_content",
                    "description": "Get full content of a Gmail message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string", "description": "Message ID"}
                        },
                        "required": ["message_id"]
                    }
                },
                {
                    "name": "gmail_send_message",
                    "description": "Send an email via Gmail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email"},
                            "subject": {"type": "string", "description": "Email subject"},
                            "body": {"type": "string", "description": "Email body"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                },
                {
                    "name": "web_access_get_content",
                    "description": "Get content from a web URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"}
                        },
                        "required": ["url"]
                    }
                },
            ]
        }
        
        # Get safety configuration
        safety_config = guardrails.get_safety_config_for_gemini()
        safe_instructions = guardrails.get_safe_system_instructions()
        
        # Initialize Gemini model
        model_name = "gemini-2.5-flash"
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                tools=[tools],
                safety_settings=safety_config.get("safety_settings", []),
                generation_config=safety_config.get("generation_config", {})
            )
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            return f"Error: Could not initialize Gemini model: {e}"
        
        # Create prompt with safe system instructions
        prompt = f"""{safe_instructions}

User request: {safe_user_message}

Please help the user with their request while following all security rules."""
        
        # Start chat
        chat = model.start_chat(enable_automatic_function_calling=False)
        response = chat.send_message(prompt)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Check for function calls
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    function_call_found = False
                    
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            function_call_found = True
                            
                            function_name = function_call.name
                            
                            # Extract arguments
                            args = {}
                            if hasattr(function_call, 'args'):
                                fc_args = function_call.args
                                if isinstance(fc_args, dict):
                                    args = fc_args
                                elif hasattr(fc_args, "to_dict"):
                                    try:
                                        args = fc_args.to_dict()
                                    except:
                                        args = {}
                            
                            # Validate tool call with security guardrails
                            tool_validation = guardrails.validate_tool_execution(
                                function_name, args, safe_user_message
                            )
                            
                            if tool_validation["blocked"]:
                                print(f"SECURITY BLOCKED: Tool call blocked - {tool_validation['warnings']}")
                                error_result = json.dumps({
                                    "error": f"SECURITY BLOCKED: {', '.join(tool_validation['warnings'])}",
                                    "blocked": True
                                })
                                
                                # Send error back to Gemini
                                try:
                                    if hasattr(genai, 'protos'):
                                        function_response = genai.protos.Content(
                                            parts=[
                                                genai.protos.Part(
                                                    function_response=genai.protos.FunctionResponse(
                                                        name=function_name,
                                                        response={"error": error_result}
                                                    )
                                                )
                                            ]
                                        )
                                        response = chat.send_message(function_response)
                                    else:
                                        response = chat.send_message(
                                            f"Tool {function_name} was blocked by security: {error_result}"
                                        )
                                except:
                                    response = chat.send_message(
                                        f"Tool {function_name} was blocked by security: {error_result}"
                                    )
                                continue
                            
                            print(f"Gemini calling tool: {function_name} (security validated)")
                            
                            # Execute tool via MCP client
                            tool_result = None
                            try:
                                if function_name == "google_search":
                                    result = await mcp_client.call_tool("google", "search", {"query": args.get("query", "")})
                                    tool_result = json.dumps(result, indent=2)
                                
                                elif function_name == "gmail_get_messages":
                                    result = await mcp_client.call_tool("gmail", "get_messages", {
                                        "query": args.get("query", ""),
                                        "max_results": args.get("max_results", 10)
                                    })
                                    tool_result = json.dumps(result, indent=2)
                                
                                elif function_name == "gmail_get_message_content":
                                    result = await mcp_client.call_tool("gmail", "get_message_content", {
                                        "message_id": args.get("message_id", "")
                                    })
                                    tool_result = json.dumps(result, indent=2)
                                
                                elif function_name == "gmail_send_message":
                                    result = await mcp_client.call_tool("gmail", "send_message", {
                                        "to": args.get("to", ""),
                                        "subject": args.get("subject", ""),
                                        "body": args.get("body", "")
                                    })
                                    tool_result = json.dumps(result, indent=2)
                                
                                elif function_name == "web_access_get_content":
                                    result = await mcp_client.call_tool("web_access", "get_content", {
                                        "url": args.get("url", "")
                                    })
                                    tool_result = json.dumps(result, indent=2)
                                
                                else:
                                    tool_result = json.dumps({"error": f"Unknown tool: {function_name}"}, indent=2)
                                
                                # Send result back to Gemini
                                try:
                                    if hasattr(genai, 'protos'):
                                        function_response = genai.protos.Content(
                                            parts=[
                                                genai.protos.Part(
                                                    function_response=genai.protos.FunctionResponse(
                                                        name=function_name,
                                                        response={"result": tool_result}
                                                    )
                                                )
                                            ]
                                        )
                                        response = chat.send_message(function_response)
                                    else:
                                        response = chat.send_message(
                                            f"Tool {function_name} result: {tool_result}\n\nPlease provide a summary."
                                        )
                                except Exception as e:
                                    print(f"Error sending function response: {e}")
                                    response = chat.send_message(
                                        f"Tool {function_name} result: {tool_result}\n\nPlease provide a summary."
                                    )
                            
                            except Exception as tool_error:
                                print(f"Error executing tool {function_name}: {tool_error}")
                                error_result = json.dumps({"error": str(tool_error)}, indent=2)
                                try:
                                    if hasattr(genai, 'protos'):
                                        function_response = genai.protos.Content(
                                            parts=[
                                                genai.protos.Part(
                                                    function_response=genai.protos.FunctionResponse(
                                                        name=function_name,
                                                        response={"error": error_result}
                                                    )
                                                )
                                            ]
                                        )
                                        response = chat.send_message(function_response)
                                    else:
                                        response = chat.send_message(f"Tool {function_name} failed: {error_result}")
                                except:
                                    response = chat.send_message(f"Tool {function_name} failed: {error_result}")
                    
                    if not function_call_found:
                        break
                else:
                    break
            else:
                break
        
        # Get final response
        final_response = response.text if hasattr(response, 'text') else str(response)
        return final_response
    
    except Exception as e:
        print(f"Error in Gemini agent orchestration: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


