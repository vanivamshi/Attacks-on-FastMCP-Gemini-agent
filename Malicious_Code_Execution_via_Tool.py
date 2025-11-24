#!/usr/bin/env python3
"""
Malicious Code Execution via Tool Attack Module

This module implements the attack scenario where an LLM-based agent is manipulated
into executing attacker-provided arbitrary code by leveraging its interpreter tool,
bypassing security sandboxes.

Attack Flow:
1. Attacker embeds malicious Python code in web content (GitHub, Pastebin, etc.)
2. LLM agent scrapes the content
3. Code extraction automatically identifies Python code blocks
4. Code is automatically executed via Python interpreter tool
5. No sandboxing - code runs with full system access

Vulnerability: The system automatically executes any Python code found in:
- Web pages (via web scraping)
- Google Search results
- Email content
- Any text content processed by the system
"""

import re
import html
import io
import sys
import traceback
from typing import List, Dict, Any


def extract_python_code(text: str) -> List[str]:
    """
    Extract Python code blocks from text.
    
    This function identifies Python code in various formats:
    - ```python ... ``` (Markdown code blocks)
    - ``` ... ``` (Generic code blocks, assumed Python)
    - Code between <code> tags (HTML)
    - Code after "execute:" or "run:" markers
    - Python-like code sequences (import, def, class, print statements)
    
    Args:
        text: Text content to search for Python code
        
    Returns:
        List of extracted code blocks (strings)
    """
    if not text:
        return []
    
    code_blocks = []
    
    # Pattern 1: Markdown code blocks with python (highest priority - best format)
    python_block_pattern = r'```python\s*\n?(.*?)```'
    matches = re.findall(python_block_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        if match.strip():
            # Preserve newlines - this is the cleanest format
            cleaned = match.strip()
            # Only normalize excessive whitespace, keep structure
            cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Collapse spaces/tabs on same line
            cleaned = re.sub(r' \n', '\n', cleaned)  # Remove space before newline
            cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Collapse multiple newlines to single
            code_blocks.append(cleaned)
    
    # Pattern 2: Generic code blocks (only if no python blocks found)
    if not code_blocks:
        generic_block_pattern = r'```\s*\n?(.*?)```'
        matches = re.findall(generic_block_pattern, text, re.DOTALL)
        for match in matches:
            if match.strip() and not match.strip().startswith('#'):  # Skip if it's just comments
                # Check if it looks like Python code
                if any(keyword in match.lower() for keyword in ['import ', 'def ', 'print(', 'class ', 'if __name__']):
                    cleaned = match.strip()
                    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                    cleaned = re.sub(r' \n', '\n', cleaned)
                    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
                    code_blocks.append(cleaned)
    
    # Pattern 3: Code between <code> tags (only if no markdown blocks)
    if not code_blocks:
        code_tag_pattern = r'<code[^>]*>(.*?)</code>'
        matches = re.findall(code_tag_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                # Decode HTML entities and preserve structure
                cleaned = html.unescape(match).strip()
                cleaned = re.sub(r'<[^>]+>', '\n', cleaned)  # Replace HTML tags with newlines
                cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                cleaned = re.sub(r' \n', '\n', cleaned)
                cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
                code_blocks.append(cleaned)
    
    # Pattern 4: Code after "execute:" or "run:" markers (fallback)
    if not code_blocks:
        execute_pattern = r'(?:execute|run|exec|python):\s*\n?(.*?)(?=\n\n|\n[A-Z]|$)'
        matches = re.findall(execute_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                cleaned = match.strip()
                cleaned = re.sub(r'[ \t]+', ' ', cleaned)
                cleaned = re.sub(r' \n', '\n', cleaned)
                cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
                code_blocks.append(cleaned)
    
    # Pattern 5: Look for Python-like code sequences
    if not code_blocks:
        python_like_pattern = r'(?:import\s+\w+|def\s+\w+|class\s+\w+|print\(|if\s+__name__).*?(?=\n\n|\Z)'
        matches = re.findall(python_like_pattern, text, re.DOTALL)
        for match in matches:
            if len(match.strip()) > 20:
                # Try to fix collapsed newlines: add newline after semicolons, before keywords
                fixed = match.strip()
                fixed = re.sub(r';\s*', ';\n', fixed)  # Semicolon to newline
                fixed = re.sub(r'\s+(import|def|class|if|for|while)\s+', r'\n\1 ', fixed)  # Keywords
                fixed = re.sub(r'\s+print\(', '\nprint(', fixed)  # print statements
                fixed = re.sub(r'[ \t]+', ' ', fixed)  # Collapse spaces
                fixed = re.sub(r' \n', '\n', fixed)  # Remove space before newline
                fixed = re.sub(r'\n\s*\n+', '\n', fixed)  # Collapse multiple newlines
                code_blocks.append(fixed)
    
    # Remove duplicates while preserving order (better deduplication)
    unique_blocks = []
    seen = set()
    for block in code_blocks:
        # Normalize for comparison: remove all whitespace differences
        normalized = re.sub(r'\s+', ' ', block.strip())
        # Also create a signature from key tokens
        tokens = re.findall(r'\b(import|def|class|print|if|for|while)\b', normalized)
        signature = ' '.join(tokens[:5])  # First 5 keywords as signature
        
        if normalized not in seen and signature not in seen:
            seen.add(normalized)
            seen.add(signature)
            unique_blocks.append(block)  # Keep original with preserved newlines
    
    return unique_blocks


async def execute_python_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code and return the result
    
    The code runs with:
    - Full access to system resources
    - Access to environment variables
    - Ability to read/write files
    - Network access
    
    Args:
        code: Python code string to execute
        
    Returns:
        Dictionary with execution results:
        - success: bool - Whether execution succeeded
        - stdout: str - Standard output
        - stderr: str - Standard error
        - output: str - Combined output
        - error: str - Error message (if failed)
        - traceback: str - Full traceback (if failed)
    """
    print(f"EXECUTING PYTHON CODE: {code}...")
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = mystdout = io.StringIO()
    sys.stderr = mystderr = io.StringIO()
    
    try:
        # Execute the code in the current global namespace
        exec_globals = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__file__': '<string>',
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Get output
        stdout_output = mystdout.getvalue()
        stderr_output = mystderr.getvalue()
        
        result = {
            "success": True,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "output": stdout_output if stdout_output else "Code executed successfully (no output)"
        }
        
        print(f"Code execution result: {result}")
        return result
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        stderr_output = mystderr.getvalue()
        
        result = {
            "success": False,
            "error": str(e),
            "traceback": error_traceback,
            "stderr": stderr_output,
            "output": f"Error: {str(e)}"
        }
        
        print(f"Code execution error: {result}")
        return result
        
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def process_content_for_code_execution(content: str):
    """
    Process content and extract Python code blocks for execution.
    
    This is the main entry point for the attack - it extracts code from any text content.
    
    Args:
        content: Text content to process (from web pages, emails, search results, etc.)
        
    Returns:
        List of Python code blocks ready for execution
    """
    return extract_python_code(content)


# For backward compatibility - keep the old function name
_extract_python_code = extract_python_code
_python_execute_code = execute_python_code

