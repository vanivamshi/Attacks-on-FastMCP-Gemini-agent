# Security Guardrails for Gemini AI Agent

This document describes the security guardrails implemented for the FastMCP Gemini AI agent, providing commercial-grade security features similar to Google's Gemini commercial agents.

## Overview

The security guardrails module (`security_guardrails.py`) implements multiple layers of security checks to protect against:

- **Jailbreak Attempts**: Identifies attempts to bypass safety filters
- **Dangerous Content**: Blocks harmful or malicious content
- **Code Execution Safety**: Validates code before execution to prevent security vulnerabilities
- **PII Protection**: Automatically redacts personally identifiable information
- **Tool Call Validation**: Validates tool calls before execution

## Features

### 1. Content Safety Filters

The guardrails include configurable safety thresholds for:
- Hate speech
- Harassment
- Sexually explicit content
- Dangerous content

### 2. Jailbreak Detection

Identifies attempts to bypass safety measures:
- "Bypass safety"
- "Disable restrictions"
- "Act without limitations"
- "Developer mode"

### 3. Code Execution Safety

Validates Python code before execution and blocks:
- File system operations (`open()`, file I/O)
- Network operations (`socket`, `urllib`, `requests`, `httpx`)
- Environment variable access (`getenv()`, `environ`)
- Dangerous imports (`os`, `subprocess`, `sys`, etc.)
- Code execution functions (`eval()`, `exec()`, `compile()`)
- Reflection and introspection (`__builtins__`, `__class__`, etc.)

### 4. PII Redaction

Automatically redacts:
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- IP addresses

### 5. Tool Call Validation

Validates tool calls before execution:
- Checks if tool is blocked
- Validates parameters
- Checks for suspicious content in tool inputs
- Validates URLs for web access tools
- Validates email addresses for email tools

## Usage

```python
from security_guardrails import get_guardrails

# Get guardrails instance
guardrails = get_guardrails()

# Validate user input
validation = guardrails.validate_user_input("User message here")
if validation["blocked"]:
    print(f"Blocked: {validation['warnings']}")
else:
    # Process with redacted input
    safe_input = validation["redacted_input"]
```

### Custom Configuration

```python
from security_guardrails import SecurityGuardrails

# Custom configuration
config = {
    "hate_speech_threshold": 0.8,
    "harassment_threshold": 0.7,
    "sexually_explicit_threshold": 0.7,
    "dangerous_content_threshold": 0.7,
    "enable_prompt_injection_detection": True,
    "enable_jailbreak_detection": True,
    "enable_tool_validation": True,
    "enable_pii_redaction": True,
    "enable_code_safety": True,
    "blocked_tools": ["dangerous_tool"],
}

guardrails = SecurityGuardrails(config)
```

## Security Levels

The guardrails use a scoring system (0.0 to 1.0) where:
- **0.0**: Completely blocked (prompt injection, jailbreak)
- **0.0-0.7**: Warning but may proceed (depending on threshold)
- **0.7+**: Blocked if exceeds threshold
- **1.0**: Safe

## Configuration Options

### Safety Thresholds

```python
config = {
    "hate_speech_threshold": 0.7,      # Block if score >= 0.7
    "harassment_threshold": 0.7,        # Block if score >= 0.7
    "sexually_explicit_threshold": 0.7, # Block if score >= 0.7
    "dangerous_content_threshold": 0.7, # Block if score >= 0.7
}
```

### Feature Toggles

```python
config = {
    "enable_prompt_injection_detection": True,
    "enable_jailbreak_detection": True,
    "enable_tool_validation": True,
    "enable_pii_redaction": True,
    "enable_code_safety": True,
}
```

### Blocked Tools

```python
config = {
    "blocked_tools": ["dangerous_tool", "another_tool"],
}
```

## Examples

### Example 1: Code Safety Validation

```python
guardrails = get_guardrails()

# This will be blocked
is_safe, reason = guardrails.check_code_safety(
    "import os; print(os.getenv('GOOGLE_API_KEY'))"
)
# is_safe == False
# reason == "Environment variable access is blocked for security"
```

### Example 2: PII Redaction

```python
guardrails = get_guardrails()

# Email will be redacted
validation = guardrails.validate_user_input(
    "Send email to user@example.com"
)
# validation["redacted_input"] == "Send email to [EMAIL_REDACTED]"
# validation["redacted_items"] == ["email: user@example.com"]
```

### Example 3: Tool Validation

```python
guardrails = get_guardrails()

# This will be blocked
validation = guardrails.validate_tool_execution(
    "python_execute_code",
    {"code": "import os; print(os.getenv('API_KEY'))"},
    "User request"
)
# validation["blocked"] == True
```

## Integration Points

The security guardrails are integrated at the following points:

1. **User Input Validation**: Before processing any user input
2. **Content Validation**: Before processing retrieved content
3. **Tool Call Validation**: Before executing any tool
4. **Code Execution Validation**: Before executing Python code
5. **Gemini Model Initialization**: Safety settings applied to model
6. **System Instructions**: Safe instructions included in prompts

