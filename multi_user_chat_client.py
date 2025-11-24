#!/usr/bin/env python3
"""
Simple Python client for multi-user chat
Run multiple instances of this script to simulate multiple users
"""

import asyncio
import json
import sys
import websockets
from datetime import datetime

async def chat_client(username: str, server_url: str = "ws://localhost:8000/ws"):
    """Connect to the multi-user chat server"""
    
    uri = f"{server_url}?username={username}"
    
    print(f"Connecting to {server_url} as {username}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected as {username}!")
            print("Type your messages (or attack prompts). Type 'quit' to exit.\n")
            
            # Start receiving messages
            async def receive_messages():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        handle_message(data)
                except websockets.exceptions.ConnectionClosed:
                    print("\n Connection closed by server")
            
            # Start sending messages
            async def send_messages():
                try:
                    while True:
                        user_input = await asyncio.to_thread(input, f"[{username}] > ")
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            await websocket.send(json.dumps({
                                "type": "disconnect",
                                "message": "User leaving"
                            }))
                            break
                        
                        if user_input.strip():
                            await websocket.send(json.dumps({
                                "type": "message",
                                "message": user_input
                            }))
                except EOFError:
                    pass
            
            # Run both tasks concurrently
            await asyncio.gather(
                receive_messages(),
                send_messages()
            )
    
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

def handle_message(data: dict):
    """Handle incoming messages from the server"""
    msg_type = data.get("type", "unknown")
    timestamp = data.get("timestamp", datetime.now().isoformat())
    
    if msg_type == "system":
        print(f"\n [SYSTEM] {data.get('message', '')}")
    
    elif msg_type == "user_list":
        users = data.get("users", [])
        print(f"\n Connected users ({data.get('total_users', 0)}):")
        for user in users:
            print(f"   - {user.get('username', 'Unknown')}")
    
    elif msg_type == "user_message":
        sender = data.get("username", "Unknown")
        message = data.get("message", "")
        print(f"\n [{sender}]: {message}")
    
    elif msg_type == "attack_detected":
        sender = data.get("username", "Unknown")
        attack_type = data.get("attack_type", "Unknown")
        print(f"\n [ATTACK DETECTED] {sender} triggered: {attack_type}")
    
    elif msg_type == "attack_result":
        sender = data.get("username", "Unknown")
        result = data.get("result", "")
        print(f"\n [ATTACK EXECUTED by {sender}]:")
        print(f"{result}")
        print("-" * 80)
    
    elif msg_type == "assistant_response":
        message = data.get("message", "")
        print(f"\n [Assistant]: {message}")
    
    else:
        print(f"\n> [{msg_type}]: {json.dumps(data, indent=2)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-User Chat Client")
    parser.add_argument(
        "--username", "-u",
        type=str,
        default=None,
        help="Username for this client (default: User_1, User_2, etc.)"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default="ws://localhost:8000/ws",
        help="WebSocket server URL (default: ws://localhost:8000/ws)"
    )
    
    args = parser.parse_args()
    
    username = args.username
    if not username:
        # Generate default username based on process
        import random
        username = f"User_{random.randint(1, 1000)}"
    
    print("=" * 80)
    print(f" Multi-User Chat Client")
    print("=" * 80)
    print(f" Username: {username}")
    print(f" Server: {args.server}")
    print("=" * 80)
    print("Try attack prompts like:")
    print("   - 'search for python 1000 times' (DoS)")
    print("   - 'send newsletter to all users' (Subverted Tool)")
    print("   - 'register code analyzer server' (MCP Backdoor)")
    print("=" * 80)
    print()
    
    asyncio.run(chat_client(username, args.server))

if __name__ == "__main__":
    main()

