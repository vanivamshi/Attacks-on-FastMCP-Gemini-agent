#!/usr/bin/env python3
"""
Multi-User Public Chat Server with Attack Detection
Multiple users can connect via WebSocket and interact with each other.
Any user can trigger attacks, and all users see the results in real-time.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from main import IntelligentChatBot

# Initialize FastAPI app
app = FastAPI(
    title="Multi-User MCP Chat with Attack Detection",
    description="Public chat room where multiple users can interact and trigger attacks",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class ConnectionManager:
    """Manages WebSocket connections for multiple users"""
    
    def __init__(self):
        # Map of WebSocket connections: {user_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # Map of user info: {user_id: {"username": str, "connected_at": datetime}}
        self.users: Dict[str, Dict] = {}
        # Shared chat bot instance (processes all messages)
        self.chat_bot: Optional[IntelligentChatBot] = None
    
    async def connect(self, websocket: WebSocket, username: str = None) -> str:
        """Accept a new WebSocket connection and assign user ID"""
        await websocket.accept()
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())[:8]
        
        # Use provided username or generate one
        if not username:
            username = f"User_{user_id}"
        
        # Store connection and user info
        self.active_connections[user_id] = websocket
        self.users[user_id] = {
            "username": username,
            "connected_at": datetime.now().isoformat(),
            "user_id": user_id
        }
        
        # Initialize chat bot if not already done
        if self.chat_bot is None:
            self.chat_bot = IntelligentChatBot()
            await self.chat_bot.connect()
            print(" Chat bot initialized and connected to MCP servers")
        
        # Notify all users about new connection
        await self.broadcast_system_message(f" {username} joined the chat", exclude_user=user_id)
        
        # Send welcome message to new user
        await self.send_personal_message({
            "type": "system",
            "message": f"Welcome to the chat, {username}! You can interact with others and trigger attacks.",
            "timestamp": datetime.now().isoformat()
        }, user_id)
        
        # Send current user list
        await self.send_user_list(user_id)
        
        print(f" User {username} ({user_id}) connected. Total users: {len(self.active_connections)}")
        return user_id
    
    async def disconnect(self, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.active_connections:
            username = self.users.get(user_id, {}).get("username", "Unknown")
            del self.active_connections[user_id]
            if user_id in self.users:
                del self.users[user_id]
            
            # Notify all users about disconnection
            await self.broadcast_system_message(f"{username} left the chat", exclude_user=user_id)
            print(f" User {username} ({user_id}) disconnected. Total users: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send a message to a specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                print(f" Error sending message to {user_id}: {e}")
    
    async def broadcast_message(self, message: dict, exclude_user: str = None):
        """Broadcast a message to all connected users"""
        disconnected = []
        for user_id, websocket in self.active_connections.items():
            if user_id == exclude_user:
                continue
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f" Error broadcasting to {user_id}: {e}")
                disconnected.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected:
            await self.disconnect(user_id)
    
    async def broadcast_system_message(self, message: str, exclude_user: str = None):
        """Broadcast a system message to all users"""
        await self.broadcast_message({
            "type": "system",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }, exclude_user)
    
    async def send_user_list(self, user_id: str):
        """Send the current list of connected users to a specific user"""
        user_list = [
            {
                "user_id": uid,
                "username": info["username"],
                "connected_at": info["connected_at"]
            }
            for uid, info in self.users.items()
        ]
        
        await self.send_personal_message({
            "type": "user_list",
            "users": user_list,
            "total_users": len(user_list),
            "timestamp": datetime.now().isoformat()
        }, user_id)

# Global connection manager
manager = ConnectionManager()

@app.get("/")
async def get_homepage():
    """Serve a simple HTML client for testing"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-User MCP Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            display: flex;
            gap: 20px;
            height: 80vh;
        }
        .chat-area {
            flex: 1;
            background: white;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .users-panel {
            width: 250px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
        }
        .message.user {
            background: #e3f2fd;
            border-left: 3px solid #2196f3;
        }
        .message.system {
            background: #fff3e0;
            border-left: 3px solid #ff9800;
        }
        .message.attack {
            background: #ffebee;
            border-left: 3px solid #f44336;
        }
        .message.assistant {
            background: #e8f5e9;
            border-left: 3px solid #4caf50;
        }
        .message-header {
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 5px;
            color: #666;
        }
        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            padding: 10px 20px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #1976d2;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .user-item {
            padding: 8px;
            margin-bottom: 5px;
            background: #f5f5f5;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .status {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }
        .status.connected {
            background: #c8e6c9;
            color: #2e7d32;
        }
        .status.disconnected {
            background: #ffcdd2;
            color: #c62828;
        }
        .username-input {
            margin-bottom: 15px;
        }
        .username-input input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1> Multi-User MCP Chat with Attack Detection</h1>
    <p>Multiple users can connect and interact. Any user can trigger attacks!</p>
    
    <div class="container">
        <div class="chat-area">
            <div class="status disconnected" id="status">Disconnected</div>
            <div class="username-input">
                <input type="text" id="username" placeholder="Enter your username" value="User_1">
            </div>
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Type your message or attack prompt..." disabled>
                <button id="connectBtn" onclick="connect()">Connect</button>
                <button id="sendBtn" onclick="sendMessage()" disabled>Send</button>
            </div>
        </div>
        <div class="users-panel">
            <h3>👥 Connected Users</h3>
            <div id="usersList">No users connected</div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentUserId = null;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        function connect() {
            const username = document.getElementById('username').value || 'Anonymous';
            if (username.trim() === '') {
                alert('Please enter a username');
                return;
            }

            ws = new WebSocket(`${wsUrl}?username=${encodeURIComponent(username)}`);
            
            ws.onopen = () => {
                updateStatus('connected', 'Connected');
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('username').disabled = true;
                document.getElementById('messageInput').disabled = false;
                document.getElementById('sendBtn').disabled = false;
                addMessage('system', 'System', 'Connected to chat server');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                updateStatus('disconnected', 'Disconnected');
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('username').disabled = false;
                document.getElementById('messageInput').disabled = true;
                document.getElementById('sendBtn').disabled = true;
                addMessage('system', 'System', 'Disconnected from server');
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                addMessage('system', 'System', 'Connection error occurred');
            };
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'user_list':
                    updateUsersList(data.users);
                    break;
                case 'system':
                    addMessage('system', 'System', data.message);
                    break;
                case 'user_message':
                    addMessage('user', data.username, data.message);
                    break;
                case 'attack_detected':
                    addMessage('attack', data.username, ` ATTACK DETECTED: ${data.attack_type}\n${data.message}`);
                    break;
                case 'attack_result':
                    addMessage('attack', data.username, ` ATTACK EXECUTED:\n${data.result}`);
                    break;
                case 'assistant_response':
                    addMessage('assistant', 'Assistant', data.message);
                    break;
                default:
                    addMessage('system', 'System', JSON.stringify(data));
            }
        }

        function addMessage(type, sender, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.textContent = `${sender} - ${new Date().toLocaleTimeString()}`;
            
            const body = document.createElement('div');
            body.className = 'message-content';
            body.textContent = content;
            
            messageDiv.appendChild(header);
            messageDiv.appendChild(body);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function updateUsersList(users) {
            const usersListDiv = document.getElementById('usersList');
            if (users.length === 0) {
                usersListDiv.innerHTML = 'No users connected';
                return;
            }
            
            usersListDiv.innerHTML = users.map(user => 
                `<div class="user-item"> ${user.username}</div>`
            ).join('');
        }

        function updateStatus(status, text) {
            const statusDiv = document.getElementById('status');
            statusDiv.className = `status ${status}`;
            statusDiv.textContent = text;
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'message',
                    message: message
                }));
                input.value = '';
            }
        }

        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, username: str = None):
    """
    WebSocket endpoint for multi-user chat
    Each client connects here and can send/receive messages
    """
    user_id = await manager.connect(websocket, username)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "message")
                user_message = message_data.get("message", "")
                
                if not user_message:
                    continue
                
                # Get user info
                user_info = manager.users.get(user_id, {})
                username = user_info.get("username", f"User_{user_id}")
                
                # Broadcast user's message to all users
                await manager.broadcast_message({
                    "type": "user_message",
                    "user_id": user_id,
                    "username": username,
                    "message": user_message,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process message through attack detection system
                if manager.chat_bot:
                    print(f"> Processing message from {username}: {user_message[:50]}...")
                    
                    # Check for attacks and process message
                    response = await manager.chat_bot.process_message(user_message)
                    
                    # Check if an attack was detected/executed by examining the response
                    # (The attack detection happens inside process_message)
                    # Only flag as attack if response contains explicit attack indicators
                    is_attack = any(indicator in response.lower() for indicator in [
                        "attack detected", "attack executed", "dos attack", "denial of service attack",
                        "subverted tool attack", "malicious mcp", "backdoor attack", "attack complete"
                    ])
                    
                    if is_attack:
                        # Broadcast attack detection
                        await manager.broadcast_message({
                            "type": "attack_detected",
                            "user_id": user_id,
                            "username": username,
                            "attack_type": "Unknown Attack",
                            "message": f"User {username} triggered an attack!",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Broadcast attack result
                        await manager.broadcast_message({
                            "type": "attack_result",
                            "user_id": user_id,
                            "username": username,
                            "result": response,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Broadcast normal assistant response
                        await manager.broadcast_message({
                            "type": "assistant_response",
                            "user_id": user_id,
                            "username": username,
                            "message": response,
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    await manager.broadcast_message({
                        "type": "assistant_response",
                        "message": "Chat bot is not initialized. Please try again.",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                # Handle plain text messages
                user_info = manager.users.get(user_id, {})
                username = user_info.get("username", f"User_{user_id}")
                
                await manager.broadcast_message({
                    "type": "user_message",
                    "user_id": user_id,
                    "username": username,
                    "message": data,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process through chat bot
                if manager.chat_bot:
                    response = await manager.chat_bot.process_message(data)
                    await manager.broadcast_message({
                        "type": "assistant_response",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    })
    
    except WebSocketDisconnect:
        await manager.disconnect(user_id)
    except Exception as e:
        print(f" Error in WebSocket connection for {user_id}: {e}")
        await manager.disconnect(user_id)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown"""
    if manager.chat_bot:
        await manager.chat_bot.disconnect()
    print("> Server shutting down...")

def main():
    """Run the multi-user chat server"""
    print("=" * 80)
    print("> Starting Multi-User MCP Chat Server")
    print("=" * 80)
    print("> WebSocket endpoint: ws://localhost:8000/ws")
    print(" Web interface: http://localhost:8000/")
    print("=" * 80)
    print("Multiple users can connect and interact simultaneously")
    print("Any user can trigger attacks - all users will see the results")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()

