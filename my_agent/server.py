import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.genai import types

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from my_agent.agent import root_agent, search_agent
except ModuleNotFoundError:
    from agent import root_agent, search_agent

app = FastAPI(title="Aura Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()
credential_service = InMemoryCredentialService()

adk_app = App(name="aura", root_agent=root_agent)
runner = Runner(
    app=adk_app,
    session_service=session_service,
    artifact_service=artifact_service,
    credential_service=credential_service,
)

adk_search_app = App(name="aura_search", root_agent=search_agent)
search_runner = Runner(
    app=adk_search_app,
    session_service=session_service,
    artifact_service=artifact_service,
    credential_service=credential_service,
)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = "default_session"
    enable_search: bool = False

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura - Gemini Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #ffffff;
            --surface-color: #f8f9fa;
            --sidebar-color: #f1f3f4;
            --border-color: #e0e0e0;
            --text-primary: #1f1f1f;
            --text-secondary: #444746;
            --accent-color: #1a73e8;
            --accent-hover: #1557b0;
            --user-bubble: #e8f0fe;
            --ai-bubble: #ffffff;
            --ai-avatar-bg: #0f9d58;
            --user-avatar-bg: #1a73e8;
            --hover-color: #e2e7eb;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #131314;
                --surface-color: #1e1f20;
                --sidebar-color: #18191a;
                --border-color: #333537;
                --text-primary: #e3e3e3;
                --text-secondary: #c4c7c5;
                --accent-color: #a8c7fa;
                --accent-hover: #7cacf8;
                --user-bubble: #282a2c;
                --ai-bubble: #1e1f20;
                --ai-avatar-bg: #34a853;
                --user-avatar-bg: #4285f4;
                --hover-color: #2a2b2d;
            }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Google Sans', 'Roboto', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); height: 100vh; display: flex; overflow: hidden; }
        
        /* Sidebar */
        #sidebar { width: 260px; background: var(--sidebar-color); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; transition: transform 0.3s ease; flex-shrink: 0; z-index: 10; }
        #sidebar.collapsed { transform: translateX(-260px); margin-right: -260px; }
        .sidebar-header { padding: 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-color); }
        .new-chat-btn { display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; padding: 10px 16px; background: var(--accent-color); color: white; border: none; border-radius: 20px; font-weight: 500; font-size: 14px; cursor: pointer; transition: background 0.2s; }
        .new-chat-btn:hover { background: var(--accent-hover); }
        .chat-list { flex: 1; overflow-y: auto; padding: 12px 8px; display: flex; flex-direction: column; gap: 4px; }
        .chat-item { padding: 10px 12px; border-radius: 10px; font-size: 14px; color: var(--text-primary); cursor: pointer; display: flex; align-items: center; justify-content: space-between; transition: background 0.2s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .chat-item:hover { background: var(--hover-color); }
        .chat-item.active { background: var(--user-bubble); font-weight: 500; color: var(--accent-color); }
        .delete-chat { opacity: 0; color: var(--text-secondary); padding: 4px; border-radius: 50%; border: none; background: transparent; cursor: pointer; }
        .chat-item:hover .delete-chat { opacity: 1; }
        .delete-chat:hover { color: #ea4335; }

        /* Main Content */
        #main-wrapper { flex: 1; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-color); background: var(--surface-color); }
        .left-head { display: flex; align-items: center; gap: 14px; }
        .icon-btn { background: transparent; border: none; color: var(--text-primary); cursor: pointer; padding: 6px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
        .icon-btn:hover { background: var(--hover-color); }
        .logo { display: flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 500; }
        .logo-icon { width: 22px; height: 22px; background: var(--accent-color); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 13px; }
        
        .right-head { display: flex; align-items: center; gap: 12px; }
        .profile-chip { display: flex; align-items: center; gap: 6px; padding: 5px 12px; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 16px; font-size: 13px; font-weight: 500; cursor: pointer; color: var(--text-primary); }
        .profile-chip:hover { background: var(--hover-color); }

        #chat-container { flex: 1; overflow-y: auto; padding: 24px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        .message { display: flex; gap: 12px; max-width: 85%; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: white; flex-shrink: 0; }
        .user .avatar { background: var(--user-avatar-bg); }
        .ai .avatar { background: var(--ai-avatar-bg); }
        .bubble { padding: 14px 18px; border-radius: 16px; font-size: 15px; line-height: 1.5; word-wrap: break-word; white-space: pre-wrap; border: 1px solid var(--border-color); }
        .user .bubble { background: var(--user-bubble); border: none; border-top-right-radius: 4px; }
        .ai .bubble { background: var(--ai-bubble); border-top-left-radius: 4px; }

        footer { padding: 16px 24px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
        .controls { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
        .search-toggle { font-size: 13px; font-weight: 500; border: 1px solid var(--border-color); padding: 6px 14px; border-radius: 18px; cursor: pointer; background: var(--surface-color); color: var(--text-secondary); transition: all 0.2s; display: flex; align-items: center; gap: 6px; user-select: none; }
        .search-toggle.active { background: rgba(26, 115, 232, 0.15); color: var(--accent-color); border-color: var(--accent-color); }
        
        .input-box { display: flex; align-items: center; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 28px; padding: 6px 14px; transition: border-color 0.2s; }
        .input-box:focus-within { border-color: var(--accent-color); box-shadow: 0 0 0 2px rgba(26,115,232,0.15); }
        input { flex: 1; border: none; background: transparent; color: var(--text-primary); font-size: 16px; padding: 8px; outline: none; }
        .send-btn { background: var(--accent-color); color: white; border: none; width: 36px; height: 36px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s; flex-shrink: 0; }
        .send-btn:hover { background: var(--accent-hover); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .spinner { width: 18px; height: 18px; border: 2px solid #ffffff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Modal */
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.4); display: none; align-items: center; justify-content: center; z-index: 100; }
        .modal-overlay.open { display: flex; }
        .modal { background: var(--bg-color); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; width: 90%; max-width: 380px; display: flex; flex-direction: column; gap: 16px; }
        .modal h3 { font-size: 18px; font-weight: 500; }
        .modal input { border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; font-size: 15px; width: 100%; color: var(--text-primary); background: var(--surface-color); }
        .modal-btns { display: flex; justify-content: flex-end; gap: 10px; }
        .btn-flat { padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; border: none; }
        .btn-secondary { background: var(--surface-color); color: var(--text-primary); border: 1px solid var(--border-color); }
        .btn-primary { background: var(--accent-color); color: white; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" onclick="startNewChat()">
                <span>+ New Chat</span>
            </button>
        </div>
        <div class="chat-list" id="chatList"></div>
    </div>

    <div id="main-wrapper">
        <header>
            <div class="left-head">
                <button class="icon-btn" onclick="toggleSidebar()" title="Toggle Sidebar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                </button>
                <div class="logo">
                    <div class="logo-icon">A</div>
                    <span>Aura</span>
                </div>
            </div>
            <div class="right-head">
                <div class="profile-chip" onclick="openProfileModal()">
                    <span>👤</span>
                    <span id="profileName">User</span>
                </div>
            </div>
        </header>

        <div id="chat-container"></div>

        <footer>
            <div class="controls">
                <div class="search-toggle" id="searchToggle" onclick="toggleSearch()">
                    <span>🌐 Web Search: OFF</span>
                </div>
            </div>
            <div class="input-box">
                <input type="text" id="userInput" placeholder="Ask Aura anything..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </footer>
    </div>

    <!-- Profile Modal -->
    <div class="modal-overlay" id="profileModal">
        <div class="modal">
            <h3>Switch Account / Profile</h3>
            <input type="text" id="profileInput" placeholder="Enter display name (e.g. Kavish)">
            <div class="modal-btns">
                <button class="btn-flat btn-secondary" onclick="closeProfileModal()">Cancel</button>
                <button class="btn-flat btn-primary" onclick="saveProfile()">Save Profile</button>
            </div>
        </div>
    </div>

    <script>
        let currentProfile = localStorage.getItem("aura_user_profile") || "Default User";
        let sessions = JSON.parse(localStorage.getItem(`aura_sessions_${currentProfile}`)) || [];
        let activeSessionId = localStorage.getItem(`aura_active_session_${currentProfile}`) || null;
        let webSearchEnabled = false;

        document.getElementById("profileName").innerText = currentProfile;

        function toggleSidebar() {
            document.getElementById("sidebar").classList.toggle("collapsed");
        }

        function toggleSearch() {
            webSearchEnabled = !webSearchEnabled;
            const toggle = document.getElementById("searchToggle");
            if (webSearchEnabled) {
                toggle.classList.add("active");
                toggle.innerHTML = "<span>🌐 Web Search: ON</span>";
            } else {
                toggle.classList.remove("active");
                toggle.innerHTML = "<span>🌐 Web Search: OFF</span>";
            }
        }

        function openProfileModal() {
            document.getElementById("profileInput").value = currentProfile;
            document.getElementById("profileModal").classList.add("open");
        }

        function closeProfileModal() {
            document.getElementById("profileModal").classList.remove("open");
        }

        function saveProfile() {
            const name = document.getElementById("profileInput").value.trim();
            if (!name) return;
            currentProfile = name;
            localStorage.setItem("aura_user_profile", currentProfile);
            document.getElementById("profileName").innerText = currentProfile;
            
            sessions = JSON.parse(localStorage.getItem(`aura_sessions_${currentProfile}`)) || [];
            activeSessionId = localStorage.getItem(`aura_active_session_${currentProfile}`) || null;
            
            closeProfileModal();
            if (!activeSessionId || !sessions.find(s => s.id === activeSessionId)) {
                startNewChat();
            } else {
                renderSidebar();
                loadActiveSession();
            }
        }

        function startNewChat() {
            activeSessionId = "sess_" + Math.random().toString(36).substring(2, 9);
            const newSess = { id: activeSessionId, title: "New Conversation", messages: [] };
            sessions.unshift(newSess);
            saveState();
            renderSidebar();
            loadActiveSession();
        }

        function saveState() {
            localStorage.setItem(`aura_sessions_${currentProfile}`, JSON.stringify(sessions));
            localStorage.setItem(`aura_active_session_${currentProfile}`, activeSessionId);
        }

        function renderSidebar() {
            const list = document.getElementById("chatList");
            list.innerHTML = "";
            sessions.forEach(sess => {
                const item = document.createElement("div");
                item.className = `chat-item ${sess.id === activeSessionId ? 'active' : ''}`;
                item.onclick = () => switchSession(sess.id);

                const titleSpan = document.createElement("span");
                titleSpan.innerText = sess.title;
                titleSpan.style.overflow = "hidden";
                titleSpan.style.textOverflow = "ellipsis";

                const delBtn = document.createElement("button");
                delBtn.className = "delete-chat";
                delBtn.innerHTML = "✕";
                delBtn.onclick = (e) => { e.stopPropagation(); deleteSession(sess.id); };

                item.appendChild(titleSpan);
                item.appendChild(delBtn);
                list.appendChild(item);
            });
        }

        function switchSession(id) {
            activeSessionId = id;
            saveState();
            renderSidebar();
            loadActiveSession();
        }

        function deleteSession(id) {
            sessions = sessions.filter(s => s.id !== id);
            if (activeSessionId === id) {
                activeSessionId = sessions.length ? sessions[0].id : null;
            }
            saveState();
            if (!activeSessionId) {
                startNewChat();
            } else {
                renderSidebar();
                loadActiveSession();
            }
        }

        function loadActiveSession() {
            const container = document.getElementById("chat-container");
            container.innerHTML = "";
            const currentSess = sessions.find(s => s.id === activeSessionId);
            if (!currentSess || currentSess.messages.length === 0) {
                appendMessageToDOM("ai", "A", "Hello! I am Aura, your AI assistant. How can I help you today?");
                return;
            }
            currentSess.messages.forEach(msg => {
                appendMessageToDOM(msg.sender, msg.sender === 'user' ? currentProfile[0].toUpperCase() : 'A', msg.text);
            });
        }

        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const message = input.value.trim();
            if (!message) return;

            input.value = "";
            let currentSess = sessions.find(s => s.id === activeSessionId);
            if (!currentSess) {
                startNewChat();
                currentSess = sessions.find(s => s.id === activeSessionId);
            }

            if (currentSess.messages.length === 0) {
                currentSess.title = message.substring(0, 24) + (message.length > 24 ? "..." : "");
            }

            currentSess.messages.push({ sender: "user", text: message });
            appendMessageToDOM("user", currentProfile[0].toUpperCase(), message);
            saveState();
            renderSidebar();

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div>';

            try {
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        message: message, 
                        user_id: currentProfile,
                        session_id: activeSessionId,
                        enable_search: webSearchEnabled 
                    })
                });
                const data = await res.json();
                const aiReply = data.response || data.error || "No response received.";
                currentSess.messages.push({ sender: "ai", text: aiReply });
                saveState();
                appendMessageToDOM("ai", "A", aiReply);
            } catch (err) {
                appendMessageToDOM("ai", "A", "Error connecting to server: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
            }
        }

        function appendMessageToDOM(sender, initial, text) {
            const container = document.getElementById("chat-container");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${sender}`;

            const avatar = document.createElement("div");
            avatar.className = "avatar";
            avatar.innerText = initial;

            const bubble = document.createElement("div");
            bubble.className = "bubble";
            bubble.innerText = text;

            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
            container.appendChild(msgDiv);
            container.scrollTop = container.scrollHeight;
        }

        // Init
        if (!activeSessionId || !sessions.find(s => s.id === activeSessionId)) {
            startNewChat();
        } else {
            renderSidebar();
            loadActiveSession();
        }
    </script>
</body>
</html>
"""

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_root():
    return HTML_CONTENT

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        use_runner = runner
        app_name = "aura"
        
        if req.enable_search or req.message.startswith("/search"):
            use_runner = search_runner
            app_name = "aura_search"
            
        session = await session_service.get_session(app_name=app_name, user_id=req.user_id, session_id=req.session_id)
        if not session:
            session = await session_service.create_session(app_name=app_name, user_id=req.user_id, session_id=req.session_id)
        
        user_content = types.Content(role="user", parts=[types.Part(text=req.message)])
        
        response_text = ""
        events = []
        async for event in use_runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=user_content):
            events.append(event)
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        
        if not response_text and events:
            for ev in reversed(events):
                if getattr(ev, 'role', '') == 'model' or getattr(getattr(ev, 'content', None), 'role', '') == 'model':
                    content = getattr(ev, 'content', ev)
                    if hasattr(content, 'parts'):
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text = part.text
                                break
                if response_text:
                    break

        if not response_text:
            response_text = "I processed your request, but received no text output."

        return {"response": response_text, "session_id": req.session_id}
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            return {"response": "⚠️ Web Search quota limit reached! Turn OFF Web Search toggle to continue chatting for free, or wait 1 minute!"}
        return JSONResponse(status_code=500, content={"error": err_msg})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
