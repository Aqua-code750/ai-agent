import os
import asyncio
from fastapi import FastAPI, Request
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

from my_agent.agent import root_agent

app = FastAPI(title="Aura Agent")

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

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = "default_session"

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        session = await session_service.get_session(app_name="aura", user_id=req.user_id, session_id=req.session_id)
        if not session:
            session = await session_service.create_session(app_name="aura", user_id=req.user_id, session_id=req.session_id)
        
        user_content = types.Content(role="user", parts=[types.Part(text=req.message)])
        
        response_text = ""
        events = []
        async for event in runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=user_content):
            events.append(event)
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        
        if not response_text and events:
            # Extract last model output if present
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
        return JSONResponse(status_code=500, content={"error": str(e)})

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
            --border-color: #e0e0e0;
            --text-primary: #1f1f1f;
            --text-secondary: #444746;
            --accent-color: #1a73e8;
            --accent-hover: #1557b0;
            --user-bubble: #e8f0fe;
            --ai-bubble: #f1f3f4;
            --card-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #131314;
                --surface-color: #1e1f20;
                --border-color: #333537;
                --text-primary: #e3e3e3;
                --text-secondary: #c4c7c5;
                --accent-color: #a8c7fa;
                --accent-hover: #7cacf8;
                --user-bubble: #282a2c;
                --ai-bubble: #1e1f20;
                --card-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Google Sans', 'Roboto', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); height: 100vh; display: flex; flex-direction: column; }
        header { padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-color); background: var(--surface-color); }
        .logo { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 500; color: var(--text-primary); }
        .logo-icon { width: 24px; height: 24px; background: linear-gradient(135deg, #4285f4, #9b51e0); border-radius: 50%; }
        .status-badge { font-size: 12px; padding: 4px 10px; background: rgba(26, 115, 232, 0.1); color: var(--accent-color); border-radius: 12px; font-weight: 500; }
        #chat-container { flex: 1; overflow-y: auto; padding: 24px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        .message { display: flex; gap: 12px; max-width: 85%; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; flex-shrink: 0; }
        .user .avatar { background: var(--accent-color); color: #fff; }
        .ai .avatar { background: linear-gradient(135deg, #4285f4, #34a853); color: #fff; }
        .bubble { padding: 14px 18px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; white-space: pre-wrap; }
        .user .bubble { background: var(--user-bubble); color: var(--text-primary); border-top-right-radius: 4px; }
        .ai .bubble { background: var(--ai-bubble); color: var(--text-primary); border: 1px solid var(--border-color); border-top-left-radius: 4px; }
        footer { padding: 16px 24px; max-width: 800px; width: 100%; margin: 0 auto; }
        .input-box { display: flex; align-items: center; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 28px; padding: 8px 16px; transition: border-color 0.2s; }
        .input-box:focus-within { border-color: var(--accent-color); box-shadow: 0 0 0 2px rgba(26,115,232,0.2); }
        input { flex: 1; border: none; background: transparent; color: var(--text-primary); font-size: 16px; padding: 8px; outline: none; }
        button { background: var(--accent-color); color: white; border: none; width: 38px; height: 38px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s; flex-shrink: 0; }
        button:hover { background: var(--accent-hover); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .spinner { width: 18px; height: 18px; border: 2px solid #ffffff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <div class="logo-icon"></div>
            <span>Aura</span>
        </div>
        <div class="status-badge">Powered by Gemini 3.6 Flash</div>
    </header>

    <div id="chat-container">
        <div class="message ai">
            <div class="avatar">A</div>
            <div class="bubble">Hello! I am Aura, your AI assistant. How can I help you today?</div>
        </div>
    </div>

    <footer>
        <div class="input-box">
            <input type="text" id="userInput" placeholder="Ask Aura anything..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button id="sendBtn" onclick="sendMessage()">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
        </div>
    </footer>

    <script>
        const sessionId = "session_" + Math.random().toString(36).substring(2, 9);
        
        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const message = input.value.trim();
            if (!message) return;

            input.value = "";
            appendMessage("user", "U", message);

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div>';

            try {
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: message, session_id: sessionId })
                });
                const data = await res.json();
                appendMessage("ai", "A", data.response || data.error || "No response received.");
            } catch (err) {
                appendMessage("ai", "A", "Error connecting to server: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
            }
        }

        function appendMessage(sender, initial, text) {
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
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async function get_root():
    return HTML_CONTENT

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("my_agent.main:app", host="0.0.0.0", port=port, reload=False)
