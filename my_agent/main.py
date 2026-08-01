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

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from my_agent.agent import root_agent, search_agent, image_agent, code_agent
except ModuleNotFoundError:
    from agent import root_agent, search_agent, image_agent, code_agent

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
runner = Runner(app=adk_app, session_service=session_service, artifact_service=artifact_service, credential_service=credential_service)

adk_search_app = App(name="aura_search", root_agent=search_agent)
search_runner = Runner(app=adk_search_app, session_service=session_service, artifact_service=artifact_service, credential_service=credential_service)

adk_image_app = App(name="aura_image", root_agent=image_agent)
image_runner = Runner(app=adk_image_app, session_service=session_service, artifact_service=artifact_service, credential_service=credential_service)

adk_code_app = App(name="aura_code", root_agent=code_agent)
code_runner = Runner(app=adk_code_app, session_service=session_service, artifact_service=artifact_service, credential_service=credential_service)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = "default_session"
    enable_search: bool = False
    mode: str = "chat"  # "chat", "image", "code", "search"
    image_base64: str | None = None
    image_mime_type: str | None = None

def perform_live_search(query: str) -> str:
    """Zero-quota real-time search engine using Google News RSS."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')[:5]
        results = []
        for item in items:
            t = item.find('title')
            l = item.find('link')
            title = t.text if t is not None else ''
            link = l.text if l is not None else ''
            results.append(f"- {title} ({link})")
        return "\n".join(results) if results else "No specific news headlines found."
    except Exception as e:
        return f"Real-time news search unavailable ({e})"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Aura - Multimodal AI Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
            --code-bg: #1e1e1e;
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
                --code-bg: #0d0d0d;
            }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Google Sans', 'Roboto', sans-serif; -webkit-tap-highlight-color: transparent; }
        body { background-color: var(--bg-color); color: var(--text-primary); height: 100vh; height: 100dvh; display: flex; overflow: hidden; position: fixed; width: 100%; }
        
        #sidebarBackdrop { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.4); z-index: 99; display: none; opacity: 0; transition: opacity 0.3s ease; }
        #sidebarBackdrop.open { display: block; opacity: 1; }

        #sidebar { width: 280px; background: var(--sidebar-color); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; transition: transform 0.3s ease, margin-right 0.3s ease; flex-shrink: 0; z-index: 100; }
        .sidebar-header { padding: 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-color); }
        .new-chat-btn { display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; padding: 12px 16px; background: var(--accent-color); color: white; border: none; border-radius: 24px; font-weight: 500; font-size: 14px; cursor: pointer; transition: background 0.2s; min-height: 44px; }
        .new-chat-btn:hover { background: var(--accent-hover); }
        .chat-list { flex: 1; overflow-y: auto; padding: 12px 8px; display: flex; flex-direction: column; gap: 4px; }
        .chat-item { padding: 12px 14px; border-radius: 12px; font-size: 14px; color: var(--text-primary); cursor: pointer; display: flex; align-items: center; justify-content: space-between; transition: background 0.2s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-height: 44px; }
        .chat-item:hover { background: var(--hover-color); }
        .chat-item.active { background: var(--user-bubble); font-weight: 500; color: var(--accent-color); }
        .delete-chat { color: var(--text-secondary); padding: 6px; border-radius: 50%; border: none; background: transparent; cursor: pointer; min-width: 32px; min-height: 32px; display: flex; align-items: center; justify-content: center; }
        .delete-chat:hover { color: #ea4335; background: var(--hover-color); }

        @media (min-width: 769px) {
            #sidebar.collapsed { transform: translateX(-280px); margin-right: -280px; }
        }
        @media (max-width: 768px) {
            #sidebar { position: fixed; top: 0; left: 0; bottom: 0; height: 100dvh; transform: translateX(-100%); }
            #sidebar.open { transform: translateX(0); box-shadow: 4px 0 24px rgba(0,0,0,0.25); }
        }

        #main-wrapper { flex: 1; display: flex; flex-direction: column; height: 100vh; height: 100dvh; overflow: hidden; width: 100%; }
        header { padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-color); background: var(--surface-color); min-height: 56px; }
        .left-head { display: flex; align-items: center; gap: 10px; }
        .icon-btn { background: transparent; border: none; color: var(--text-primary); cursor: pointer; padding: 8px; border-radius: 50%; display: flex; align-items: center; justify-content: center; min-width: 40px; min-height: 40px; }
        .icon-btn:hover { background: var(--hover-color); }
        .logo { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 500; }
        .logo-icon { width: 24px; height: 24px; background: var(--accent-color); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 13px; }
        
        .right-head { display: flex; align-items: center; gap: 8px; }
        .profile-chip { display: flex; align-items: center; gap: 6px; padding: 6px 12px; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 18px; font-size: 13px; font-weight: 500; cursor: pointer; color: var(--text-primary); max-width: 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-height: 36px; }
        .profile-chip:hover { background: var(--hover-color); }
        .logout-btn { padding: 6px 12px; background: transparent; border: 1px solid var(--border-color); border-radius: 18px; font-size: 13px; font-weight: 500; cursor: pointer; color: #ea4335; display: none; min-height: 36px; }

        #chat-container { flex: 1; overflow-y: auto; padding: 16px 12px; max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; -webkit-overflow-scrolling: touch; }
        .message { display: flex; gap: 10px; max-width: 90%; animation: fadeIn 0.2s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: white; flex-shrink: 0; }
        .user .avatar { background: var(--user-avatar-bg); }
        .ai .avatar { background: var(--ai-avatar-bg); }
        .bubble { padding: 14px 18px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; border: 1px solid var(--border-color); overflow: hidden; }
        .user .bubble { background: var(--user-bubble); border: none; border-top-right-radius: 4px; }
        .ai .bubble { background: var(--ai-bubble); border-top-left-radius: 4px; }
        
        .bubble img { max-width: 100%; border-radius: 12px; margin-top: 8px; border: 1px solid var(--border-color); display: block; }
        .bubble pre { background: var(--code-bg); color: #00ff66; padding: 12px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 13px; overflow-x: auto; margin: 8px 0; }
        .bubble code { font-family: 'Fira Code', monospace; font-size: 13px; background: rgba(0,0,0,0.06); padding: 2px 5px; border-radius: 4px; }

        footer { padding: 10px 14px calc(10px + env(safe-area-inset-bottom, 0px)); max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; background: var(--bg-color); }
        
        .mode-bar { display: flex; align-items: center; gap: 6px; overflow-x: auto; padding-bottom: 2px; -webkit-overflow-scrolling: touch; }
        .mode-pill { font-size: 12px; font-weight: 500; border: 1px solid var(--border-color); padding: 6px 14px; border-radius: 16px; cursor: pointer; background: var(--surface-color); color: var(--text-secondary); transition: all 0.2s; display: flex; align-items: center; gap: 5px; user-select: none; flex-shrink: 0; min-height: 32px; }
        .mode-pill.active { background: rgba(26, 115, 232, 0.15); color: var(--accent-color); border-color: var(--accent-color); font-weight: 700; }

        #imagePreviewContainer { display: none; align-items: center; gap: 10px; padding: 6px 12px; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 12px; width: max-content; max-width: 100%; }
        #imagePreviewContainer img { width: 44px; height: 44px; border-radius: 8px; object-fit: cover; }
        #imagePreviewContainer button { background: transparent; border: none; color: #ea4335; font-size: 16px; cursor: pointer; padding: 4px; }

        .input-box { display: flex; align-items: center; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 24px; padding: 4px 8px 4px 8px; transition: border-color 0.2s; min-height: 48px; gap: 4px; }
        .input-box:focus-within { border-color: var(--accent-color); box-shadow: 0 0 0 2px rgba(26,115,232,0.15); }
        input[type="text"] { flex: 1; border: none; background: transparent; color: var(--text-primary); font-size: 16px; padding: 8px 4px; outline: none; width: 100%; }
        .attach-btn { background: transparent; border: none; color: var(--text-secondary); padding: 8px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; }
        .attach-btn:hover { background: var(--hover-color); color: var(--accent-color); }

        .send-btn { background: var(--accent-color); color: white; border: none; width: 38px; height: 38px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s; flex-shrink: 0; }
        .send-btn:hover { background: var(--accent-hover); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .spinner { width: 18px; height: 18px; border: 2px solid #ffffff; border-top: 2px solid transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: none; align-items: center; justify-content: center; z-index: 2000; padding: 16px; }
        .modal-overlay.open { display: flex; }
        .modal { background: var(--bg-color); border: 1px solid var(--border-color); border-radius: 20px; padding: 24px 20px; width: 100%; max-width: 380px; display: flex; flex-direction: column; gap: 14px; }
        .modal h3 { font-size: 18px; font-weight: 500; }
        .tab-bar { display: flex; border-bottom: 1px solid var(--border-color); margin-bottom: 4px; }
        .tab { flex: 1; text-align: center; padding: 10px; font-size: 14px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; color: var(--text-secondary); }
        .tab.active { border-color: var(--accent-color); color: var(--accent-color); }
        .modal input { border: 1px solid var(--border-color); border-radius: 12px; padding: 12px 14px; font-size: 16px; width: 100%; color: var(--text-primary); background: var(--surface-color); outline: none; }
        .modal-btns { display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px; }
        .btn-flat { padding: 12px 20px; border-radius: 12px; font-size: 14px; font-weight: 500; cursor: pointer; border: none; min-height: 44px; }
        .btn-secondary { background: var(--surface-color); color: var(--text-primary); border: 1px solid var(--border-color); }
        .btn-primary { background: var(--accent-color); color: white; }
        .auth-error { color: #ea4335; font-size: 13px; display: none; }
    </style>
</head>
<body>
    <div id="sidebarBackdrop" onclick="closeSidebar()"></div>

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
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                </button>
                <div class="logo">
                    <div class="logo-icon">A</div>
                    <span>Aura</span>
                </div>
            </div>
            <div class="right-head">
                <div class="profile-chip" id="profileChip" onclick="openAuthModal()">
                    <span>👤</span>
                    <span id="profileName">Sign In</span>
                </div>
                <button class="logout-btn" id="logoutBtn" onclick="handleLogout()">Log Out</button>
            </div>
        </header>

        <div id="chat-container"></div>

        <footer>
            <div class="mode-bar">
                <div class="mode-pill active" id="modeChat" onclick="setMode('chat')">💬 Chat</div>
                <div class="mode-pill" id="modeImage" onclick="setMode('image')">🎨 Image Gen</div>
                <div class="mode-pill" id="modeCode" onclick="setMode('code')">💻 Coding</div>
                <div class="mode-pill" id="modeSearch" onclick="setMode('search')">🔍 Search</div>
            </div>

            <div id="imagePreviewContainer">
                <img id="imagePreview" src="" alt="Attachment">
                <span id="imageName" style="font-size: 13px;"></span>
                <button onclick="removeAttachedImage()">✕</button>
            </div>

            <div class="input-box">
                <input type="file" id="filePicker" accept="image/*" style="display: none;" onchange="handleFileSelected(event)">
                <button class="attach-btn" onclick="document.getElementById('filePicker').click()" title="Attach image">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
                </button>
                <input type="text" id="userInput" placeholder="Ask Aura anything or attach an image..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </footer>
    </div>

    <!-- Auth Modal -->
    <div class="modal-overlay" id="authModal">
        <div class="modal">
            <div class="tab-bar">
                <div class="tab active" id="tabLogin" onclick="switchAuthTab('login')">Log In</div>
                <div class="tab" id="tabSignup" onclick="switchAuthTab('signup')">Sign Up</div>
            </div>
            <div id="signupFields" style="display: none;">
                <input type="text" id="authName" placeholder="Full Name or Username" style="margin-bottom: 10px;">
            </div>
            <input type="email" id="authEmail" placeholder="Email address" style="margin-bottom: 10px;">
            <input type="password" id="authPassword" placeholder="Password (6+ characters)">
            <div class="auth-error" id="authError"></div>
            <div class="modal-btns">
                <button class="btn-flat btn-secondary" onclick="closeAuthModal()">Cancel</button>
                <button class="btn-flat btn-primary" id="authSubmitBtn" onclick="handleAuthSubmit()">Log In</button>
            </div>
        </div>
    </div>

    <script type="module">
        import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
        import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged, updateProfile } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

        const firebaseConfig = {
            apiKey: "AIzaSyDepR3o-75rgL4Ah5MXGZtFoDJFCk2BEB4",
            authDomain: "fir-fb9da.firebaseapp.com",
            projectId: "fir-fb9da",
            storageBucket: "fir-fb9da.firebasestorage.app",
            messagingSenderId: "606836149753",
            appId: "1:606836149753:web:bb8b8ea9ed705680528128",
            measurementId: "G-PW22WVC6L1"
        };

        const fbApp = initializeApp(firebaseConfig);
        const auth = getAuth(fbApp);

        window.fbAuth = auth;
        window.fbCreateUser = createUserWithEmailAndPassword;
        window.fbSignIn = signInWithEmailAndPassword;
        window.fbSignOut = signOut;
        window.fbUpdateProfile = updateProfile;

        onAuthStateChanged(auth, (user) => {
            if (user) {
                window.currentUser = {
                    uid: user.uid,
                    displayName: user.displayName || user.email.split('@')[0],
                    email: user.email
                };
                document.getElementById("profileName").innerText = window.currentUser.displayName;
                document.getElementById("logoutBtn").style.display = "block";
            } else {
                window.currentUser = null;
                document.getElementById("profileName").innerText = "Sign In";
                document.getElementById("logoutBtn").style.display = "none";
            }
            window.loadUserSessions();
        });
    </script>

    <script>
        let currentAuthMode = "login";
        let sessions = [];
        let activeSessionId = null;
        let currentMode = "chat";
        let attachedImageBase64 = null;
        let attachedImageMime = null;

        function isMobile() { return window.innerWidth <= 768; }

        function setMode(mode) {
            currentMode = mode;
            ["chat", "image", "code", "search"].forEach(m => {
                const el = document.getElementById("mode" + m.charAt(0).toUpperCase() + m.slice(1));
                if (m === mode) el.classList.add("active");
                else el.classList.remove("active");
            });

            const input = document.getElementById("userInput");
            if (mode === "image") input.placeholder = "Describe an image to generate...";
            else if (mode === "code") input.placeholder = "Ask for code or debugging...";
            else if (mode === "search") input.placeholder = "Search live Google facts and news...";
            else input.placeholder = "Ask Aura anything or attach an image...";
        }

        function handleFileSelected(event) {
            const file = event.target.files[0];
            if (!file) return;

            attachedImageMime = file.type || "image/jpeg";
            const reader = new FileReader();
            reader.onload = function(e) {
                attachedImageBase64 = e.target.result;
                document.getElementById("imagePreview").src = attachedImageBase64;
                document.getElementById("imageName").innerText = file.name;
                document.getElementById("imagePreviewContainer").style.display = "flex";
            };
            reader.readAsDataURL(file);
        }

        function removeAttachedImage() {
            attachedImageBase64 = null;
            attachedImageMime = null;
            document.getElementById("filePicker").value = "";
            document.getElementById("imagePreviewContainer").style.display = "none";
        }

        function toggleSidebar() {
            const sidebar = document.getElementById("sidebar");
            const backdrop = document.getElementById("sidebarBackdrop");
            if (isMobile()) {
                sidebar.classList.toggle("open");
                if (sidebar.classList.contains("open")) backdrop.classList.add("open");
                else backdrop.classList.remove("open");
            } else {
                sidebar.classList.toggle("collapsed");
            }
        }

        function closeSidebar() {
            document.getElementById("sidebar").classList.remove("open");
            document.getElementById("sidebarBackdrop").classList.remove("open");
        }

        function openAuthModal() {
            document.getElementById("authError").style.display = "none";
            document.getElementById("authModal").classList.add("open");
        }
        function closeAuthModal() { document.getElementById("authModal").classList.remove("open"); }

        function switchAuthTab(mode) {
            currentAuthMode = mode;
            document.getElementById("authError").style.display = "none";
            if (mode === "signup") {
                document.getElementById("tabSignup").classList.add("active");
                document.getElementById("tabLogin").classList.remove("active");
                document.getElementById("signupFields").style.display = "block";
                document.getElementById("authSubmitBtn").innerText = "Sign Up";
            } else {
                document.getElementById("tabLogin").classList.add("active");
                document.getElementById("tabSignup").classList.remove("active");
                document.getElementById("signupFields").style.display = "none";
                document.getElementById("authSubmitBtn").innerText = "Log In";
            }
        }

        async function handleAuthSubmit() {
            const email = document.getElementById("authEmail").value.trim();
            const password = document.getElementById("authPassword").value;
            const name = document.getElementById("authName").value.trim();
            const errDiv = document.getElementById("authError");
            errDiv.style.display = "none";

            if (!email || !password) {
                errDiv.innerText = "Please enter email and password.";
                errDiv.style.display = "block";
                return;
            }

            try {
                if (currentAuthMode === "signup") {
                    const res = await window.fbCreateUser(window.fbAuth, email, password);
                    if (name && res.user) await window.fbUpdateProfile(res.user, { displayName: name });
                } else {
                    await window.fbSignIn(window.fbAuth, email, password);
                }
                closeAuthModal();
            } catch (err) {
                errDiv.innerText = err.message.replace("Firebase: ", "");
                errDiv.style.display = "block";
            }
        }

        async function handleLogout() {
            if (window.fbSignOut && window.fbAuth) await window.fbSignOut(window.fbAuth);
        }

        window.loadUserSessions = function() {
            const userId = window.currentUser ? window.currentUser.uid : "guest";
            sessions = JSON.parse(localStorage.getItem(`aura_fb_sessions_${userId}`)) || [];
            activeSessionId = localStorage.getItem(`aura_fb_active_${userId}`) || null;

            if (!activeSessionId || !sessions.find(s => s.id === activeSessionId)) startNewChat();
            else { renderSidebar(); loadActiveSession(); }
        };

        function startNewChat() {
            activeSessionId = "sess_" + Math.random().toString(36).substring(2, 9);
            const newSess = { id: activeSessionId, title: "New Conversation", messages: [] };
            sessions.unshift(newSess);
            saveState();
            renderSidebar();
            loadActiveSession();
            if (isMobile()) closeSidebar();
        }

        function saveState() {
            const userId = window.currentUser ? window.currentUser.uid : "guest";
            localStorage.setItem(`aura_fb_sessions_${userId}`, JSON.stringify(sessions));
            localStorage.setItem(`aura_fb_active_${userId}`, activeSessionId);
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
            if (isMobile()) closeSidebar();
        }

        function deleteSession(id) {
            sessions = sessions.filter(s => s.id !== id);
            if (activeSessionId === id) activeSessionId = sessions.length ? sessions[0].id : null;
            saveState();
            if (!activeSessionId) startNewChat();
            else { renderSidebar(); loadActiveSession(); }
        }

        function loadActiveSession() {
            const container = document.getElementById("chat-container");
            container.innerHTML = "";
            const currentSess = sessions.find(s => s.id === activeSessionId);
            if (!currentSess || currentSess.messages.length === 0) {
                appendMessageToDOM("ai", "A", "Hello! I am Aura, your Multimodal AI Assistant. You can send text, attach images 📎, or select a mode below!");
                return;
            }
            const userInitial = window.currentUser ? window.currentUser.displayName[0].toUpperCase() : "U";
            currentSess.messages.forEach(msg => {
                appendMessageToDOM(msg.sender, msg.sender === 'user' ? userInitial : 'A', msg.text, msg.image);
            });
        }

        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const message = input.value.trim();
            const sendImageBase64 = attachedImageBase64;
            const sendImageMime = attachedImageMime;

            if (!message && !sendImageBase64) return;

            input.value = "";
            removeAttachedImage();

            let currentSess = sessions.find(s => s.id === activeSessionId);
            if (!currentSess) {
                startNewChat();
                currentSess = sessions.find(s => s.id === activeSessionId);
            }

            if (currentSess.messages.length === 0) {
                currentSess.title = (message || "Image Attached").substring(0, 24) + ((message || "").length > 24 ? "..." : "");
            }

            const userInitial = window.currentUser ? window.currentUser.displayName[0].toUpperCase() : "U";
            currentSess.messages.push({ sender: "user", text: message, image: sendImageBase64 });
            appendMessageToDOM("user", userInitial, message, sendImageBase64);
            saveState();
            renderSidebar();

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div>';

            try {
                const userId = window.currentUser ? window.currentUser.uid : "guest";
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        message: message || "Please analyze this image.", 
                        user_id: userId,
                        session_id: activeSessionId,
                        enable_search: currentMode === "search",
                        mode: currentMode,
                        image_base64: sendImageBase64,
                        image_mime_type: sendImageMime
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

        function appendMessageToDOM(sender, initial, text, imageSrc = null) {
            const container = document.getElementById("chat-container");
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${sender}`;

            const avatar = document.createElement("div");
            avatar.className = "avatar";
            avatar.innerText = initial;

            const bubble = document.createElement("div");
            bubble.className = "bubble";

            if (imageSrc) {
                const imgEl = document.createElement("img");
                imgEl.src = imageSrc;
                bubble.appendChild(imgEl);
            }

            if (text) {
                const textDiv = document.createElement("div");
                if (window.marked) textDiv.innerHTML = marked.parse(text);
                else textDiv.innerText = text;
                bubble.appendChild(textDiv);
            }

            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
            container.appendChild(msgDiv);
            container.scrollTop = container.scrollHeight;
        }

        window.addEventListener("resize", () => { if (!isMobile()) closeSidebar(); });
        window.loadUserSessions();
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
        
        # Route to appropriate runner mode
        if req.mode == "image" or req.message.startswith("/image"):
            use_runner = image_runner
            app_name = "aura_image"
        elif req.mode == "code" or req.message.startswith("/code"):
            use_runner = code_runner
            app_name = "aura_code"
        elif req.mode == "search" or req.enable_search or req.message.startswith("/search"):
            use_runner = search_runner
            app_name = "aura_search"
            
        session = await session_service.get_session(app_name=app_name, user_id=req.user_id, session_id=req.session_id)
        if not session:
            session = await session_service.create_session(app_name=app_name, user_id=req.user_id, session_id=req.session_id)
        
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        live_date_context = now.strftime("%A, %B %d, %Y")
        
        # Quota-free real-time search context injection
        search_ctx = ""
        if req.mode == "search" or req.enable_search or req.message.startswith("/search"):
            search_ctx = f"\n\nReal-Time Web Search Headlines for '{req.message}':\n" + perform_live_search(req.message)

        prompt_text = f"[System Note: Live Date Context - Today is {live_date_context}. Current Year: {now.year}. Mode: {req.mode}. Fully trust and accept all 2026 news, tech developments, and AI model releases.{search_ctx}]\n\n{req.message}"
        
        parts = [types.Part(text=prompt_text)]
        if req.image_base64:
            import base64
            clean_b64 = req.image_base64.split(",")[-1]
            img_bytes = base64.b64decode(clean_b64)
            mime = req.image_mime_type or "image/jpeg"
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        user_content = types.Content(role="user", parts=parts)
        
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
            return {"response": "⚠️ Rate limit reached! Please wait 30 seconds and try again!"}
        return JSONResponse(status_code=500, content={"error": err_msg})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
