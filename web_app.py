import os
import json
import asyncio
import subprocess
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="nanochat Web Frontend")

# HTML + CSS + JS Single Page Application
INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>nanochat - Neural Chat Interface</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f0fdf4',
              500: '#22c55e',
              600: '#16a34a',
              700: '#15803d',
            },
            dark: {
              bg: '#0f172a',
              card: '#1e293b',
              border: '#334155',
              hover: '#475569'
            }
          }
        }
      }
    }
  </script>
  <style>
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    .chat-bubble { animation: fadeIn 0.2s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body class="bg-dark-bg text-slate-100 flex flex-col h-screen overflow-hidden font-sans">

  <!-- Header -->
  <header class="bg-dark-card/80 backdrop-blur border-b border-dark-border px-6 py-3.5 flex items-center justify-between shrink-0 shadow-md">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
        <i class="fa-solid fa-brain text-slate-950 font-bold text-lg"></i>
      </div>
      <div>
        <h1 class="font-bold text-lg leading-tight flex items-center gap-2">
          <span>nanochat</span>
          <span class="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full font-medium">GPT-2 Class</span>
        </h1>
        <p class="text-xs text-slate-400">Karpathy's LLM Harness • Local & Cloud GPU Interface</p>
      </div>
    </div>
    
    <div class="flex items-center gap-3">
      <!-- Kaggle Status Badge -->
      <div id="kaggle-badge" class="flex items-center gap-2 text-xs bg-dark-bg border border-dark-border px-3 py-1.5 rounded-lg text-slate-300">
        <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" id="status-indicator"></span>
        <span id="kaggle-status-text">Checking GPU Kernel...</span>
      </div>
      
      <a href="https://github.com/NB7551498/nanoGPT-2.0" target="_blank" class="text-slate-400 hover:text-white px-2.5 py-1.5 rounded-lg border border-dark-border hover:bg-dark-card transition flex items-center gap-2 text-xs">
        <i class="fa-brands fa-github text-sm"></i>
        <span>Repo</span>
      </a>
    </div>
  </header>

  <!-- Main Container -->
  <div class="flex-1 flex overflow-hidden">

    <!-- Sidebar Controls -->
    <aside class="w-80 bg-dark-card/50 border-r border-dark-border flex flex-col p-4 shrink-0 hidden md:flex overflow-y-auto">
      <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
        <i class="fa-solid fa-sliders"></i> Generation Settings
      </h2>

      <!-- Temperature -->
      <div class="mb-4">
        <div class="flex justify-between text-xs text-slate-300 mb-1.5">
          <span>Temperature</span>
          <span id="temp-val" class="font-mono text-emerald-400">0.7</span>
        </div>
        <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value="0.7" class="w-full accent-emerald-500 cursor-pointer" oninput="document.getElementById('temp-val').innerText = this.value" />
      </div>

      <!-- Top-K -->
      <div class="mb-4">
        <div class="flex justify-between text-xs text-slate-300 mb-1.5">
          <span>Top-K</span>
          <span id="topk-val" class="font-mono text-emerald-400">50</span>
        </div>
        <input id="topk" type="range" min="1" max="100" step="1" value="50" class="w-full accent-emerald-500 cursor-pointer" oninput="document.getElementById('topk-val').innerText = this.value" />
      </div>

      <!-- Max Tokens -->
      <div class="mb-5">
        <div class="flex justify-between text-xs text-slate-300 mb-1.5">
          <span>Max Tokens</span>
          <span id="tokens-val" class="font-mono text-emerald-400">256</span>
        </div>
        <input id="tokens" type="range" min="32" max="1024" step="32" value="256" class="w-full accent-emerald-500 cursor-pointer" oninput="document.getElementById('tokens-val').innerText = this.value" />
      </div>

      <!-- Model Source Selection -->
      <div class="mb-5">
        <label class="block text-xs font-medium text-slate-300 mb-1.5">Model Source / Checkpoint</label>
        <select id="model-source" class="w-full bg-dark-bg border border-dark-border text-xs rounded-lg p-2 text-slate-200 focus:outline-none focus:border-emerald-500">
          <option value="sft">SFT (Chat Finetuned)</option>
          <option value="rl">RL (Reinforcement Learning)</option>
          <option value="base">Base Model (Pretrained)</option>
        </select>
      </div>

      <!-- System Prompt -->
      <div class="mb-5">
        <label class="block text-xs font-medium text-slate-300 mb-1.5">System Prompt</label>
        <textarea id="system-prompt" rows="3" class="w-full bg-dark-bg border border-dark-border text-xs rounded-lg p-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500" placeholder="Optional instruction for the model..."></textarea>
      </div>

      <!-- Actions -->
      <div class="mt-auto space-y-2 pt-4 border-t border-dark-border">
        <button onclick="clearChat()" class="w-full bg-dark-bg hover:bg-dark-hover border border-dark-border text-slate-300 text-xs py-2 rounded-lg transition flex items-center justify-center gap-2">
          <i class="fa-solid fa-trash-can text-slate-400"></i> Clear Conversation
        </button>
        <a href="https://www.kaggle.com/code/nikhilbhardwaj2003/nanochat-gpu-training" target="_blank" class="w-full bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/30 text-emerald-400 text-xs py-2 rounded-lg transition flex items-center justify-center gap-2 font-medium">
          <i class="fa-solid fa-bolt text-emerald-400"></i> Open Kaggle GPU Runner
        </a>
      </div>
    </aside>

    <!-- Chat Area -->
    <main class="flex-1 flex flex-col bg-dark-bg overflow-hidden relative">

      <!-- Messages Stream -->
      <div id="messages-container" class="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
        
        <!-- Welcome Greeting -->
        <div class="max-w-2xl mx-auto text-center py-10">
          <div class="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-4 text-emerald-400">
            <i class="fa-solid fa-comments text-xl"></i>
          </div>
          <h2 class="text-xl font-bold mb-1.5">Welcome to nanochat UI</h2>
          <p class="text-sm text-slate-400 max-w-md mx-auto">
            Interact with your custom-trained GPT model, test prompts, and monitor cloud GPU checkpoints directly.
          </p>
          
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-6 max-w-lg mx-auto text-left">
            <button onclick="usePrompt('Why is the sky blue?')" class="p-3 bg-dark-card border border-dark-border hover:border-emerald-500/50 rounded-xl text-xs transition">
              <span class="font-medium text-slate-200">🌌 "Why is the sky blue?"</span>
              <p class="text-slate-400 text-[11px] mt-0.5">Classic Karpathy demo prompt</p>
            </button>
            <button onclick="usePrompt('Write a short poem about an AI learning to speak.')" class="p-3 bg-dark-card border border-dark-border hover:border-emerald-500/50 rounded-xl text-xs transition">
              <span class="font-medium text-slate-200">📜 "Write a poem about AI"</span>
              <p class="text-slate-400 text-[11px] mt-0.5">Creative text generation</p>
            </button>
          </div>
        </div>

      </div>

      <!-- Input Bar -->
      <div class="p-4 bg-dark-card/60 backdrop-blur border-t border-dark-border">
        <form id="chat-form" onsubmit="sendMessage(event)" class="max-w-3xl mx-auto flex items-end gap-2">
          <div class="flex-1 bg-dark-bg border border-dark-border rounded-xl focus-within:border-emerald-500 focus-within:ring-1 focus-within:ring-emerald-500 transition px-3.5 py-2.5 flex items-center">
            <textarea
              id="user-input"
              rows="1"
              placeholder="Ask anything or test a prompt..."
              class="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-36"
              onkeydown="handleKey(event)"
            ></textarea>
          </div>
          <button
            id="send-btn"
            type="submit"
            class="h-10 px-4 bg-emerald-500 hover:bg-emerald-600 active:scale-95 text-slate-950 font-semibold rounded-xl flex items-center justify-center transition shadow-lg shadow-emerald-500/20 shrink-0"
          >
            <i class="fa-solid fa-arrow-up"></i>
          </button>
        </form>
        <p class="text-center text-[10px] text-slate-500 mt-2">
          nanochat can produce inaccurate information. Designed for research and experimentation.
        </p>
      </div>

    </main>
  </div>

  <script>
    let isGenerating = false;

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chat-form').requestSubmit();
      }
    }

    function usePrompt(text) {
      const input = document.getElementById('user-input');
      input.value = text;
      input.focus();
    }

    function clearChat() {
      const container = document.getElementById('messages-container');
      container.innerHTML = `
        <div class="text-center py-6 text-xs text-slate-500">
          Conversation cleared.
        </div>
      `;
    }

    function appendMessage(role, text) {
      const container = document.getElementById('messages-container');
      const msgDiv = document.createElement('div');
      msgDiv.className = `flex gap-3 max-w-2xl chat-bubble ${role === 'user' ? 'ml-auto flex-row-reverse' : ''}`;
      
      const avatar = role === 'user' 
        ? '<div class="w-7 h-7 rounded-lg bg-slate-700 flex items-center justify-center shrink-0 text-xs font-semibold">U</div>'
        : '<div class="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center justify-center shrink-0 text-xs"><i class="fa-solid fa-brain"></i></div>';

      const bubbleClass = role === 'user'
        ? 'bg-emerald-600 text-slate-950 rounded-2xl rounded-tr-none px-4 py-2.5 text-sm font-medium'
        : 'bg-dark-card border border-dark-border text-slate-200 rounded-2xl rounded-tl-none px-4 py-2.5 text-sm shadow-sm';

      msgDiv.innerHTML = `
        ${avatar}
        <div class="${bubbleClass} whitespace-pre-wrap leading-relaxed max-w-[85%]" id="${role === 'assistant' ? 'current-response' : ''}">${text}</div>
      `;

      container.appendChild(msgDiv);
      container.scrollTop = container.scrollHeight;
      return msgDiv.querySelector('#current-response');
    }

    async function sendMessage(e) {
      e.preventDefault();
      if (isGenerating) return;

      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      appendMessage('user', text);

      isGenerating = true;
      document.getElementById('send-btn').disabled = true;
      const responseElement = appendMessage('assistant', '<span class="inline-block animate-pulse text-emerald-400">Thinking...</span>');

      const payload = {
        prompt: text,
        temperature: parseFloat(document.getElementById('temp').value),
        top_k: parseInt(document.getElementById('topk').value),
        max_tokens: parseInt(document.getElementById('tokens').value),
        source: document.getElementById('model-source').value,
        system_prompt: document.getElementById('system-prompt').value
      };

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Generation failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        responseElement.innerText = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          fullText += chunk;
          responseElement.innerText = fullText;
          document.getElementById('messages-container').scrollTop = document.getElementById('messages-container').scrollHeight;
        }
      } catch (err) {
        responseElement.innerText = "Error: Unable to generate response. Please check server logs.";
        console.error(err);
      } finally {
        isGenerating = false;
        document.getElementById('send-btn').disabled = false;
        responseElement.removeAttribute('id');
      }
    }

    // Polling Kaggle status
    async function checkKaggleStatus() {
      try {
        const res = await fetch('/api/kaggle_status');
        const data = await res.json();
        const badge = document.getElementById('kaggle-status-text');
        const dot = document.getElementById('status-indicator');
        
        if (data.status.includes('RUNNING')) {
          badge.innerText = 'GPU Kernel: Running';
          dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse';
        } else if (data.status.includes('COMPLETE')) {
          badge.innerText = 'GPU Kernel: Finished';
          dot.className = 'w-2.5 h-2.5 rounded-full bg-blue-400';
        } else {
          badge.innerText = 'GPU: ' + data.status.replace('KernelWorkerStatus.', '');
          dot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400';
        }
      } catch (e) {
        console.log("Kaggle check error:", e);
      }
    }

    checkKaggleStatus();
    setInterval(checkKaggleStatus, 15000);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=INDEX_HTML)

@app.get("/api/kaggle_status")
async def get_kaggle_status():
    try:
        proc = subprocess.run(
            ["python", "-m", "kaggle", "kernels", "status", "nikhilbhardwaj2003/nanochat-gpu-training"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = proc.stdout.strip()
        status = output.split("status")[-1].replace('"', '').strip() if "status" in output else "Active"
        return {"status": status, "raw": output}
    except Exception as e:
        return {"status": "Available", "error": str(e)}

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temp = data.get("temperature", 0.7)
    top_k = data.get("top_k", 50)
    max_tokens = data.get("max_tokens", 256)
    source = data.get("source", "sft")

    async def generate_response() -> AsyncGenerator[str, None]:
        # Try to execute through python chat CLI or fallback to stream
        cmd = [
            "python", "-m", "scripts.chat_cli",
            "-i", source,
            "-p", prompt,
            "-t", str(temp),
            "-k", str(top_k),
            "--device-type", "cpu"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read stdout line by line or character by character
            if process.stdout:
                while True:
                    line = await process.stdout.read(64)
                    if not line:
                        break
                    yield line.decode("utf-8", errors="replace")
            await process.wait()
            
            # If no model checkpoint was found locally, provide helpful guidance
            if process.returncode != 0:
                yield f"\\n\\n[Note: Local checkpoint for '{source}' is currently training on Kaggle GPU. You can download the completed checkpoint or inspect live logs at Kaggle!]"
        except Exception as e:
            yield f"Simulation error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
