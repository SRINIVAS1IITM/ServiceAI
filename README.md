# 🤖 AI Customer Support Agent

An intelligent, agentic system that automates the initial stages of customer inquiry processing — classifying intent, retrieving knowledge, drafting responses, and escalating to human agents when needed. Powered by a local LLM via [Ollama](https://ollama.com/) and semantic search via FAISS.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Knowledge Base](#knowledge-base)
- [Customization](#customization)

---

## Overview

This system receives customer queries and autonomously:

1. **Classifies** the intent of the inquiry
2. **Acts** on the classification (knowledge search, logging, lead capture)
3. **Drafts** a tailored preliminary response
4. **Escalates** to a human agent when necessary

All NLP tasks run locally — no external API keys required.

---

## Features

### Intent Classification
Incoming inquiries are categorized into one of four types:

| Intent | Description |
|---|---|
| 🔧 Technical Support | Issues, bugs, and how-to questions |
| 💡 Product Feature Request | Suggestions for new features |
| 💼 Sales Lead | Purchase interest and pricing inquiries |
| 💬 General Inquiry | Everything else |

### Intelligent Actions per Intent
- **Technical Support** — Searches the knowledge base using RAG (Retrieval-Augmented Generation) and returns relevant results.
- **Feature Requests** — Logs the request to `backend/data/feature_requests.log`.
- **Sales Leads** — Gathers missing information (e.g., company name) before proceeding.

### Escalation Logic
Human escalation is triggered when:
- The knowledge base search returns no useful results
- Customer sentiment is highly negative
- The sales inquiry is too complex to automate

### Other Highlights
- 🧠 **Local LLM** via Ollama (`phi3:mini`) — runs fully offline
- 🔍 **Semantic Search** with FAISS + `sentence-transformers`
- 🖥️ **React Frontend** with Chakra UI for a clean chat interface
- 🪲 **Debug Panel** in the UI showing intent, KB search details, and escalation reasoning

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Ollama (`phi3:mini`) | Local LLM inference |
| `sentence-transformers` | Text embeddings |
| `faiss-cpu` | Vector similarity search |
| Pydantic | Data validation |
| Requests | HTTP client for Ollama |

### Frontend
| Technology | Purpose |
|---|---|
| React.js | UI framework |
| Chakra UI | Component library & styling |
| Axios | API communication |

---

## Project Structure

```
customer-support-agent/
├── backend/
│   ├── app/                    # FastAPI application (routes, agent logic, KB manager)
│   ├── data/
│   │   ├── knowledge_base.json # Mock knowledge base entries
│   │   └── feature_requests.log# Logged feature requests
│   ├── venv/                   # Python virtual environment (git-ignored)
│   └── requirements.txt
├── frontend/
│   ├── public/                 # Static assets & index.html
│   ├── src/                    # React source code
│   ├── node_modules/           # Node dependencies (git-ignored)
│   └── package.json
├── .gitignore
└── README.md
```

---

## Prerequisites

Make sure the following are installed before proceeding:

- **Python** 3.8 or higher
- **Node.js** LTS (includes npm)
- **Ollama** — download from [https://ollama.com/](https://ollama.com/)
- **Git** *(optional, for cloning)*

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd customer-support-agent
```

### 2. Set Up Ollama

Ensure Ollama is installed and running, then pull the required model:

```bash
ollama pull phi3:mini
```

> **Note:** The Ollama server usually runs in the background after installation. If not, start it manually with `ollama serve` in a separate terminal.

### 3. Set Up the Backend

Navigate to the backend directory:

```bash
cd backend
```

Create and activate a Python virtual environment:

```bash
# Create the environment
python3 -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set Up the Frontend

Navigate to the frontend directory:

```bash
cd ../frontend
```

Install Node.js dependencies:

```bash
npm install
```

---

## Running the Application

You need **two terminals** — one for the backend and one for the frontend.

### Terminal 1 — Start the Backend

From `customer-support-agent/backend/` with your virtual environment active:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

The API will be available at: **http://localhost:8000**

### Terminal 2 — Start the Frontend

From `customer-support-agent/frontend/`:

```bash
npm start
```

The app will open automatically at: **http://localhost:3000**

---

## Usage

1. Open **http://localhost:3000** in your browser.
2. Type a customer inquiry into the input box and press **Enter** or click **Send**.
3. The agent will process your query and display:
   - The **drafted response**
   - Detected **intent**
   - **Escalation status** (human needed or not)
   - **Debug information** (KB search results, reasoning)

---

## Knowledge Base

The mock knowledge base lives at `backend/data/knowledge_base.json`.

You can add or modify entries freely. The backend re-indexes the KB on startup. If you're running with `--reload`, a manual server restart may be needed for structural changes to take effect.

---

## Customization

| What | Where |
|---|---|
| Change the LLM model | `backend/app/agent_logic.py` → `OLLAMA_MODEL_NAME` |
| Edit knowledge base entries | `backend/data/knowledge_base.json` |
| Adjust escalation rules | `backend/app/agent_logic.py` |
| Modify UI components | `frontend/src/` |

> When switching models, make sure to tune the prompts in `agent_logic.py` accordingly, as different models respond to different prompt styles.
