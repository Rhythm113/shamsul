# shamsul-server

shamsul-server is a local, offline-first middleware between the Claude Code CLI and your local Ollama instance. It implements a server-side context-engineering workflow that can route agentic coding requests dynamically using the Multi-Model Bridge orchestration system.

It features a web-based Control Center to manage configurations, download models, and edit system planning prompts in real-time.

---

## Key Features
* **Multi-Model Bridge**: Route requests dynamically:
  1. **Bridge Head Model** (Reasoning Lead, e.g. qwen3.5:latest) parses instructions and generates a plan.
  2. **Coding Executor** (e.g. qwen3.5:latest) or **Tooling Executor** consumes the plan and runs tasks.
* **Optimized Prompt Compression**: Automatically extracts actionable parameters (OS, shell, working directory, git state) and discards bloated system prompts to fit local model context.
* **Extended OS Detection**: Automatically detects win32, darwin (macOS), and linux platform environments.
* **Strict Execution Constraints**: Injects 10 strict guardrails to prevent local models from simulating tool execution, inventing outputs, or writing mock logs.
* **Shorter Tool Representations**: Compacts parameter typings and descriptions to fit local model context windows.
* **Reasoning Fallback**: Automatically injects fallback execution plans if the reasoning model query fails.
* **100% Local & Private**: No external API keys or cloud dependencies.
* **Web Control Center**: Interactive dashboard served at http://localhost:8082/web (and /admin) to select models and modify prompts.
* **Integrated Model Downloader**: Download any model directly from the Ollama registry via the web dashboard.
* **FastAPI Backend & SSE Streams**: Real-time plan streaming inside native Claude Code thinking blocks.

---

## 1. System Requirements
1. **Ollama**: Download and install it from ollama.com. Ensure the Ollama service is running locally (http://localhost:11434).
2. **Node.js & npm** (required to run the Claude Code client CLI).

---

## 2. Manual Setup Instructions

Clone the repository, navigate to the folder, and follow the setup instructions for your operating system:

### Windows (PowerShell)
1. Install uv if not already present:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. Run the automated workspace setup script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
   *This installs Python 3.14.0, synchronizes local dependencies, and scaffolds the configuration files.*

### Linux & macOS (Bash)
1. Install uv if not already present:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Make the setup script executable and run it:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

---

## 3. Starting the Server

Launch the FastAPI middleware server using the script corresponding to your platform:
* **Windows Command Prompt**: Double-click or run run.bat
* **Windows PowerShell**: Run .\run.ps1
* **Linux & macOS**: Run ./run.sh

The server will start listening at http://localhost:8082.

---

## 4. Web Control Center & Model Pulling

Once the server is running, open your browser and navigate to:
* **http://localhost:8082/web** (or /admin which automatically redirects)

### Model Installation Guide (Ollama)
To run the server-side pipeline, you need reasoning and coding models pulled in Ollama.

You can download them in two ways:
1. **Via the Web Control Center**:
    * Go to the **Ollama Model Downloader** card at the bottom of the dashboard.
    * Input the model name (e.g. qwen3.5:latest or llama3.2:latest) and click **Pull Model**.
 2. **Via Command Line**:
    * Open your terminal and run:
      ```bash
      ollama pull qwen3.5:latest
      ollama pull llama3.2:latest
      ```

### Recommended Model Pairs
* **Standard (8GB - 12GB VRAM)**:
  * Reasoning Lead: qwen3.5:latest (9b)
  * Coding/Tooling Executor: qwen3.5:latest (9b)
* **Resource Constrained (<8GB VRAM)**:
  * Reasoning Lead: llama3.2:latest (3b)
  * Coding Executor: qwen2.5-coder:3b

---

## 5. Global CLI Setup (Running Claude Code)

To execute Claude Code through shamsul-server from anywhere on your system:

1. **Install tools globally**:
   Run the following command in the project directory to register the launcher executables:
   ```bash
   uv tool install --force .
   ```
   This registers 4 global commands:
   * `shamsul-server` (launches the FastAPI gateway)
   * `shamsul-init` (scaffolds the default environment config)
   * `shamsul-claude` (launches the Claude Code CLI routing to shamsul-server)
   * `shamsul-codex` (launches the Codex CLI routing to shamsul-server)

2. **Verify PATH Environment Variable**:
   Ensure that the global uv tool bin directory is in your user PATH variable:
   * **Windows**: %USERPROFILE%\.local\bin
   * **Linux/macOS**: ~/.local/bin

3. **Launch**:
   * Start the server in one window with `shamsul-server`.
   * Open a new window and type `shamsul-claude` to start pair programming with Claude Code fully offline!
