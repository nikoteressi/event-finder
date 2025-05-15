Okay, this is an ambitious and exciting project! Generating a plan detailed enough for an AI like Gemini 2.5 Pro to implement step-by-step, incorporating best practices and simulating documentation knowledge from May 2025, requires a very granular approach.

Here's the comprehensive, raw Markdown plan. I've made assumptions about the evolution of these libraries towards greater stability and developer-friendliness, especially in areas like LangChain's tool use with Ollama and LangServe's capabilities.

```markdown
# AI Event Agent Implementation Plan (May 2025) - Detailed for AI Implementation

**Project Overview:**
AI Event Agent is a proactive event discovery system. The backend, built with Python, LangChain, and FastAPI (using LangServe), will manage event scraping tasks and interact with a remote Ollama instance. The frontend will be a SvelteKit application. Web scraping will initially use LangChain's Playwright Browser Toolkit. The high-risk phone interaction module (`scrcpy`-based) will be developed in a later phase. The Ollama instance is pre-configured and operational.

**Guiding Principles:**
- SOLID, DRY, KISS.
- Clear separation of concerns (Frontend, Backend API, AI Core).
- Asynchronous programming (`async/await`) where beneficial (FastAPI, LangChain async methods).
- Comprehensive error handling and logging.
- Test-driven development prácticas (unit, integration, E2E tests).
- Environment variables for configuration.
- Modularity for LangChain components (Tools, Chains, Agents).

**Assumed Project Structure (Monorepo):**
```
<PROJECT_ROOT>/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app & LangServe setup
│   │   ├── api/                # FastAPI routers (for tasks, etc.)
│   │   │   ├── __init__.py
│   │   │   └── task_router.py
│   │   ├── agents/             # LangChain agent definitions
│   │   │   └── event_agent.py
│   │   ├── chains/             # LangChain chains (parsing, RAG)
│   │   │   └── event_parsing_chain.py
│   │   ├── core/               # Config, DB (if any beyond Chroma)
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── models/             # Pydantic models for API and data
│   │   │   ├── __init__.py
│   │   │   ├── common_models.py
│   │   │   ├── task_models.py
│   │   │   └── event_models.py
│   │   ├── services/           # Business logic not in agents/chains
│   │   │   └── task_scheduler_service.py
│   │   ├── tools/              # LangChain custom tools
│   │   │   ├── __init__.py
│   │   │   ├── web_scraper_tool.py
│   │   │   └── datetime_parser_tool.py
│   │   └── memory/             # Memory setup (ChromaDB client)
│   │       └── vector_store.py
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── Dockerfile              # For backend deployment
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app.html
│   │   ├── hooks.server.js     # Optional: server-side hooks
│   │   ├── lib/
│   │   │   ├── components/     # Svelte components
│   │   │   │   ├── TaskForm.svelte
│   │   │   │   ├── TaskList.svelte
│   │   │   │   └── EventCalendar.svelte
│   │   │   ├── services/       # API client services
│   │   │   │   └── apiService.js
│   │   │   ├── stores/         # Svelte stores
│   │   │   │   └── taskStore.js
│   │   │   └── utils/
│   │   │       └── dateTimeUtils.js
│   │   ├── routes/             # SvelteKit routes
│   │   │   ├── +layout.svelte
│   │   │   ├── +page.svelte
│   │   │   └── tasks/
│   │   │       ├── +page.svelte
│   │   │       └── [id]/
│   │   │           └── +page.svelte
│   ├── static/
│   ├── tests/
│   │   ├── unit/
│   │   └── e2e/
│   ├── svelte.config.js
│   ├── vite.config.js
│   ├── package.json
│   ├── postcss.config.cjs
│   ├── tailwind.config.cjs  # If using Tailwind CSS
│   └── .env.example
├── .gitignore
└── README.md
```

---

## Phase 1: Environment Setup and Core Backend Infrastructure

### Step 1.1: Backend - Python Environment & Project Structure
1.  **Action:** Create directory `<PROJECT_ROOT>/backend`.
2.  **Action:** Initialize Python virtual environment (e.g., `python -m venv .venv`). Activate it.
3.  **Action:** Create `backend/requirements.txt` with initial dependencies:
    ```
    fastapi
    uvicorn[standard]
    python-dotenv
    pydantic
    # LangChain Core & Community (assuming modular packages by May 2025)
    langchain-core
    langchain-community
    langchain-ollama
    langchain-experimental # For Playwright toolkit if still experimental
    langchain-text-splitters
    langchain-chroma # Or chromadb directly if preferred for client setup
    langserve

    # Ollama Python client (might be used by langchain-ollama or directly)
    ollama

    # Web Scraping (Playwright)
    playwright
    beautifulsoup4

    # Task Scheduling
    apscheduler

    # Vector DB Client
    chromadb-client

    # Testing
    pytest
    pytest-asyncio
    httpx # For testing FastAPI endpoints

    # ADB utilities (for later phone module)
    # adbutils
    ```
4.  **Action:** Install dependencies: `pip install -r backend/requirements.txt`.
5.  **Action:** Run `playwright install` to install browser binaries.
6.  **Action:** Create basic directory structure under `backend/app/` as outlined in "Assumed Project Structure". Create `__init__.py` files in `app`, `api`, `core`, `models`, `services`, `tools`, `agents`, `chains`, `memory`.
7.  **Action:** Create `backend/.env.example`:
    ```env
    OLLAMA_BASE_URL="http://localhost:11434" # Default Ollama URL
    TEXT_LLM_MODEL_NAME="llama3"
    VISION_LLM_MODEL_NAME="llava" # For future use
    CHROMA_DB_PATH="./chromadb_data"
    # Add other configurations as needed
    ```
8.  **Action:** Create `backend/app/core/config.py` to load environment variables:
    ```python
    # backend/app/core/config.py
    from pydantic_settings import BaseSettings
    import os
    from dotenv import load_dotenv

    load_dotenv() # Load .env file

    class Settings(BaseSettings):
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        TEXT_LLM_MODEL_NAME: str = os.getenv("TEXT_LLM_MODEL_NAME", "llama3")
        VISION_LLM_MODEL_NAME: str = os.getenv("VISION_LLM_MODEL_NAME", "llava")
        CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chromadb_data")
        # Add other configurations

        class Config:
            env_file = ".env" # Redundant if load_dotenv() is used, but good practice
            extra = "ignore"

    settings = Settings()
    ```

### Step 1.2: Frontend - SvelteKit Environment & Project Structure
1.  **Action:** Create directory `<PROJECT_ROOT>/frontend`.
2.  **Action:** Initialize SvelteKit project in `frontend/`:
    ```bash
    cd <PROJECT_ROOT>/frontend
    npm create svelte@latest . # Or yarn create svelte .
    # Choose: Skeleton project, TypeScript, ESLint, Prettier, Playwright for browser testing, Vitest for unit testing.
    ```
3.  **Action:** Install additional frontend dependencies (if not included or for specific needs, e.g., Tailwind, date formatting):
    ```bash
    npm install -D tailwindcss postcss autoprefixer # If choosing Tailwind
    npm install svelte-headless-table # Example for tables
    npm install date-fns # For date formatting
    # Add other UI libraries or utilities as needed
    ```
4.  **Action:** Initialize Tailwind CSS if chosen: `npx tailwindcss init -p`. Configure `tailwind.config.cjs` and `app.html`.
5.  **Action:** Create basic directory structure under `frontend/src/lib/` and `frontend/src/routes/` as outlined.
6.  **Action:** Create `frontend/.env.example`:
    ```env
    VITE_API_BASE_URL="http://localhost:8000/api" # Backend API URL
    ```
    Ensure variables are prefixed with `VITE_` for SvelteKit to expose them to client-side code.

### Step 1.3: Backend - Basic FastAPI & LangServe Setup
1.  **Action:** Create `backend/app/main.py`:
    ```python
    # backend/app/main.py
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    # from langserve import add_routes # This will be used later
    # from app.api import task_router # This will be used later
    from app.core.config import settings
    import uvicorn

    app = FastAPI(
        title="AI Event Agent API",
        version="0.1.0",
        description="API for managing AI event agent tasks and results."
    )

    # CORS Middleware
    # Adjust origins as necessary for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"], # SvelteKit default dev port
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "ollama_url": settings.OLLAMA_BASE_URL}

    # Placeholder for LangServe routes - will be populated later
    # Example:
    # from app.agents.event_agent import get_event_discovery_agent_executor # Assuming agent is defined
    # add_routes(
    #     app,
    #     get_event_discovery_agent_executor(), # This needs to be a Runnable
    #     path="/event-agent"
    # )

    # Placeholder for FastAPI routers - will be populated later
    # app.include_router(task_router.router, prefix="/api/v1", tags=["Tasks"])


    if __name__ == "__main__":
        # This is for development only. For production, use a process manager like Gunicorn.
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```
2.  **Action:** Test basic FastAPI setup:
    - Ensure `.env` file exists in `backend/` with `OLLAMA_BASE_URL`.
    - Run `python backend/app/main.py`.
    - Open browser to `http://localhost:8000/health` and `http://localhost:8000/docs`.

---

## Phase 2: Agent Core Development (LangChain Focused)

### Step 2.1: Backend - LLM Interface (LangChain Wrappers)
1.  **Action:** Create `backend/app/core/llm_services.py` (or integrate into `config.py` or a dedicated LLM setup file):
    ```python
    # backend/app/core/llm_services.py
    from langchain_ollama.chat_models import ChatOllama
    # from langchain_ollama.llms import OllamaLLM # For non-chat models if needed
    # from langchain_community.llms.ollama import Ollama # Alternative import for vision
    from app.core.config import settings

    # It's anticipated that by May 2025, ChatOllama might handle vision directly,
    # or a specific OllamaVision wrapper will be stable.
    # For now, let's define them separately and assume ChatOllama handles text.
    # The vision model wrapper might be different.

    def get_text_llm() -> ChatOllama:
        """Initializes and returns the primary text-based LLM."""
        return ChatOllama(
            model=settings.TEXT_LLM_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1, # Adjust as needed for predictability
            # Add other parameters like top_p, mirostat, etc.
        )

    def get_vision_llm() -> ChatOllama: # Or a specific VisionLLM wrapper
        """
        Initializes and returns the vision-capable LLM.
        Note: As of early 2024, LLaVA interaction with ChatOllama for images
        might require specific formatting of image inputs (e.g., base64).
        This is expected to be streamlined by May 2025.
        """
        # This might require a different class or specific parameters
        # if ChatOllama doesn't transparently handle multimodal inputs by then.
        # For LLaVA, it's often used as a multi-modal model within ChatOllama.
        return ChatOllama(
            model=settings.VISION_LLM_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2, # May need different settings for vision tasks
        )

    # Optional: Centralized LLM instances if you prefer not to recreate them on each call
    # text_llm = get_text_llm()
    # vision_llm = get_vision_llm()
    ```
2.  **Action:** Test LLM connection (can be a simple script or a pytest unit test):
    ```python
    # Example test snippet (adapt for pytest)
    # from app.core.llm_services import get_text_llm
    # llm = get_text_llm()
    # try:
    #     response = llm.invoke("Hello, how are you?")
    #     print(f"LLM Response: {response.content}")
    # except Exception as e:
    #     print(f"Error connecting to LLM: {e}")
    ```
3.  **Action:** Setup LangChain Caching (Optional but Recommended).
    Example with in-memory cache (can be extended to SQLite or Redis via LangChain):
    ```python
    # Potentially in backend/app/core/llm_services.py or a dedicated cache_setup.py
    import langchain
    from langchain.cache import InMemoryCache # or SQLiteCache, RedisCache

    def setup_langchain_cache():
        # For simplicity, using InMemoryCache. For persistence, use SQLiteCache or other.
        langchain.llm_cache = InMemoryCache()
        # To use SQLite:
        # langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
        print("LangChain LLM Caching enabled.")

    # Call this once at application startup, e.g., in main.py or an init hook
    # setup_langchain_cache() # Add to backend/app/main.py before route definitions
    ```
    Update `backend/app/main.py` to call `setup_langchain_cache()`.

### Step 2.2: Backend - Pydantic Models for Core Entities
1.  **Action:** Create `backend/app/models/common_models.py`:
    ```python
    # backend/app/models/common_models.py
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Any
    from datetime import datetime

    class APIResponse(BaseModel):
        success: bool = True
        message: Optional[str] = None
        data: Optional[Any] = None # Can be a specific type or List of types
    ```
2.  **Action:** Create `backend/app/models/task_models.py`:
    ```python
    # backend/app/models/task_models.py
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict
    from datetime import datetime
    import uuid

    class TaskBase(BaseModel):
        natural_language_query: str = Field(..., description="User's query for event search")
        # Fields to be extracted by LLM or set by user
        keywords: Optional[List[str]] = None
        location: Optional[str] = None
        date_range_start: Optional[datetime] = None
        date_range_end: Optional[datetime] = None
        # For scheduled tasks
        cron_schedule: Optional[str] = None # e.g., "0 0 * * 0" for weekly Sunday at midnight
        is_recurring: bool = False

    class TaskCreateRequest(TaskBase):
        pass

    class TaskUpdateRequest(TaskBase):
        natural_language_query: Optional[str] = None # All fields optional for update

    class TaskResponse(TaskBase):
        id: uuid.UUID = Field(default_factory=uuid.uuid4)
        status: str = Field(default="PENDING", description="e.g., PENDING, RUNNING, COMPLETED, FAILED")
        created_at: datetime = Field(default_factory=datetime.utcnow)
        updated_at: datetime = Field(default_factory=datetime.utcnow)
        last_run_at: Optional[datetime] = None
        results_summary: Optional[str] = None # Or a link/ID to detailed results
        # execution_logs: Optional[List[str]] = None # Potentially too verbose for direct response

        class Config:
            orm_mode = True # or from_attributes = True for Pydantic v2
    ```
3.  **Action:** Create `backend/app/models/event_models.py`:
    ```python
    # backend/app/models/event_models.py
    from pydantic import BaseModel, Field, HttpUrl
    from typing import Optional, List
    from datetime import datetime

    class Event(BaseModel):
        title: str
        description: Optional[str] = None
        start_datetime: Optional[datetime] = None
        end_datetime: Optional[datetime] = None
        location_name: Optional[str] = None
        address: Optional[str] = None
        url: Optional[HttpUrl] = None
        source_url: HttpUrl # Where the event was found
        organizer: Optional[str] = None
        categories: Optional[List[str]] = None
        raw_extracted_data: Optional[dict] = None # For debugging or further processing

    class EventScrapingResult(BaseModel):
        task_id: str # Corresponds to TaskResponse.id (as string for broader compatibility if needed)
        events_found: List[Event] = []
        errors: List[str] = []
        processed_at: datetime = Field(default_factory=datetime.utcnow)
    ```

### Step 2.3: Backend - Agent Memory (ChromaDB Setup)
1.  **Action:** Create `backend/app/memory/vector_store.py`:
    ```python
    # backend/app/memory/vector_store.py
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings
    from langchain_ollama.embeddings import OllamaEmbeddings # Assuming this exists and is stable
    # Alternatively, use a sentence-transformer model locally if Ollama embeddings are slow/problematic
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    from app.core.config import settings as app_settings

    # Global Chroma client (can be managed better with FastAPI lifespan events)
    # This setup assumes ChromaDB is running as a persistent server or using a local persistent path.
    # For local persistent path:
    chroma_client = chromadb.PersistentClient(
        path=app_settings.CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False) # Disable telemetry if preferred
    )

    def get_ollama_embeddings() -> Embeddings:
        # Ensure the embedding model used by Ollama is suitable for retrieval tasks.
        # Models like 'nomic-embed-text' or 'mxbai-embed-large' are good.
        # The specific model name might need to be configured in .env if different from TEXT_LLM_MODEL_NAME.
        return OllamaEmbeddings(
            model=app_settings.TEXT_LLM_MODEL_NAME, # Or a dedicated embedding model name
            base_url=app_settings.OLLAMA_BASE_URL
        )
        # Example with HuggingFaceEmbeddings (if OllamaEmbeddings are problematic)
        # return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    def get_vector_store(collection_name: str = "event_agent_memory") -> Chroma:
        """Initializes and returns a Chroma vector store for a given collection."""
        embeddings = get_ollama_embeddings()
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        return vector_store

    # Example of adding documents (will be used by memory modules or RAG chains)
    # def add_texts_to_collection(texts: List[str], metadatas: List[dict], collection_name: str):
    #     vector_store = get_vector_store(collection_name)
    #     vector_store.add_texts(texts=texts, metadatas=metadatas)
    ```
2.  **Action:** Test ChromaDB connection and embedding generation (can be a script or pytest).
    - Ensure `CHROMA_DB_PATH` in `.env` points to a writable directory.

### Step 2.4: Backend - Core Agent Definition (Initial - No Tools Yet)
1.  **Action:** Create `backend/app/agents/event_agent.py`:
    ```python
    # backend/app/agents/event_agent.py
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import Runnable, RunnablePassthrough
    from langchain.agents import AgentExecutor, create_tool_calling_agent # Or create_react_agent, etc.
    # from langchain.memory import ConversationBufferMemory # More advanced memory will be added later
    # from langchain_community.chat_message_histories import ChatMessageHistory # For ConversationBufferMemory
    # from langchain_core.chat_history import BaseChatMessageHistory # For ConversationBufferMemory

    from app.core.llm_services import get_text_llm
    from app.models.task_models import TaskCreateRequest # For type hinting or direct use

    # Tools will be added later. For now, the agent will be a simple conversational one.
    TOOLS = [] # Placeholder for LangChain tools

    # Define the prompt for the agent
    # This prompt will evolve as tools are added.
    # By May 2025, Ollama models (like Llama 3 variants) should be good at tool calling.
    # The prompt structure for tool calling agents might differ slightly based on LangChain's agent constructors.
    # This is a generic example for a tool-calling agent.
    SYSTEM_PROMPT_TEMPLATE = """
    You are an AI Event Agent. Your goal is to help users find events based on their queries.
    You have access to a set of tools to find information.
    Given a user's query, understand their needs for:
    - Event type or keywords
    - Location (city, state, region in the USA)
    - Timeframe (specific dates, relative dates like "this weekend", "next month")

    If the query is ambiguous, ask clarifying questions.
    Once you have enough information, use your tools to find relevant events.
    Present the found events clearly.

    If you need to ask for clarification, do so before attempting to use tools if critical information is missing.
    """

    def get_event_discovery_agent_executor() -> AgentExecutor:
        """
        Creates and returns the LangChain AgentExecutor for event discovery.
        """
        llm = get_text_llm()

        # This prompt is simplified. For tool calling, specific placeholders for tools and agent_scratchpad are needed.
        # LangChain's `create_tool_calling_agent` typically handles this internally if you pass it a messages list.
        # For more control, you might build the prompt manually.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For tool execution steps
            ]
        )

        # Agent creation (assuming a tool-calling agent type)
        # The exact method might vary slightly (e.g. create_ollama_tools_agent)
        # but create_tool_calling_agent is a common modern pattern.
        agent = create_tool_calling_agent(llm, TOOLS, prompt)

        # AgentExecutor runs the agent
        # verbose=True is good for development
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,
            handle_parsing_errors=True, # Gracefully handle LLM output parsing errors
            # memory=... # Memory will be added later
        )
        return agent_executor

    # Example of how to run (will be exposed via LangServe)
    # async def run_agent_query(query: str, chat_history: Optional[List[tuple]] = None):
    #     agent_executor = get_event_discovery_agent_executor()
    #     # LangChain's memory components handle chat_history formatting for prompts.
    #     # For direct invocation or LangServe, input format needs to match agent's expectation.
    #     # Typically: {"input": "user query", "chat_history": [HumanMessage(...), AIMessage(...)]}
    #     input_data = {"input": query}
    #     if chat_history:
    #         # Convert chat_history to BaseMessage instances if necessary
    #         # input_data["chat_history"] = ...
    #         pass # Placeholder for chat history management
    #
    #     response = await agent_executor.ainvoke(input_data)
    #     return response
    ```
2.  **Action:** Integrate this basic agent with LangServe in `backend/app/main.py`:
    ```python
    # backend/app/main.py
    # ... (other imports)
    from langserve import add_routes # Ensure this is imported
    from app.agents.event_agent import get_event_discovery_agent_executor
    from app.core.llm_services import setup_langchain_cache # If defined separately

    # ... (app = FastAPI())

    # Call cache setup once
    setup_langchain_cache() # Assuming this function exists

    # ... (CORS middleware)

    # Add LangServe routes for the agent
    # The runnable from get_event_discovery_agent_executor() is the AgentExecutor itself.
    event_agent_executor = get_event_discovery_agent_executor()
    add_routes(
        app,
        event_agent_executor,
        path="/event-agent",
        # You can configure input/output types for better OpenAPI docs if LangServe supports it well
        # input_type=... # Pydantic model for input
        # output_type=... # Pydantic model for output
        enabled_endpoints=['invoke', 'stream', 'batch', 'playground', 'stream_log'] # Customize as needed
    )

    # ... (health check and other routers)
    ```
3.  **Action:** Test the LangServe endpoint:
    - Run `python backend/app/main.py`.
    - Open browser to `http://localhost:8000/event-agent/playground`.
    - Try a simple query like `{"input": "find music events"}`. Expect a conversational response as no tools are active.

---
## Phase 3: Web Scraping Implementation as LangChain Tools

### Step 3.1: Backend - DateTime Parser Tool
1.  **Action:** Create `backend/app/tools/datetime_parser_tool.py`:
    ```python
    # backend/app/tools/datetime_parser_tool.py
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
    from datetime import datetime, timedelta
    from dateutil import parser as dateutil_parser
    from typing import Optional, Tuple

    # It's often useful to have the LLM call this tool with the current date for context.
    # Alternatively, the tool can fetch the current date itself.

    class DateTimeParserInput(BaseModel):
        natural_language_date_query: str = Field(description="The natural language date query, e.g., 'this weekend', 'next Friday', 'in 2 weeks'")
        current_datetime_utc_iso: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Current UTC datetime in ISO format. If not provided, tool will use its current time.")

    @tool("datetime_parser", args_schema=DateTimeParserInput, return_direct=False)
    def datetime_parser(natural_language_date_query: str, current_datetime_utc_iso: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses a natural language date query (e.g., "this weekend", "next month", "July 15th to July 20th")
        and returns a tuple of (start_datetime_iso, end_datetime_iso).
        Uses the provided current_datetime_utc_iso as a reference for relative queries.
        Returns None for start or end if they cannot be determined.
        """
        try:
            if current_datetime_utc_iso:
                now = dateutil_parser.isoparse(current_datetime_utc_iso)
            else:
                now = datetime.utcnow()

            query = natural_language_date_query.lower()
            start_date: Optional[datetime] = None
            end_date: Optional[datetime] = None

            # Simple relative date logic (can be significantly expanded)
            if "today" in query:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif "tomorrow" in query:
                tomorrow = now + timedelta(days=1)
                start_date = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif "this weekend" in query:
                # Friday to Sunday (adjust logic based on desired definition of weekend)
                days_until_friday = (4 - now.weekday() + 7) % 7
                start_date = (now + timedelta(days=days_until_friday)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (start_date + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=999999)
            elif "next week" in query:
                days_until_monday = (0 - now.weekday() + 7) % 7
                if days_until_monday == 0: # if today is Monday, next week's Monday
                    days_until_monday = 7
                start_date = (now + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (start_date + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=999999)
            # Add more sophisticated parsing using dateutil.parser for specific dates or ranges
            # For example, using LLM to first structure "from X to Y" then parse X and Y.
            # Or, a more complex rule-based system or a specialized date parsing library if dateutil is insufficient.
            else:
                # Fallback to dateutil.parser for single dates or simple phrases it understands
                # This is very basic, LLM might need to be prompted to provide clearer date strings.
                try:
                    parsed_dt = dateutil_parser.parse(query, default=now)
                    start_date = parsed_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    # Assume a single day event if only one date is parsed clearly
                    end_date = parsed_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                except dateutil_parser.ParserError:
                    # Could attempt to split ranges "July 10 to July 15" here
                    return None, None # Cannot parse

            start_iso = start_date.isoformat() if start_date else None
            end_iso = end_date.isoformat() if end_date else None
            return start_iso, end_iso

        except Exception as e:
            # Log the error
            print(f"Error in datetime_parser: {e}")
            return None, None # Indicate failure
    ```

### Step 3.2: Backend - Web Scraper Tool (LangChain Playwright)
1.  **Action:** Create `backend/app/tools/web_scraper_tool.py`:
    ```python
    # backend/app/tools/web_scraper_tool.py
    from langchain_core.tools import tool, ToolException
    from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit # Or specific tools
    from langchain_community.tools.playwright.utils import (
        create_async_playwright_browser, # If not using the toolkit's browser directly
        # get_current_page,
    )
    # from playwright.async_api import Page, Browser # For type hinting if needed
    from pydantic import BaseModel, Field, HttpUrl
    from typing import List, Optional, Dict
    import asyncio

    # It's good practice to initialize the browser once if the tool is long-lived,
    # or manage its lifecycle carefully. LangChain toolkits often handle this.
    # For a standalone tool, you might want to manage the browser instance.
    # Let's assume we'll create/close browser per high-level call for simplicity in this tool,
    # or rely on a toolkit that manages it.
    # By May 2025, Playwright toolkit might offer more granular control or better context management.

    # Global browser instance (consider lifespan management for FastAPI app)
    # This approach is simplified. For production, manage the browser lifecycle with FastAPI lifespan events.
    _browser_instance = None

    async def get_playwright_browser_instance(headless=True):
        global _browser_instance
        if _browser_instance is None:
            _browser_instance = await create_async_playwright_browser(headless=headless)
        return _browser_instance

    # Tool for navigating and extracting structured data from a single event page.
    # The agent might first use a search engine tool (not defined here) or a site-specific search tool
    # to find a list of event URLs, then pass each URL to this tool.
    class ScrapeEventPageInput(BaseModel):
        event_url: HttpUrl = Field(description="The URL of the specific event page to scrape.")
        extraction_schema: Dict = Field(description="A dictionary defining what to extract, e.g. {'title': 'css_selector_for_title', 'date': 'css_selector_for_date'}. Use 'textContent' for text, 'href' for links, etc.")

    @tool("scrape_specific_event_page", args_schema=ScrapeEventPageInput, return_direct=False)
    async def scrape_specific_event_page(event_url: str, extraction_schema: Dict) -> Dict:
        """
        Navigates to a specific event page URL and extracts information based on the provided CSS selectors
        defined in the extraction_schema.
        Returns a dictionary of extracted data.
        Keys in extraction_schema can be like 'title', 'description', 'start_datetime_str', 'location_str'.
        The value for each key in extraction_schema should be its CSS selector.
        """
        browser = await get_playwright_browser_instance()
        page = None
        scraped_data = {"source_url": event_url}
        errors = []

        try:
            page = await browser.new_page()
            await page.goto(event_url, timeout=20000) # Increased timeout
            await page.wait_for_load_state("domcontentloaded", timeout=15000)

            for key, selector in extraction_schema.items():
                try:
                    element = page.locator(selector).first # Take the first match
                    # A more robust implementation might check if element exists
                    # await element.wait_for(timeout=5000) # Wait for element to be visible/attached
                    if await element.count() > 0:
                        # Simplistic extraction, could be more nuanced (e.g., get attribute 'href')
                        # For now, assumes text content is desired.
                        # The LLM should be prompted to provide selectors that point to text.
                        # Or the schema could specify 'attribute: href' etc.
                        content = await element.text_content()
                        scraped_data[key] = content.strip() if content else None
                    else:
                        scraped_data[key] = None
                        errors.append(f"Element for '{key}' with selector '{selector}' not found.")
                except Exception as e_extract:
                    errors.append(f"Error extracting '{key}' with selector '{selector}': {str(e_extract)}")
                    scraped_data[key] = None

            if errors:
                scraped_data["_scraping_errors"] = errors

        except Exception as e:
            # Log the error
            error_message = f"Error scraping event page {event_url}: {e}"
            print(error_message)
            # This tool itself should not raise ToolException if it can return partial data or an error message.
            # The agent can then decide how to handle it.
            # However, if it's a total failure, ToolException might be appropriate.
            # For now, return errors within the data.
            scraped_data["_critical_scraping_failure"] = error_message
        finally:
            if page:
                await page.close()
            # Consider when to close the browser. If managed globally, don't close here.
            # If created per call (not recommended for performance):
            # if browser: await browser.close()

        return scraped_data

    # A higher-level tool that the agent might use first.
    # This tool itself could use LLM to generate selectors or use a predefined list for known sites.
    # For now, it's a placeholder to show structure.
    # The actual implementation of site-agnostic scraping is complex.
    # A more realistic approach is a set of tools, each specialized for a few sites,
    # or a tool that takes a site type and applies a known strategy.

    class WebSearchAndExtractEventsInput(BaseModel):
        search_query: str = Field(description="The user's event search query, including keywords, location, and rough timeframe.")
        # target_sites: Optional[List[HttpUrl]] = Field(description="Optional list of specific website URLs to search within.")
        # num_results_to_process: int = Field(default=3, description="Number of search results or event pages to attempt to process.")

    @tool("general_web_event_search_and_initial_scrape", args_schema=WebSearchAndExtractEventsInput, return_direct=False)
    async def general_web_event_search_and_initial_scrape(search_query: str) -> List[Dict]:
        """
        Performs a general web search for events based on the query.
        (This tool is a simplified placeholder for a complex process).
        It should ideally identify event listing pages or specific event pages and attempt a basic scrape.
        Returns a list of initially scraped event data (potentially unstructured or semi-structured).
        The agent can then use `scrape_specific_event_page` for more detailed extraction if needed.

        For a real implementation, this tool would:
        1. Use a search engine (e.g., via an API or another LangChain tool like DuckDuckGoSearchRun) to find relevant URLs.
        2. For each promising URL, attempt to identify if it's an event page or a list of events.
        3. Perform a very generic scrape (e.g., extract all text, or try to find common event-like patterns).
        4. OR, if a known site, use a predefined strategy/selector set.
        This current version is a stub.
        """
        # This is a STUB implementation.
        # A real version would use Playwright to interact with a search engine or predefined event sites.
        print(f"Simulating web search and initial scrape for: {search_query}")
        # Example: If we had a Google Search tool:
        # search_results = await google_search_tool.arun(f"events {search_query}")
        # For each result, navigate and attempt basic scrape.

        # Placeholder result
        return [
            {
                "title": "Placeholder Event from Web Search",
                "description": "This is a placeholder found by the general web search tool for query: " + search_query,
                "url": "http://example.com/placeholder-event",
                "raw_text_content": "Some raw text about the placeholder event...",
                "_tool_comment": "This data requires further refinement using scrape_specific_event_page if a detailed schema is available."
            }
        ]

    # TODO: Add FastAPI lifespan event to close the browser when the app shuts down.
    # async def close_playwright_browser():
    #     global _browser_instance
    #     if _browser_instance:
    #         await _browser_instance.close()
    #         _browser_instance = None
    # Add to main.py: app.add_event_handler("shutdown", close_playwright_browser)
    ```
    **Note:** Managing the Playwright browser instance lifecycle within a FastAPI application is crucial. Using lifespan events (`startup` to initialize, `shutdown` to close) is the recommended approach for a shared browser instance. The current tool uses a simplified global that gets initialized on first use.

### Step 3.3: Backend - Integrate Tools into Agent
1.  **Action:** Update `backend/app/agents/event_agent.py` to include the new tools:
    ```python
    # backend/app/agents/event_agent.py
    # ... (other imports)
    from app.tools.datetime_parser_tool import datetime_parser
    from app.tools.web_scraper_tool import scrape_specific_event_page, general_web_event_search_and_initial_scrape

    # Add the new tools to the TOOLS list
    TOOLS = [
        datetime_parser,
        general_web_event_search_and_initial_scrape, # Agent uses this first
        scrape_specific_event_page # Agent uses this for detailed extraction from URLs found by the above
    ]

    # Update SYSTEM_PROMPT_TEMPLATE to guide the LLM on how/when to use these tools.
    # This is a critical step and requires careful crafting.
    SYSTEM_PROMPT_TEMPLATE = """
    You are an AI Event Agent. Your goal is to help users find events in the USA based on their queries.
    You have access to the following tools:
    1. `datetime_parser`: Use this to convert natural language timeframes (e.g., "next weekend", "July 15th") into specific start and end ISO date strings. Always call this tool if the user mentions a relative or unclear timeframe. Provide the current UTC date to it for context if possible.
    2. `general_web_event_search_and_initial_scrape`: Use this tool as your primary method to search for events on the web based on the user's query (keywords, location, parsed dates). This tool will return a list of potential events with basic information.
    3. `scrape_specific_event_page`: If `general_web_event_search_and_initial_scrape` returns a URL for a promising event, or if you independently find a specific event page URL, use this tool to extract detailed information. You need to provide this tool with an `extraction_schema` (a dictionary where keys are desired fields like 'title', 'description', 'start_datetime_str', 'location_str', 'address_str', 'organizer_str', and values are CSS selectors for those fields on the page). Try to determine sensible CSS selectors by examining the page structure if necessary (though you cannot directly browse, you can infer or use common patterns). If selectors are unknown, you might have to skip using this tool or use it with very generic selectors like 'body' to get all text, then parse with LLM.

    Workflow:
    1.  Understand the user's query: event type/keywords, location, timeframe.
    2.  If timeframe is unclear, use `datetime_parser` to get ISO start/end dates.
    3.  Use `general_web_event_search_and_initial_scrape` with the full query details (keywords, location, parsed dates).
    4.  Review the results from `general_web_event_search_and_initial_scrape`. For each promising event URL:
        a.  Decide if `scrape_specific_event_page` is needed for more details.
        b.  If so, formulate an `extraction_schema` with CSS selectors. (This is the hardest part for you. Try to use common semantic HTML tags like h1, p.event-description, span.date, div.location, or ask the user if they know the site structure if extraction fails repeatedly).
        c.  Call `scrape_specific_event_page`.
    5.  Consolidate all found event information.
    6.  If information is missing or ambiguous from the query, ask clarifying questions BEFORE extensive tool use. For example, if location is missing.
    7.  Present the final list of events to the user in a clear, structured format. Mention any difficulties in scraping or missing information.
    """

    # get_event_discovery_agent_executor() function remains largely the same,
    # but it will now use the populated TOOLS list and the updated prompt.
    # ... (rest of event_agent.py)
    ```
2.  **Action:** Test the agent with tools via LangServe playground (`/event-agent/playground`):
    - Query: `{"input": "Find AI conferences in California next month"}`.
    - Observe agent's thoughts, tool calls (`datetime_parser`, `general_web_event_search_and_initial_scrape`), and final output.
    - Debug prompts and tool logic as needed. This step is highly iterative.

### Step 3.4: Backend - Data Extraction & Normalization Chain (using LLM)
1.  **Action:** Create `backend/app/chains/event_parsing_chain.py`:
    ```python
    # backend/app/chains/event_parsing_chain.py
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
    from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel # For structured output
    from typing import Optional, List
    from datetime import datetime

    from app.core.llm_services import get_text_llm
    from app.models.event_models import Event # Target Pydantic model for output

    # Define the Pydantic model that the LLM should fill.
    # This matches our app.models.event_models.Event but might be a subset
    # or include fields like `confidence_score`.
    class ParsedEventData(LangchainBaseModel): # Must use langchain_core.pydantic_v1 for LLM structured output
        title: Optional[str]
        description: Optional[str] = None
        start_datetime_str: Optional[str] = Field(description="Extracted start date/time as a string, try to make it ISO 8601 compatible if possible.")
        end_datetime_str: Optional[str] = Field(description="Extracted end date/time as a string, try to make it ISO 8601 compatible if possible.")
        location_name: Optional[str] = None
        address: Optional[str] = None
        organizer: Optional[str] = None
        categories: Optional[List[str]] = None
        # We will attempt to convert *_str fields to datetime objects later.

    EVENT_PARSING_PROMPT_TEMPLATE = """
    You are an expert information extraction AI.
    Given the following raw text scraped from an event webpage (or a dictionary of semi-structured data),
    extract the specified event details and structure them according to the provided format.
    If a piece of information is not present, omit the field or set it to null.
    Focus on accuracy. Pay close attention to dates, times, and locations.

    Raw scraped data:
    ---
    {scraped_text_or_data}
    ---

    If the input is a dictionary, pay attention to its keys and values.
    If the input is raw text, parse it carefully.

    Desired information fields: title, description, start_datetime_str, end_datetime_str, location_name, address, organizer, categories.
    Ensure date strings are as complete as possible (including year, month, day, time, timezone if available).
    """

    def get_event_details_extraction_chain() -> Runnable:
        """
        Creates a LangChain Runnable that takes scraped text or a dictionary
        and extracts structured event details using an LLM.
        """
        llm = get_text_llm()

        # By May 2025, structured_output with Ollama models should be more robust.
        # We might use .with_structured_output(ParsedEventData) on the llm or prompt.
        # The prompt needs to guide the LLM to output in the desired JSON schema.
        # For now, the prompt instructs the LLM, and we might parse its JSON output.
        # LangChain's create_extraction_chain or similar utilities simplify this.

        # Assuming LLM can follow instructions to output JSON matching ParsedEventData schema.
        # If not, a more complex prompt with JSON examples or function calling (if supported well by Ollama model) would be needed.
        # LangChain's `with_structured_output` is the preferred way.
        structured_llm = llm.with_structured_output(ParsedEventData) # This is the modern way

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVENT_PARSING_PROMPT_TEMPLATE),
            # The {scraped_text_or_data} will be the input to the chain.
        ])

        # The chain: prompt -> LLM (with structured output)
        chain = prompt | structured_llm
        return chain

    # Example usage (will be integrated into the agent or a service)
    # async def parse_scraped_data_to_event(scraped_data: Union[str, Dict]) -> Optional[ParsedEventData]:
    #     extraction_chain = get_event_details_extraction_chain()
    #     try:
    #         # LangChain's with_structured_output expects the input to the chain to be a dict
    #         # where the key matches the variable in the prompt.
    #         if isinstance(scraped_data, str):
    #             input_for_chain = {"scraped_text_or_data": scraped_data}
    #         else: # It's a dict
    #             input_for_chain = {"scraped_text_or_data": str(scraped_data)} # Convert dict to string for generic prompt
    #
    #         parsed_data = await extraction_chain.ainvoke(input_for_chain)
    #         return parsed_data
    #     except Exception as e:
    #         print(f"Error during event data parsing: {e}")
    #         return None

    # This chain should be callable by the agent after scraping, or the agent itself can do the parsing.
    # A dedicated chain is cleaner. The agent could use a Tool that wraps this chain.
    ```
2.  **Action:** (Optional) Create a LangChain Tool that uses this extraction chain. This makes it easy for the agent to call.
    ```python
    # Potentially in backend/app/tools/data_processing_tool.py
    from langchain_core.tools import tool
    from app.chains.event_parsing_chain import get_event_details_extraction_chain, ParsedEventData
    from typing import Union, Dict, Optional

    @tool("parse_scraped_event_data", return_direct=False) # Input schema can be defined if needed
    async def parse_scraped_event_data(scraped_data: Union[str, Dict]) -> Optional[Dict]:
        """
        Takes raw scraped text or a semi-structured dictionary from a webpage
        and uses an LLM to parse it into a structured event format (ParsedEventData model).
        Returns the structured data as a dictionary, or None if parsing fails.
        """
        extraction_chain = get_event_details_extraction_chain()
        try:
            if isinstance(scraped_data, str):
                input_for_chain = {"scraped_text_or_data": scraped_data}
            else: # It's a dict
                input_for_chain = {"scraped_text_or_data": str(scraped_data)} # Or pass dict and adapt prompt

            parsed_event_model: ParsedEventData = await extraction_chain.ainvoke(input_for_chain)
            return parsed_event_model.dict() if parsed_event_model else None
        except Exception as e:
            print(f"Error in parse_scraped_event_data tool: {e}")
            # ToolException(f"Failed to parse event data: {e}")
            return {"error": f"Failed to parse event data: {e}"} # Return error info
    ```
3.  **Action:** Add this new `parse_scraped_event_data_tool` to `TOOLS` in `backend/app/agents/event_agent.py` and update the agent's system prompt to instruct it on how to use this tool after `scrape_specific_event_page` or `general_web_event_search_and_initial_scrape`.

    *Update `SYSTEM_PROMPT_TEMPLATE` in `event_agent.py`*:
    ```diff
    +   4. `parse_scraped_event_data`: After using `scrape_specific_event_page` or `general_web_event_search_and_initial_scrape`, if the output is raw text or semi-structured, use this tool to parse it into a clean, structured event object.
    ...
    Workflow:
    ...
    +   4d. (After scraping) Use `parse_scraped_event_data` on the scraped content to get structured event information.
    ...
    ```

4.  **Action:** (Post-Parsing Logic) Implement a Python function to convert `ParsedEventData` (or the dict from the tool) into the application's main `Event` model (`app/models/event_models.py`), including converting date strings to `datetime` objects. This would typically happen in a service layer after the agent returns the parsed data.
    ```python
    # Example (could be in a service file or agent's post-processing)
    # from app.models.event_models import Event
    # from app.chains.event_parsing_chain import ParsedEventData # Or the dict from the tool
    # from dateutil import parser as dateutil_parser

    # def convert_parsed_to_app_event(parsed_data: dict, source_url: str) -> Event:
    #     start_dt = None
    #     end_dt = None
    #     try:
    #         if parsed_data.get("start_datetime_str"):
    #             start_dt = dateutil_parser.parse(parsed_data["start_datetime_str"])
    #     except: pass # Add logging
    #     try:
    #         if parsed_data.get("end_datetime_str"):
    #             end_dt = dateutil_parser.parse(parsed_data["end_datetime_str"])
    #     except: pass # Add logging

    #     return Event(
    #         title=parsed_data.get("title", "N/A"),
    #         description=parsed_data.get("description"),
    #         start_datetime=start_dt,
    #         end_datetime=end_dt,
    #         location_name=parsed_data.get("location_name"),
    #         address=parsed_data.get("address"),
    #         organizer=parsed_data.get("organizer"),
    #         categories=parsed_data.get("categories", []),
    #         source_url=source_url, # Needs to be passed in or part of parsed_data
    #         raw_extracted_data=parsed_data
    #     )
    ```
    This conversion logic will be crucial when processing agent outputs before storing or sending to the frontend.

---
## Phase 4: Task Management System Development (Backend & API)

This phase focuses on creating FastAPI endpoints for tasks, integrating the scheduler, and storing task/event data.

### Step 4.1: Backend - Task CRUD API Endpoints
1.  **Action:** Create `backend/app/api/task_router.py`:
    ```python
    # backend/app/api/task_router.py
    from fastapi import APIRouter, HTTPException, Depends, Body, status
    from typing import List, Optional
    import uuid
    from datetime import datetime

    from app.models.task_models import TaskCreateRequest, TaskUpdateRequest, TaskResponse
    from app.models.event_models import EventScrapingResult, Event # For results
    # from app.services.task_service import TaskService # To be created
    # from app.services.task_scheduler_service import schedule_new_task, remove_scheduled_task # To be created

    # In-memory storage for tasks and results (Replace with DB in a real app)
    # This is a MAJOR simplification. A database (SQL or NoSQL) is essential for persistence.
    # For now, to keep it simple for the AI, we use in-memory dicts.
    # THIS MUST BE REPLACED WITH A DATABASE (e.g., PostgreSQL with SQLAlchemy/SQLModel, or MongoDB).
    TEMP_DB_TASKS: Dict[uuid.UUID, TaskResponse] = {}
    TEMP_DB_EVENT_RESULTS: Dict[uuid.UUID, List[Event]] = {} # Keyed by Task ID

    router = APIRouter()

    # Dependency for TaskService (if we create a service layer)
    # def get_task_service():
    #     return TaskService()

    @router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
    async def create_task(task_in: TaskCreateRequest):
        # task_service: TaskService = Depends(get_task_service)
        # return await task_service.create_task(task_in)

        # Simplified in-memory version:
        task_id = uuid.uuid4()
        new_task = TaskResponse(
            id=task_id,
            **task_in.dict(),
            status="PENDING",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        TEMP_DB_TASKS[task_id] = new_task
        if new_task.is_recurring and new_task.cron_schedule:
            # await schedule_new_task(new_task) # Integrate with APScheduler service
            print(f"Task {task_id} would be scheduled with cron: {new_task.cron_schedule}")
        return new_task

    @router.get("/", response_model=List[TaskResponse])
    async def get_all_tasks(skip: int = 0, limit: int = 100):
        # return await task_service.get_all_tasks(skip, limit)
        return list(TEMP_DB_TASKS.values())[skip : skip + limit]

    @router.get("/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: uuid.UUID):
        # task = await task_service.get_task_by_id(task_id)
        task = TEMP_DB_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return task

    @router.put("/{task_id}", response_model=TaskResponse)
    async def update_task(task_id: uuid.UUID, task_in: TaskUpdateRequest):
        # return await task_service.update_task(task_id, task_in)
        existing_task = TEMP_DB_TASKS.get(task_id)
        if not existing_task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        update_data = task_in.dict(exclude_unset=True)
        updated_task_response = existing_task.copy(update=update_data)
        updated_task_response.updated_at = datetime.utcnow()
        TEMP_DB_TASKS[task_id] = updated_task_response

        # Handle rescheduling if cron_schedule changed
        # old_schedule = existing_task.cron_schedule
        # new_schedule = updated_task_response.cron_schedule
        # if updated_task_response.is_recurring and old_schedule != new_schedule:
        #     if old_schedule: await remove_scheduled_task(str(task_id))
        #     if new_schedule: await schedule_new_task(updated_task_response) # Pass TaskResponse
        print(f"Task {task_id} would be updated and potentially rescheduled.")
        return updated_task_response

    @router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_task(task_id: uuid.UUID):
        # await task_service.delete_task(task_id)
        if task_id not in TEMP_DB_TASKS:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        task_to_delete = TEMP_DB_TASKS.pop(task_id)
        TEMP_DB_EVENT_RESULTS.pop(task_id, None) # Remove associated results
        # if task_to_delete.is_recurring and task_to_delete.cron_schedule:
        #     await remove_scheduled_task(str(task_id))
        print(f"Task {task_id} would be deleted and unscheduled.")
        return None # FastAPI handles 204

    # Endpoint to manually trigger a task (run the agent)
    @router.post("/{task_id}/run", response_model=APIResponse) # Should return updated TaskResponse or job ID
    async def run_task_now(task_id: uuid.UUID):
        from app.agents.event_agent import get_event_discovery_agent_executor # Local import
        from app.models.common_models import APIResponse

        task = TEMP_DB_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        if task.status == "RUNNING":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task is already running")

        task.status = "RUNNING"
        task.last_run_at = datetime.utcnow()
        TEMP_DB_TASKS[task_id] = task

        try:
            agent_executor = get_event_discovery_agent_executor()
            # Construct agent input from task.natural_language_query
            # And potentially pre-parsed fields like keywords, location, date_range if available.
            # For simplicity, just use the natural language query.
            agent_input = {"input": task.natural_language_query}
            # Add chat_history if conversational tasks are supported
            # agent_input["chat_history"] = []

            print(f"Running agent for task {task_id} with query: {task.natural_language_query}")
            agent_response = await agent_executor.ainvoke(agent_input)
            # The 'output' from the agent_response is what we need to process.
            # This output might be a string, or a dict if the agent is designed to produce structured output.
            # For now, assume it's a string or a dict containing event information.
            raw_agent_output = agent_response.get("output", str(agent_response))
            print(f"Raw agent output for task {task_id}: {raw_agent_output}")

            # TODO: Process raw_agent_output
            # 1. If it's a list of already structured Event objects (ideal but complex for agent to produce directly without specific output parser)
            # 2. If it's text, use an LLM chain (like event_parsing_chain) to extract Event objects.
            # 3. If it's a list of dicts from tools like parse_scraped_event_data_tool, convert them.

            # Placeholder for event processing and storage
            # For simplicity, let's assume the agent's output (after some parsing not shown here)
            # results in a list of Event models.
            # This processing step is CRITICAL and needs robust implementation.
            # For example, if agent_response['output'] contains a list of dictionaries that look like Event:
            # processed_events = [Event(**event_dict) for event_dict in extracted_event_data_list]

            # For this simplified plan, we'll store the raw output as a summary.
            # In a real system, you'd parse this into List[Event] and store properly.
            TEMP_DB_EVENT_RESULTS[task_id] = [] # Placeholder for actual List[Event]
            # Example of storing some dummy events:
            # TEMP_DB_EVENT_RESULTS[task_id] = [
            #     Event(title="Dummy Event 1", source_url="http://example.com/event1", start_datetime=datetime.utcnow()),
            #     Event(title="Dummy Event 2", source_url="http://example.com/event2", start_datetime=datetime.utcnow())
            # ]

            task.results_summary = f"Agent run completed. Raw output: {str(raw_agent_output)[:500]}..." # Truncate for summary
            task.status = "COMPLETED"

        except Exception as e:
            print(f"Error running agent for task {task_id}: {e}")
            task.status = "FAILED"
            task.results_summary = f"Agent run failed: {str(e)}"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        finally:
            task.updated_at = datetime.utcnow()
            TEMP_DB_TASKS[task_id] = task

        return APIResponse(success=True, message=f"Task {task_id} initiated and completed (simulated processing).", data=task.dict())


    @router.get("/{task_id}/results", response_model=EventScrapingResult) # Or a more generic TaskResult model
    async def get_task_results(task_id: uuid.UUID):
        task = TEMP_DB_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        # if task.status != "COMPLETED":
        #     raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Task results are not yet available. Status: {task.status}")

        events = TEMP_DB_EVENT_RESULTS.get(task_id, [])
        return EventScrapingResult(task_id=str(task_id), events_found=events)

    ```
2.  **Action:** Include this router in `backend/app/main.py`:
    ```python
    # backend/app/main.py
    # ...
    from app.api import task_router

    # ...
    # app.include_router(task_router.router, prefix="/api/v1/tasks", tags=["Tasks"])
    # Using "/api" as base for frontend, so let's adjust prefix.
    app.include_router(task_router.router, prefix="/api/tasks", tags=["Tasks"])
    ```
3.  **Action:** Test CRUD endpoints using FastAPI docs (`/docs`) or a tool like Postman/Insomnia.
    - Create a task.
    - Get tasks.
    - Run a task.
    - Get results (will be empty or placeholder initially).

### Step 4.2: Backend - Task Scheduler (APScheduler)
1.  **Action:** Create `backend/app/services/task_scheduler_service.py`:
    ```python
    # backend/app/services/task_scheduler_service.py
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    from apscheduler.triggers.cron import CronTrigger
    from pytz import utc
    import logging

    # from app.api.task_router import run_task_now_logic # Refactor run_task_now to be callable
    # from app.models.task_models import TaskResponse # For type hinting

    # Configure logging for APScheduler
    logging.basicConfig()
    logging.getLogger('apscheduler').setLevel(logging.INFO) # Or DEBUG for more verbosity

    # Global scheduler instance
    scheduler = None

    async def _job_function_wrapper(task_id_str: str):
        """
        Wrapper function to be called by the scheduler.
        It will invoke the actual task execution logic.
        """
        import uuid
        from app.api.task_router import TEMP_DB_TASKS # DANGER: Direct access to router's temp DB for simplicity
                                                     # In a real app, use a proper service layer or DB access.
                                                     # This creates a circular dependency potential if not careful.

        print(f"APScheduler: Running job for task_id: {task_id_str}")
        task_id_uuid = uuid.UUID(task_id_str)

        # This is where you would call the actual logic to run the agent for the task.
        # For example, refactor the core logic from `task_router.run_task_now`
        # into a separate async function that can be called from both the API endpoint and here.
        # e.g., `await execute_agent_for_task(task_id_uuid)`

        # Simplified: directly modify status (NOT recommended for production)
        task = TEMP_DB_TASKS.get(task_id_uuid)
        if task:
            print(f"APScheduler: Simulating run for task '{task.natural_language_query}'")
            # This should ideally call the same logic as the manual run endpoint.
            # For this plan, it's a placeholder. A robust implementation would involve:
            # 1. Fetching task details from a persistent DB.
            # 2. Calling the agent executor.
            # 3. Storing results in the DB.
            # 4. Updating task status in the DB.
            # await actual_task_execution_logic(task_id_uuid)
            task.status = "SCHEDULED_RUN_SIMULATED" # Indicate it was run by scheduler
            task.last_run_at = datetime.utcnow() # Not available in this simplified function
            TEMP_DB_TASKS[task_id_uuid] = task # Update in-memory store
            print(f"APScheduler: Task {task_id_str} status updated (simulated).")
        else:
            print(f"APScheduler: Task {task_id_str} not found for scheduled run.")


    def initialize_scheduler():
        global scheduler
        if scheduler is None:
            jobstores = {'default': MemoryJobStore()}
            executors = {'default': AsyncIOExecutor()}
            job_defaults = {'coalesce': False, 'max_instances': 1} # Only one instance of a job at a time
            scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=utc
            )
            scheduler.start()
            print("APScheduler initialized and started.")
        return scheduler

    async def schedule_new_task_job(task_id: str, cron_expression: str):
        sch = initialize_scheduler()
        try:
            sch.add_job(
                _job_function_wrapper,
                trigger=CronTrigger.from_string(cron_expression, timezone=utc),
                args=[task_id],
                id=task_id, # Job ID same as task ID
                replace_existing=True
            )
            print(f"Task {task_id} scheduled with cron: {cron_expression}")
            return True
        except Exception as e:
            print(f"Error scheduling task {task_id}: {e}")
            return False

    async def remove_scheduled_task_job(task_id: str):
        sch = initialize_scheduler()
        try:
            if sch.get_job(task_id):
                sch.remove_job(task_id)
                print(f"Task {task_id} unscheduled.")
            else:
                print(f"No scheduled job found for task {task_id} to remove.")
            return True
        except Exception as e:
            print(f"Error unscheduling task {task_id}: {e}")
            return False

    async def shutdown_scheduler():
        global scheduler
        if scheduler and scheduler.running:
            scheduler.shutdown()
            scheduler = None
            print("APScheduler shut down.")

    # Functions to be called by task_router when creating/updating/deleting recurring tasks
    # These are simplified to match the router's direct modification of TEMP_DB_TASKS.
    # Ideally, the router would call these, and these would update a persistent DB.

    async def handle_task_scheduling_on_create_or_update(task: 'TaskResponse'): # Use TaskResponse
        if task.is_recurring and task.cron_schedule:
            await schedule_new_task_job(str(task.id), task.cron_schedule)
        elif not task.is_recurring and task.cron_schedule: # Edge case: cron exists but not recurring
            await remove_scheduled_task_job(str(task.id))


    async def handle_task_unscheduling_on_delete(task_id: str):
        await remove_scheduled_task_job(task_id)

    ```
2.  **Action:** Integrate scheduler initialization and shutdown with FastAPI lifespan events in `backend/app/main.py`:
    ```python
    # backend/app/main.py
    # ...
    from contextlib import asynccontextmanager
    from app.services.task_scheduler_service import initialize_scheduler, shutdown_scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("Application startup...")
        initialize_scheduler()
        # Initialize other resources like DB connections here
        yield
        # Shutdown
        print("Application shutdown...")
        await shutdown_scheduler()
        # Close DB connections, Playwright browser, etc.
        # from app.tools.web_scraper_tool import close_playwright_browser # If defined
        # await close_playwright_browser()


    app = FastAPI(
        title="AI Event Agent API",
        version="0.1.0",
        description="API for managing AI event agent tasks and results.",
        lifespan=lifespan # Add lifespan manager
    )
    # ...
    ```
3.  **Action:** Update `backend/app/api/task_router.py` to call `handle_task_scheduling_on_create_or_update` and `handle_task_unscheduling_on_delete`:
    ```python
    # backend/app/api/task_router.py
    # ...
    from app.services.task_scheduler_service import (
        handle_task_scheduling_on_create_or_update,
        handle_task_unscheduling_on_delete
    )
    # ...

    @router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
    async def create_task(task_in: TaskCreateRequest):
        # ... (task creation logic)
        TEMP_DB_TASKS[task_id] = new_task
        await handle_task_scheduling_on_create_or_update(new_task) # Pass TaskResponse
        return new_task

    @router.put("/{task_id}", response_model=TaskResponse)
    async def update_task(task_id: uuid.UUID, task_in: TaskUpdateRequest):
        # ... (task update logic)
        # Need to get the old task's recurring status and cron before updating
        # old_task_is_recurring = existing_task.is_recurring
        # old_task_cron = existing_task.cron_schedule

        TEMP_DB_TASKS[task_id] = updated_task_response

        # If recurring status or cron changed, existing job might need removal before new one is added
        # This logic can be complex. handle_task_scheduling_on_create_or_update assumes
        # add_job with replace_existing=True handles most cases.
        # However, if a task becomes non-recurring, the job needs explicit removal.
        if not updated_task_response.is_recurring:
            await handle_task_unscheduling_on_delete(str(task_id))
        else: # Is recurring or became recurring
            await handle_task_scheduling_on_create_or_update(updated_task_response)
        return updated_task_response

    @router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_task(task_id: uuid.UUID):
        # ... (task deletion logic)
        task_to_delete_id_str = str(task_to_delete.id) # Get ID before popping
        if task_to_delete.is_recurring and task_to_delete.cron_schedule:
             await handle_task_unscheduling_on_delete(task_to_delete_id_str)
        # ...
    ```
4.  **Action:** Test scheduling by creating a task with a `cron_schedule` (e.g., `"* * * * *"` for every minute) and `is_recurring: true`. Observe logs for scheduler activity.
    **CRITICAL NOTE:** The `_job_function_wrapper` in `task_scheduler_service.py` currently only prints and simulates. It needs to be refactored to call the actual agent execution logic, similar to the `run_task_now` endpoint, and interact with a persistent database for task status and results. The current direct access to `TEMP_DB_TASKS` from the scheduler is problematic and for AI plan simplicity only.

---
## Phase 5: User Interface Development (SvelteKit) - Core Functionality

This phase focuses on creating the SvelteKit UI to interact with the backend API for tasks (web-scraping based).

### Step 5.1: Frontend - API Service Client
1.  **Action:** Create `frontend/src/lib/services/apiService.js` (or `.ts`):
    ```javascript
    // frontend/src/lib/services/apiService.js
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

    async function fetchAPI(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        const defaultHeaders = {
            'Content-Type': 'application/json',
            // Add Authorization header if/when auth is implemented
        };
        options.headers = { ...defaultHeaders, ...options.headers };

        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            if (response.status === 204) { // No Content
                return null;
            }
            return await response.json();
        } catch (error) {
            console.error(`API Error (${options.method || 'GET'} ${url}):`, error);
            throw error; // Re-throw to be caught by UI components
        }
    }

    export const taskService = {
        createTask: async (taskData) => fetchAPI('/tasks/', { method: 'POST', body: JSON.stringify(taskData) }),
        getAllTasks: async (skip = 0, limit = 100) => fetchAPI(`/tasks/?skip=${skip}&limit=${limit}`),
        getTaskById: async (taskId) => fetchAPI(`/tasks/${taskId}`),
        updateTask: async (taskId, taskData) => fetchAPI(`/tasks/${taskId}`, { method: 'PUT', body: JSON.stringify(taskData) }),
        deleteTask: async (taskId) => fetchAPI(`/tasks/${taskId}`, { method: 'DELETE' }),
        runTaskNow: async (taskId) => fetchAPI(`/tasks/${taskId}/run`, { method: 'POST' }),
        getTaskResults: async (taskId) => fetchAPI(`/tasks/${taskId}/results`),
    };

    // Example for LangServe agent endpoint (if direct interaction is needed beyond tasks)
    // export const agentService = {
    //     invokeAgent: async (inputPayload) => {
    //         const agentUrl = import.meta.env.VITE_AGENT_BASE_URL || "http://localhost:8000"; // Agent path is not under /api
    //         return fetch(`${agentUrl}/event-agent/invoke`, { // LangServe invoke endpoint
    //             method: 'POST',
    //             headers: { 'Content-Type': 'application/json' },
    //             body: JSON.stringify(inputPayload) // e.g. { "input": "user query", "config": {"configurable": {"session_id": "..."}}}
    //         }).then(res => {
    //             if (!res.ok) throw new Error(`Agent HTTP error! status: ${res.status}`);
    //             return res.json();
    //         });
    //     }
    // };
    ```

### Step 5.2: Frontend - Svelte Stores for State Management
1.  **Action:** Create `frontend/src/lib/stores/taskStore.js` (or `.ts`):
    ```javascript
    // frontend/src/lib/stores/taskStore.js
    import { writable } from 'svelte/store';
    import { taskService } from '$lib/services/apiService';

    export const tasks = writable([]); // Stores list of TaskResponse objects
    export const currentTask = writable(null); // Stores a single TaskResponse object being viewed/edited
    export const taskResults = writable(null); // Stores EventScrapingResult for a task
    export const isLoadingTasks = writable(false);
    export const error = writable(null);

    export async function loadTasks() {
        isLoadingTasks.set(true);
        error.set(null);
        try {
            const taskList = await taskService.getAllTasks();
            tasks.set(taskList);
        } catch (e) {
            error.set(e.message);
            tasks.set([]);
        } finally {
            isLoadingTasks.set(false);
        }
    }

    export async function loadTaskById(taskId) {
        isLoadingTasks.set(true); // Or a specific isLoadingCurrentTask
        error.set(null);
        try {
            const taskData = await taskService.getTaskById(taskId);
            currentTask.set(taskData);
            return taskData;
        } catch (e) {
            error.set(e.message);
            currentTask.set(null);
            throw e;
        } finally {
            isLoadingTasks.set(false);
        }
    }

    export async function createTask(taskData) {
        error.set(null);
        try {
            const newTask = await taskService.createTask(taskData);
            tasks.update(items => [newTask, ...items]); // Add to top
            return newTask;
        } catch (e) {
            error.set(e.message);
            throw e;
        }
    }
    // Add updateTask, deleteTask, runTask, loadTaskResults functions similarly
    export async function runTask(taskId) {
        error.set(null);
        try {
            const response = await taskService.runTaskNow(taskId);
            // Optionally update the specific task in the 'tasks' store or reload it
            loadTaskById(taskId); // To get updated status
            return response;
        } catch (e) {
            error.set(e.message);
            throw e;
        }
    }

    export async function loadTaskResultsFor(taskId) {
        error.set(null);
        taskResults.set(null);
        try {
            const results = await taskService.getTaskResults(taskId);
            taskResults.set(results);
            return results;
        } catch (e) {
            error.set(e.message);
            taskResults.set({ task_id: taskId, events_found: [], errors: [e.message] });
            throw e;
        }
    }
    ```

### Step 5.3: Frontend - Basic UI Components (Task List & Form)
1.  **Action:** Create `frontend/src/lib/components/TaskForm.svelte`:
    ```html
    <!-- frontend/src/lib/components/TaskForm.svelte -->
    <script>
        import { createTask, updateTask, currentTask } from '$lib/stores/taskStore'; // Assuming currentTask for edit
        import { onMount } from 'svelte';

        export let initialData = null; // Pass for editing
        export let formMode = 'create'; // 'create' or 'edit'
        export let onSuccess = () => {}; // Callback

        let formData = {
            natural_language_query: '',
            keywords: '', // Store as comma-separated string for input, convert on submit
            location: '',
            date_range_start: '',
            date_range_end: '',
            cron_schedule: '',
            is_recurring: false
        };
        let errorMsg = '';
        let isLoading = false;

        onMount(() => {
            if (formMode === 'edit' && initialData) {
                formData = {
                    natural_language_query: initialData.natural_language_query || '',
                    keywords: (initialData.keywords || []).join(', '),
                    location: initialData.location || '',
                    date_range_start: initialData.date_range_start ? initialData.date_range_start.substring(0, 16) : '', // Format for datetime-local
                    date_range_end: initialData.date_range_end ? initialData.date_range_end.substring(0, 16) : '',
                    cron_schedule: initialData.cron_schedule || '',
                    is_recurring: initialData.is_recurring || false,
                };
            }
        });

        async function handleSubmit() {
            isLoading = true;
            errorMsg = '';
            try {
                const payload = {
                    ...formData,
                    keywords: formData.keywords.split(',').map(k => k.trim()).filter(k => k),
                    // Ensure date_range_start/end are null if empty, or properly formatted
                    date_range_start: formData.date_range_start ? new Date(formData.date_range_start).toISOString() : null,
                    date_range_end: formData.date_range_end ? new Date(formData.date_range_end).toISOString() : null,
                };
                if (formMode === 'create') {
                    await createTask(payload);
                } else if (formMode === 'edit' && initialData?.id) {
                    await updateTask(initialData.id, payload); // This store function needs to be implemented
                }
                onSuccess();
            } catch (e) {
                errorMsg = e.message;
            } finally {
                isLoading = false;
            }
        }
    </script>

    <form on:submit|preventDefault={handleSubmit} class="space-y-4 p-4 border rounded-lg shadow">
        <div>
            <label for="nlq" class="block text-sm font-medium text-gray-700">Natural Language Query:</label>
            <textarea id="nlq" bind:value={formData.natural_language_query} required rows="3"
                      class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"></textarea>
        </div>
        <!-- Add other form fields for keywords, location, date_range_start/end (datetime-local), cron_schedule, is_recurring (checkbox) -->
        <div>
            <label for="location" class="block text-sm font-medium">Location:</label>
            <input type="text" id="location" bind:value={formData.location} class="mt-1 block w-full rounded-md border-gray-300 shadow-sm"/>
        </div>
        <div>
            <label for="cron" class="block text-sm font-medium">Cron Schedule (if recurring):</label>
            <input type="text" id="cron" bind:value={formData.cron_schedule} placeholder="* * * * *" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm"/>
        </div>
        <div>
            <label class="flex items-center">
                <input type="checkbox" bind:checked={formData.is_recurring} class="rounded border-gray-300 text-indigo-600 shadow-sm focus:ring-indigo-500"/>
                <span class="ml-2 text-sm text-gray-700">Is Recurring?</span>
            </label>
        </div>

        {#if errorMsg}
            <p class="text-sm text-red-600">{errorMsg}</p>
        {/if}
        <button type="submit" disabled={isLoading}
                class="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50">
            {isLoading ? 'Saving...' : (formMode === 'create' ? 'Create Task' : 'Update Task')}
        </button>
    </form>
    ```
2.  **Action:** Create `frontend/src/lib/components/TaskList.svelte`:
    ```html
    <!-- frontend/src/lib/components/TaskList.svelte -->
    <script>
        import { tasks, isLoadingTasks, error, loadTasks, runTask } from '$lib/stores/taskStore';
        import { onMount } from 'svelte';
        import { format } from 'date-fns'; // For date formatting

        onMount(async () => {
            if ($tasks.length === 0) { // Load only if not already loaded
                await loadTasks();
            }
        });

        async function handleRunTask(taskId) {
            // Add confirmation dialog if desired
            try {
                await runTask(taskId);
                // Optionally show a success notification
            } catch (e) {
                // Optionally show an error notification
                alert(`Failed to run task: ${e.message}`);
            }
        }
    </script>

    <div class="p-4">
        <h2 class="text-xl font-semibold mb-4">Tasks</h2>
        {#if $isLoadingTasks}
            <p>Loading tasks...</p>
        {:else if $error}
            <p class="text-red-500">Error loading tasks: {$error}</p>
        {:else if $tasks.length === 0}
            <p>No tasks found. Create one!</p>
        {:else}
            <ul class="space-y-3">
                {#each $tasks as task (task.id)}
                    <li class="p-3 border rounded-lg shadow-sm bg-white">
                        <div class="flex justify-between items-center">
                            <div>
                                <h3 class="font-medium text-indigo-700">
                                    <a href={`/tasks/${task.id}`} class="hover:underline">
                                        {task.natural_language_query.substring(0, 100)}{task.natural_language_query.length > 100 ? '...' : ''}
                                    </a>
                                </h3>
                                <p class="text-xs text-gray-500">ID: {task.id}</p>
                            </div>
                            <div class="text-right">
                                <span class="px-2 py-1 text-xs font-semibold rounded-full
                                    {task.status === 'COMPLETED' ? 'bg-green-100 text-green-800' :
                                    (task.status === 'PENDING' ? 'bg-yellow-100 text-yellow-800' :
                                    (task.status === 'RUNNING' ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'))}">
                                    {task.status}
                                </span>
                                <p class="text-xs text-gray-500 mt-1">
                                    Last Run: {task.last_run_at ? format(new Date(task.last_run_at), 'PPpp') : 'Never'}
                                </p>
                            </div>
                        </div>
                        <div class="mt-2">
                             <button on:click={() => handleRunTask(task.id)}
                                     disabled={task.status === 'RUNNING'}
                                     class="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50">
                                 Run Now
                             </button>
                             <!-- Add Edit/Delete buttons linking to appropriate actions/pages -->
                             <a href={`/tasks/${task.id}/edit`} class="ml-2 px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300">Edit</a>
                        </div>
                        {#if task.results_summary}
                            <p class="mt-1 text-xs text-gray-600 bg-gray-50 p-2 rounded">Summary: {task.results_summary}</p>
                        {/if}
                    </li>
                {/each}
            </ul>
        {/if}
    </div>
    ```

### Step 5.4: Frontend - SvelteKit Routes & Pages
1.  **Action:** Create main page `frontend/src/routes/+page.svelte`:
    ```html
    <!-- frontend/src/routes/+page.svelte -->
    <script>
        import TaskList from '$lib/components/TaskList.svelte';
        import TaskForm from '$lib/components/TaskForm.svelte';
        import { loadTasks } from '$lib/stores/taskStore'; // To refresh list after create

        function handleTaskCreated() {
            // Optionally navigate or just refresh the list
            loadTasks();
            // Consider resetting form or giving feedback
        }
    </script>

    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold text-center mb-8 text-indigo-600">AI Event Agent</h1>

        <div class="grid md:grid-cols-2 gap-8">
            <div>
                <h2 class="text-2xl font-semibold mb-4">Create New Task</h2>
                <TaskForm on:success={handleTaskCreated} />
            </div>
            <div>
                <TaskList />
            </div>
        </div>
    </div>
    ```
2.  **Action:** Create task detail page `frontend/src/routes/tasks/[id]/+page.svelte`:
    ```html
    <!-- frontend/src/routes/tasks/[id]/+page.svelte -->
    <script>
        import { page } from '$app/stores';
        import { currentTask, taskResults, loadTaskById, loadTaskResultsFor, error as taskErrorStore } from '$lib/stores/taskStore';
        import { onMount } from 'svelte';
        import { format }
        from 'date-fns';

        let taskId;
        let localError = null;
        let isLoadingDetails = true;
        let isLoadingResults = false;

        $: taskId = $page.params.id;

        onMount(async () => {
            if (taskId) {
                try {
                    await loadTaskById(taskId);
                    // Optionally autoload results if task is completed
                    if ($currentTask && $currentTask.status === 'COMPLETED') {
                        await handleLoadResults();
                    }
                } catch (e) {
                    localError = e.message;
                    taskErrorStore.set(e.message); // Also set global store if desired
                } finally {
                    isLoadingDetails = false;
                }
            }
        });

        async function handleLoadResults() {
            if (!taskId) return;
            isLoadingResults = true;
            localError = null;
            try {
                await loadTaskResultsFor(taskId);
            } catch (e) {
                localError = e.message;
            } finally {
                isLoadingResults = false;
            }
        }
    </script>

    <div class="container mx-auto p-4">
        {#if isLoadingDetails}
            <p>Loading task details...</p>
        {:else if !$currentTask || $taskErrorStore}
            <p class="text-red-500">Error loading task: {$taskErrorStore || localError || 'Task not found.'}</p>
            <a href="/" class="text-indigo-600 hover:underline">Go back to task list</a>
        {:else}
            <h1 class="text-2xl font-bold mb-2">Task: {$currentTask.natural_language_query}</h1>
            <p class="text-sm text-gray-600 mb-4">ID: {$currentTask.id}</p>

            <div class="bg-white shadow overflow-hidden sm:rounded-lg mb-6">
                <div class="px-4 py-5 sm:px-6">
                    <h3 class="text-lg leading-6 font-medium text-gray-900">Task Information</h3>
                </div>
                <div class="border-t border-gray-200 px-4 py-5 sm:p-0">
                    <dl class="sm:divide-y sm:divide-gray-200">
                        <div class="py-3 sm:py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                            <dt class="text-sm font-medium text-gray-500">Status</dt>
                            <dd class="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{$currentTask.status}</dd>
                        </div>
                        <!-- Add more fields: created_at, updated_at, cron_schedule, is_recurring etc. -->
                         <div class="py-3 sm:py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                            <dt class="text-sm font-medium text-gray-500">Created At</dt>
                            <dd class="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{format(new Date($currentTask.created_at), 'PPpp')}</dd>
                        </div>
                        {#if $currentTask.results_summary}
                        <div class="py-3 sm:py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                            <dt class="text-sm font-medium text-gray-500">Last Summary</dt>
                            <dd class="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{$currentTask.results_summary}</dd>
                        </div>
                        {/if}
                    </dl>
                </div>
            </div>

            <button on:click={handleLoadResults} disabled={isLoadingResults}
                    class="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50">
                {isLoadingResults ? 'Loading Results...' : 'Load/Refresh Results'}
            </button>

            {#if $taskResults && $taskResults.task_id === taskId}
                <h2 class="text-xl font-semibold mt-6 mb-3">Event Results</h2>
                {#if $taskResults.errors && $taskResults.errors.length > 0}
                    <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4" role="alert">
                        <p class="font-bold">Errors loading results:</p>
                        {#each $taskResults.errors as e}
                            <p>{e}</p>
                        {/each}
                    </div>
                {/if}
                {#if $taskResults.events_found && $taskResults.events_found.length > 0}
                    <ul class="space-y-4">
                        {#each $taskResults.events_found as event (event.source_url + event.title)}
                            <li class="p-4 border rounded-lg bg-gray-50">
                                <h3 class="text-lg font-medium text-indigo-700">{event.title}</h3>
                                {#if event.start_datetime}
                                <p class="text-sm text-gray-600">
                                    Date: {format(new Date(event.start_datetime), 'PPP')}
                                    {#if event.end_datetime && format(new Date(event.end_datetime), 'PPP') !== format(new Date(event.start_datetime), 'PPP')}
                                        - {format(new Date(event.end_datetime), 'PPP')}
                                    {/if}
                                </p>
                                {/if}
                                {#if event.location_name}<p class="text-sm text-gray-600">Location: {event.location_name}</p>{/if}
                                {#if event.description}<p class="text-sm text-gray-500 mt-1">{event.description.substring(0,200)}...</p>{/if}
                                <a href={event.source_url} target="_blank" rel="noopener noreferrer" class="text-xs text-blue-500 hover:underline">Source</a>
                            </li>
                        {/each}
                    </ul>
                {:else if !($taskResults.errors && $taskResults.errors.length > 0)}
                    <p>No events found for this task, or results not processed yet.</p>
                {/if}
            {/if}
            {#if localError && !$taskResults}
                <p class="text-red-500 mt-4">Could not load results: {localError}</p>
            {/if}
        {/if}
    </div>
    ```
3.  **Action:** Create an edit task page (e.g., `frontend/src/routes/tasks/[id]/edit/+page.svelte`) using `TaskForm.svelte` in 'edit' mode, loading initial data.
4.  **Action:** Update `frontend/src/app.html` and `+layout.svelte` for basic styling and navigation structure if needed.
    ```html
    <!-- frontend/src/routes/+layout.svelte -->
    <script>
        import '../app.css'; // If using Tailwind via app.css
    </script>

    <nav class="bg-gray-800 text-white p-4 shadow-md">
        <div class="container mx-auto flex justify-between items-center">
            <a href="/" class="text-xl font-bold hover:text-gray-300">AI Event Agent</a>
            <div>
                <a href="/" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">Home / Tasks</a>
                <!-- Add other nav links if needed -->
            </div>
        </div>
    </nav>

    <main class="container mx-auto mt-6 mb-6">
        <slot></slot>
    </main>

    <footer class="bg-gray-100 text-center p-4 text-sm text-gray-600 mt-auto">
        AI Event Agent &copy; {new Date().getFullYear()}
    </footer>
    ```

### Step 5.5: Frontend - Advanced UI Components (Placeholders for now)
1.  **Event Calendar:** Plan to integrate a Svelte calendar component (e.g., `svelte-calendar` or build a simple one). It would fetch tasks/events and display them. (Implementation deferred).
2.  **Dashboard:** Plan for a route/component that shows task statistics (counts by status, etc.). (Implementation deferred).

---
## Phase 6: Integration and Testing (Core System)

### Step 6.1: Full System - Test Core Functionality
1.  **Action:** Start both backend (`python backend/app/main.py`) and frontend (`npm run dev -- --open` in `frontend/` directory).
2.  **Action:** Perform end-to-end testing:
    - Create a task via UI (e.g., "Find tech meetups in San Francisco next week").
    - Verify task appears in the list.
    - Manually trigger the "Run Now" for the task from the UI.
    - Observe backend logs for agent activity (tool calls: `datetime_parser`, `general_web_event_search_and_initial_scrape`, `parse_scraped_event_data`).
    - Check task status updates in the UI.
    - View task results in the UI (should show placeholder/simulated events from the current stubbed tools or basic LLM parsing).
    - Test task editing and deletion.
    - Test recurring task creation (check backend logs for APScheduler messages).

### Step 6.2: Backend - Unit and Integration Tests
1.  **Action:** Write `pytest` tests for:
    - `backend/app/core/config.py`: Settings loading.
    - `backend/app/core/llm_services.py`: Mock Ollama and test LLM/Vision LLM initialization.
    - `backend/app/models/`: Pydantic model validation.
    - `backend/app/tools/datetime_parser_tool.py`: Test various natural language date inputs.
    - `backend/app/tools/web_scraper_tool.py`: Mock Playwright, test input/output schemas. Test `scrape_specific_event_page` with mock HTML and selectors.
    - `backend/app/chains/event_parsing_chain.py`: Test with sample scraped text and verify structured output (mock LLM response if needed for deterministic tests).
    - `backend/app/api/task_router.py`: Use `httpx.AsyncClient` to test FastAPI endpoints (CRUD, run). Mock agent execution to avoid real LLM calls during API tests.
    - `backend/app/agents/event_agent.py`: (More complex) Test agent logic with mocked tools and LLM. LangSmith evaluation datasets would be ideal here. For unit tests, can test prompt formatting.
    - `backend/app/services/task_scheduler_service.py`: Mock APScheduler and test job add/remove logic.

### Step 6.3: Frontend - Unit and E2E Tests
1.  **Action:** Write Vitest unit tests for:
    - `frontend/src/lib/services/apiService.js`: Mock `fetch`.
    - `frontend/src/lib/stores/taskStore.js`: Test state mutations and interactions with mocked `apiService`.
    - Critical Svelte components (e.g., `TaskForm.svelte` for form logic).
2.  **Action:** Write Playwright E2E tests for:
    - Task creation flow.
    - Task list display and navigation to detail page.
    - Task execution trigger.
    - Basic results display.

---
## Phase 7: Phone Interaction Module Development (High-Risk Component)

This phase is undertaken after the core web-scraping system is stable and tested.

### Step 7.1: Research & Prototyping (Phone Interaction)
1.  **Action:** Manually experiment with `scrcpy` and ADB for the target social media apps (Facebook, X, Threads) on an Android device/emulator.
    - Identify common UI elements for search, scrolling, event posts.
    - Understand navigation flows.
    - Capture screenshots of key UI states.
2.  **Action:** Prototype Vision LLM interaction:
    - Use `OllamaVision` (or equivalent) from `llm_services.py` with captured screenshots.
    - Prompt the Vision LLM to:
        - Locate UI elements (e.g., "Where is the search bar? Give coordinates or a bounding box.").
        - Describe screen content.
        - Identify if a post is an event.
    - Evaluate accuracy, latency, and consistency of Vision LLM responses.
3.  **Action:** Prototype basic ADB control via Python `subprocess` or `adbutils`:
    - Tapping at coordinates.
    - Swiping.
    - Inputting text.
4.  **Goal:** Determine feasibility and identify key challenges for reliable automation. Develop a strategy for handling UI changes (e.g., more abstract Vision LLM prompts, fallback OCR/CV).

### Step 7.2: Backend - `scrcpy` & ADB Infrastructure
1.  **Action:** Ensure `scrcpy` is installed and accessible in the backend environment. Ensure ADB server is running and device is connected/authorized.
2.  **Action:** Create `backend/app/utils/phone_control.py` (or similar) for `scrcpy`/ADB helper functions:
    ```python
    # backend/app/utils/phone_control.py
    import subprocess
    import asyncio
    import os
    from typing import Optional, Tuple

    # Requires adbutils: pip install adbutils
    # import adbutils

    # Adb path (configure if not in PATH)
    ADB_PATH = "adb" # or full path
    SCRCPY_PATH = "scrcpy" # or full path

    # async def get_adb_device():
    #     adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    #     try:
    #         devices = adb.device_list()
    #         if not devices:
    #             raise ConnectionError("No ADB devices found.")
    #         return adb.device(serial=devices.serial) # Use first device
    #     except Exception as e:
    #         print(f"ADB connection error: {e}")
    #         raise

    async def run_adb_command(command_args: list, device_serial: Optional[str] = None) -> Tuple[str, str]:
        cmd = [ADB_PATH]
        if device_serial:
            cmd.extend(["-s", device_serial])
        cmd.extend(command_args)
        # print(f"Executing ADB: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"ADB command failed: {stderr.decode().strip()} (command: {' '.join(cmd)})")
        return stdout.decode().strip(), stderr.decode().strip()

    async def tap_screen(x: int, y: int, device_serial: Optional[str] = None):
        await run_adb_command(["shell", "input", "tap", str(x), str(y)], device_serial)
        await asyncio.sleep(0.5) # Small delay after tap

    async def swipe_screen(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, device_serial: Optional[str] = None):
        await run_adb_command(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)], device_serial)
        await asyncio.sleep(0.5)

    async def input_text_adb(text: str, device_serial: Optional[str] = None):
        # Escaping special characters for ADB shell might be needed
        escaped_text = text.replace(" ", "%s").replace("'", "\\'") # Basic escaping
        await run_adb_command(["shell", "input", "text", escaped_text], device_serial)
        await asyncio.sleep(0.5)

    async def capture_screenshot_adb(output_path: str = "temp_screenshot.png", device_serial: Optional[str] = None) -> str:
        # Ensure output_path directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        # Capture to device, then pull
        device_path = "/data/local/tmp/screenshot.png"
        await run_adb_command(["shell", "screencap", "-p", device_path], device_serial)
        await run_adb_command(["pull", device_path, output_path], device_serial)
        await run_adb_command(["shell", "rm", device_path], device_serial) # Clean up
        return os.path.abspath(output_path)

    # scrcpy specific functions might be more complex if needing to control scrcpy process itself,
    # for now, assuming scrcpy is run independently for viewing, and ADB is used for control.
    # If scrcpy is used for screen mirroring TO the LLM, a different approach is needed to pipe frames.
    # For this plan, we use ADB for screenshots for simplicity to feed Vision LLM.
    ```

### Step 7.3: Backend - Phone Interaction Tool (`PhoneInteractionTool`)
1.  **Action:** Create `backend/app/tools/phone_interaction_tool.py`:
    ```python
    # backend/app/tools/phone_interaction_tool.py
    from langchain_core.tools import tool, ToolException
    from pydantic import BaseModel, Field
    from typing import List, Dict, Optional
    import base64
    import os

    from app.core.llm_services import get_vision_llm
    from app.utils.phone_control import (
        capture_screenshot_adb,
        tap_screen,
        swipe_screen,
        input_text_adb
    )
    # from langchain_core.messages import HumanMessage # For vision LLM if it takes messages

    # This tool will be stateful internally or require careful prompting if stateless.
    # The agent will need to call it iteratively: e.g., "analyze_screen", then "perform_action_on_phone".

    class PhoneActionInput(BaseModel):
        action_type: str = Field(description="Type of action: 'tap', 'swipe', 'input_text', 'capture_and_analyze', 'scroll_and_search'")
        app_name: Optional[str] = Field(description="Name of the app to interact with, e.g., 'Facebook', 'X'. For context.")
        # For tap/swipe
        x: Optional[int] = Field(description="X coordinate for tap/swipe start.")
        y: Optional[int] = Field(description="Y coordinate for tap/swipe start.")
        x2: Optional[int] = Field(description="X coordinate for swipe end.")
        y2: Optional[int] = Field(description="Y coordinate for swipe end.")
        # For input_text
        text_to_input: Optional[str] = Field(description="Text to input into a field.")
        # For capture_and_analyze
        analysis_prompt: Optional[str] = Field(default="Describe the current screen and identify interactive elements like buttons, search bars, and list items. Identify any potential events.", description="Prompt for the Vision LLM to analyze the screen.")
        # For scroll_and_search (higher-level conceptual action)
        search_target_description: Optional[str] = Field(description="Description of what to look for while scrolling, e.g., 'event posts about AI'")
        max_scrolls: Optional[int] = Field(default=3, description="Maximum number of scroll attempts.")

        device_serial: Optional[str] = Field(default=None, description="Optional Android device serial ID.")


    @tool("phone_interaction_agent", args_schema=PhoneActionInput, return_direct=False)
    async def phone_interaction_agent(
        action_type: str,
        app_name: Optional[str] = None,
        x: Optional[int] = None, y: Optional[int] = None,
        x2: Optional[int] = None, y2: Optional[int] = None,
        text_to_input: Optional[str] = None,
        analysis_prompt: Optional[str] = "Describe screen.",
        search_target_description: Optional[str] = None,
        max_scrolls: Optional[int] = 3,
        device_serial: Optional[str] = None
    ) -> Dict:
        """
        Interacts with a connected Android device via ADB and Vision LLM.
        Supported action_types:
        - 'tap': Taps at (x, y). Requires x, y.
        - 'swipe': Swipes from (x, y) to (x2, y2). Requires x, y, x2, y2.
        - 'input_text': Inputs text. Requires text_to_input.
        - 'capture_and_analyze': Captures screenshot, sends to Vision LLM with analysis_prompt. Returns LLM analysis and image path.
        - 'scroll_and_search': (Conceptual) Scrolls N times, capturing and analyzing screen for search_target_description. Returns findings. THIS IS COMPLEX.

        The Vision LLM is used by 'capture_and_analyze'. Coordinates for tap/swipe must be determined by prior analysis.
        """
        vision_llm = get_vision_llm()
        screenshot_path = "temp_phone_screenshot.png" # Define a consistent path

        try:
            if action_type == "tap":
                if x is None or y is None: raise ValueError("x and y coordinates required for tap.")
                await tap_screen(x, y, device_serial=device_serial)
                return {"status": "success", "action": "tap", "x": x, "y": y}

            elif action_type == "swipe":
                if x is None or y is None or x2 is None or y2 is None: raise ValueError("x, y, x2, y2 coordinates required for swipe.")
                await swipe_screen(x, y, x2, y2, device_serial=device_serial)
                return {"status": "success", "action": "swipe", "from": (x,y), "to": (x2,y2)}

            elif action_type == "input_text":
                if text_to_input is None: raise ValueError("text_to_input required.")
                await input_text_adb(text_to_input, device_serial=device_serial)
                return {"status": "success", "action": "input_text", "text": text_to_input}

            elif action_type == "capture_and_analyze":
                abs_screenshot_path = await capture_screenshot_adb(output_path=screenshot_path, device_serial=device_serial)
                with open(abs_screenshot_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # Vision LLM invocation (LangChain specific)
                # By May 2025, ChatOllama should handle base64 images in its standard message format.
                # Example: llm.invoke([HumanMessage(content=[{"type": "text", "text": analysis_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}])])
                # This is a simplified representation. Check LangChain docs for precise `OllamaVision` usage.
                # response = await vision_llm.ainvoke( ... image + prompt ... )
                # For now, assume a simplified invoke method for `ChatOllama` with vision.
                # The `invoke` method for multimodal models might require a specific input structure.
                # We might need to construct a list of messages.
                # Example using a common pattern for multimodal LLMs in LangChain:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ]
                # Note: The actual invocation structure might vary based on LangChain's OllamaVision wrapper.
                # It could be `vision_llm.invoke(prompt_with_image_data)` or using a `HumanMessage` as above.
                # Let's assume a simplified direct invoke for planning, but this needs to match the actual wrapper.
                # response = await vision_llm.ainvoke(f"{analysis_prompt}\n[IMAGE_BASE64_DATA_HERE: {img_base64}]") # This is too simple
                # The correct way would be to pass structured input if ChatOllama is multimodal
                # or use a specific method on OllamaVision.
                # Assuming ChatOllama supports a content list:
                from langchain_core.messages import HumanMessage
                llm_response_obj = await vision_llm.ainvoke([HumanMessage(content=messages["content"])])
                llm_analysis = llm_response_obj.content


                os.remove(abs_screenshot_path) # Clean up
                return {"status": "success", "action": "capture_and_analyze", "analysis": llm_analysis, "screenshot_path_removed": abs_screenshot_path}

            elif action_type == "scroll_and_search":
                # This is a complex multi-step process that the AGENT should orchestrate by calling
                # 'capture_and_analyze' and 'swipe' (for scroll) iteratively.
                # This tool action is too high-level for a single tool call if it involves LLM decisions mid-way.
                # Instead, the agent should:
                # 1. Call capture_and_analyze.
                # 2. LLM (agent) decides if target is found or if scroll is needed based on analysis.
                # 3. If scroll, agent calculates swipe coords (e.g., swipe up from bottom-center to top-center) and calls 'swipe'.
                # 4. Repeat.
                # This stub will just simulate one scroll and analysis.
                # A real implementation would be an agent loop.
                if not search_target_description:
                    raise ValueError("search_target_description required for scroll_and_search.")

                findings = []
                for i in range(max_scrolls or 1):
                    analysis_result = await phone_interaction_agent(
                        action_type="capture_and_analyze",
                        analysis_prompt=f"Current screen of app {app_name}. {analysis_prompt} Look for: {search_target_description}",
                        device_serial=device_serial
                    )
                    findings.append(analysis_result["analysis"])
                    # Simple check in LLM analysis (needs to be robust)
                    if search_target_description.lower() in analysis_result["analysis"].lower(): # Very naive check
                        return {"status": "success_target_potentially_found", "action": "scroll_and_search", "scroll_attempts": i+1, "findings": findings}

                    if i < (max_scrolls or 1) - 1: # Don't swipe on last iteration
                        # Generic scroll up (adjust coordinates for device resolution)
                        await swipe_screen(500, 1500, 500, 500, device_serial=device_serial) # Swipe from bottom-ish to top-ish
                        await asyncio.sleep(1) # Wait for UI to settle

                return {"status": "success_max_scrolls_reached", "action": "scroll_and_search", "scroll_attempts": max_scrolls, "findings": findings}

            else:
                raise ToolException(f"Unsupported phone_interaction_agent action_type: {action_type}")

        except Exception as e:
            print(f"Error in phone_interaction_agent ({action_type}): {e}")
            # ToolException should be raised if the tool itself cannot proceed or has a critical failure.
            # For user errors (e.g. missing params), ValueError is fine and caught by LangChain.
            raise ToolException(f"Phone interaction failed for action '{action_type}': {str(e)}") from e
    ```
2.  **Action:** Add `phone_interaction_agent` to `TOOLS` in `backend/app/agents/event_agent.py`.
3.  **Action:** Update the `SYSTEM_PROMPT_TEMPLATE` in `event_agent.py` to instruct the agent on how to use `phone_interaction_agent`. This will be very complex, as the agent needs to:
    - Decide which app to open (not covered by this tool, assume app is open).
    - Call `capture_and_analyze` to understand the screen.
    - Interpret the Vision LLM analysis to decide coordinates for `tap` or `swipe`.
    - Iterate this process for navigation (e.g., tap search -> input text -> tap search button -> analyze results -> scroll -> analyze).
    - This requires a highly capable agent and very clear, step-by-step prompting.
    *Example addition to prompt (highly simplified)*:
    ```diff
    +   5. `phone_interaction_agent`: Use this tool to interact with a connected Android phone to find events on social media apps.
    +      You need to call it iteratively. First, use `action_type="capture_and_analyze"` with a descriptive `analysis_prompt`.
    +      Based on the analysis, decide your next `action_type` ('tap', 'swipe', 'input_text') and its parameters (x, y coordinates, text).
    +      The `scroll_and_search` action is a high-level helper but you might need to break down complex searches into multiple calls.
    +      Be very specific with `analysis_prompt` to guide the Vision LLM. For example, "Analyze this Facebook screen. Where is the search events input field? Give its approximate center coordinates (x,y)."
    +      Then, if Vision LLM returns coordinates, use `action_type="tap"` with those coordinates.
    ```
4.  **Action:** Test iteratively. This will be the most challenging part. Start with simple sequences (e.g., capture screen, analyze, then manually provide coordinates for a tap based on your own viewing of the screenshot).

---
## Phase 8: Full System Integration, Testing, Optimization, and Deployment

This phase integrates the phone module fully and prepares for release.

### Step 8.1: Backend & Frontend - Integrate Phone Module Results
1.  **Action (Backend):** Modify `task_router.run_task_now` and the agent's output processing to handle events potentially found via `phone_interaction_agent`. This might involve a different data structure or a flag indicating the source. The agent's final output should consolidate events from web scraping and phone interaction.
2.  **Action (Backend):** Ensure the `Event` model and `EventScrapingResult` can accommodate data from phone sources (e.g., `source_app_name` field).
3.  **Action (Frontend):** Update SvelteKit components (`TaskList.svelte`, task detail page) to correctly display events sourced from phone interactions if they have distinct characteristics or data fields.

### Step 8.2: Comprehensive Testing (Including Phone Module)
1.  **Action:** Write E2E tests (manual or automated with extreme caution and mocks for phone interaction) that cover scenarios involving the `phone_interaction_agent`. This is very hard to automate reliably.
2.  **Action:** Conduct extensive manual user testing with the phone module active, targeting various social media apps and event search scenarios.
3.  **Action (Backend):** Add specific integration tests for `phone_interaction_tool` by mocking ADB/scrcpy calls and Vision LLM responses to test the tool's internal logic.

### Step 8.3: Performance Optimization & Reliability
1.  **Action (Backend):** Profile `phone_interaction_tool`, especially Vision LLM calls and ADB interactions. Optimize image handling (compression, resizing before sending to Vision LLM if appropriate) and minimize unnecessary ADB calls.
2.  **Action (Backend):** Implement robust error handling and retry mechanisms within `phone_interaction_tool` and the agent's logic when using it. Consider fallback strategies (e.g., OCR if Vision LLM fails consistently for certain UI elements).
3.  **Action (Backend & Frontend):** General performance review of API, database (if using one instead of `TEMP_DB_*`), and SvelteKit app.

### Step 8.4: Documentation and Deployment Preparation
1.  **Action:** Create comprehensive `README.md` files for both backend and frontend, covering setup, configuration, running, testing, and deployment.
2.  **Action (Backend):** Create `backend/Dockerfile` for containerizing the FastAPI/LangServe application.
    ```dockerfile
    # backend/Dockerfile
    FROM python:3.10-slim

    WORKDIR /app

    # Install Playwright dependencies
    RUN apt-get update && apt-get install -y \
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libdbus-1-3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libatspi2.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libxkbcommon0 \
        pango \
        libcairo2 \
        libasound2 \
        # ADB and scrcpy dependencies (scrcpy itself is usually a pre-built binary)
        # android-sdk-platform-tools-common contains adb, or install adb separately
        adb \
        # If scrcpy is to be run inside the container, its installation is more complex.
        # For this plan, scrcpy and the Android device are assumed external to this container,
        # and the container only needs ADB to connect to an (emulated or physical) device
        # accessible via network or USB passthrough.
        && rm -rf /var/lib/apt/lists/*

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Download Playwright browsers
    RUN playwright install --with-deps # Installs browsers and OS dependencies

    COPY ./app /app/app

    # Expose port
    EXPOSE 8000

    # Command to run the application
    # For production, use Gunicorn with Uvicorn workers
    # CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "app.main:app", "-b", "0.0.0.0:8000"]
    # For development/simplicity:
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
    **Note on Docker & scrcpy/ADB:** Running `scrcpy` and interacting with Android devices from within a Docker container can be complex due to USB/network access. The Dockerfile above primarily sets up ADB. `scrcpy` might be run on the host, or if inside Docker, it requires significant setup (e.g., X11 forwarding for display, or running headless and piping video, which is advanced). The plan assumes ADB can reach the device from where the backend runs.
3.  **Action (Frontend):** Configure SvelteKit adapter (e.g., `adapter-node` for Node.js server, `adapter-static` for static hosting, or platform-specific adapters like `adapter-vercel`). Build the frontend: `npm run build`.
4.  **Action:** Plan deployment strategy (e.g., Docker Compose for backend + SvelteKit Node server, or separate deployments for backend API and static frontend).

This detailed plan provides a step-by-step guide. The AI implementing this will need to:
- Fill in detailed logic for many functions.
- Write robust error handling.
- Iteratively debug, especially the agent prompts and tool interactions.
- **Crucially, replace all `TEMP_DB_*` in-memory stores in `backend/app/api/task_router.py` with a proper persistent database solution (e.g., PostgreSQL with SQLAlchemy or SQLModel, or a NoSQL DB like MongoDB with Motor/Beanie). This is a major simplification in the current plan to keep it manageable for AI generation but is not production-ready.**
- Refine the Playwright browser instance management for production.
- Make the `_job_function_wrapper` in the scheduler robustly call the agent execution logic and update a persistent DB.

Good luck to Gemini 2.5 Pro!
```