# AI Event Agent Project Description (April 2025)

## 1. Project Goal

To develop a local AI agent using **LangChain** that proactively finds events in the USA based on user requests. The agent will leverage web scraping through LangChain's Playwright integration, an innovative (though high-risk) phone interaction module via `scrcpy` for social media scraping, and a remote Ollama instance for LLM (including Vision) capabilities. The agent must support task creation, scheduling, natural language understanding for timeframes, and present results clearly to the user through an intuitive UI.

## 2. Core Functionality

* **Event Discovery:**
    * Scrape predefined popular event websites and calendars using LangChain's Playwright Browser Toolkit.
    * Scrape social media apps (Facebook, X, Threads) on an Android device connected via `scrcpy`, using a Vision LLM (or CV/OCR as fallback) to interpret the UI and navigate.
* **Task Management:**
    * Allow users to create event search tasks using natural language (e.g., "Find AI events in California this weekend").
    * Parse task descriptions using an LLM to extract keywords, location, and timeframes.
    * Interpret relative timeframes ("next week", "this weekend") into specific dates.
    * Support editing existing tasks and viewing task history.
* **Scheduling & Execution:**
    * Allow manual triggering of tasks.
    * Support scheduling tasks to run periodically (e.g., weekly).
    * Provide visual feedback on task execution status.
* **User Interface:**
    * Provide a comprehensive UI (Streamlit) for task creation, scheduling, manual execution, and viewing results.
    * Display a list of found events upon task completion with filtering and sorting options.
    * Include a dashboard view showing task statistics and upcoming scheduled runs.
    * Support responsive design for mobile and desktop access.
* **Intelligence:**
    * Utilize a remote Ollama server for:
        * Task parsing and planning (Text LLM).
        * Screen analysis for `scrcpy` interaction (Vision LLM).
        * Information extraction from scraped content (Text LLM).
        * Data deduplication and synthesis (Text LLM).

## 3. Technology Stack

* **Programming Language:** Python 3.10+
* **AI Agent Framework:** LangChain
* **LLM Backend:** Remote Ollama server hosting:
    * Text LLM (e.g., Llama 3, Mistral)
    * Vision LLM (e.g., LLaVA)
* **LLM Interaction:** `ollama-python` library or direct HTTP requests
* **Web Scraping:** 
    * LangChain's Playwright Browser Toolkit (primary)
    * `requests`, `BeautifulSoup4` (secondary/fallback)
* **Phone Interaction (`scrcpy` module):**
    * `scrcpy` (system installation)
    * Python `subprocess` module
    * Ollama Client (for Vision LLM calls)
    * `opencv-python` (optional/fallback for CV tasks)
    * `pytesseract`/`easyocr` (optional/fallback for OCR)
    * `adbutils` (or similar for ADB control)
* **Data Handling & Time:** `pandas`, `json`, `datetime`, `dateutil`
* **Task Scheduling:** `schedule` or `APScheduler`
* **User Interface:** Streamlit with custom components
* **Vector Database (Optional):** `chromadb-client` (for memory/RAG)
* **Dependency Management:** `pip` with `requirements.txt` or `Poetry`
* **Containerization (Ollama):** Docker

## 4. Architecture Overview

The system follows a modular architecture:

```mermaid
graph TD
    UI[User Interface (Streamlit - Task Mgmt & Results)] --> TaskManager[Task Manager / Scheduler (schedule/APScheduler)];
    TaskManager --> AgentCore{Agent Core / Orchestrator (LangChain)};

    AgentCore --> LLMInterface[LLM Interface (Ollama Client - Text & Vision)];
    LLMInterface --> RemoteOllama[Remote Ollama Server (Text + Vision Models)];

    AgentCore --> ToolSuite[Tool Suite];

    subgraph ToolSuite
        direction LR
        ScrapingTools[Web Scrapers (LangChain Playwright Toolkit)];
        PhoneTool[Phone Interaction Module (scrcpy + Vision LLM / CV/OCR)];
        DataTimeUtils[Data Processing & Time Utils (pandas, datetime, dateutil)];
    end

    PhoneTool -->|Controls/Reads/Screenshots| ScrcpyProcess[scrcpy Process];
    ScrcpyProcess <--> AndroidDevice[Android Device];
    PhoneTool -->|Image Analysis Request| LLMInterface;


    AgentCore --> MemoryDB[(Optional Memory DB)];

    style AgentCore fill:#f9f,stroke:#333,stroke-width:2px
    style ToolSuite fill:#ccf,stroke:#333,stroke-width:1px
    style PhoneTool fill:#fcc,stroke:#f00,stroke-width:2px,color:#f00
    style RemoteOllama fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style TaskManager fill:#cff,stroke:#333,stroke-width:2px
```

## 5. Core Component Descriptions

1.  **User Interface (UI - Streamlit):** Comprehensive frontend for user interaction featuring:
    * Task creation and editing interface with natural language input
    * Scheduling configuration with visual calendar
    * Task execution dashboard with status indicators
    * Results viewer with filtering and sorting capabilities
    * Settings panel for configuration options
2.  **Task Manager / Scheduler (schedule/APScheduler):** Manages the lifecycle of tasks. Stores task definitions (description, schedule), triggers `Agent Core` based on schedule or manual requests, and maintains task execution history.
3.  **Agent Core / Orchestrator (LangChain):** The brain of the operation. Receives tasks, uses LLM to parse requirements and dates, plans execution steps (tool calls), invokes tools from the `Tool Suite`, handles errors, manages conversational memory (optional), and orchestrates the flow until results are generated.
4.  **LLM Interface:** An abstraction layer to communicate with the remote Ollama server, handling both text and vision model requests and responses.
5.  **Tool Suite:** A collection of specialized modules:
    * **Web Scrapers:** Tools using LangChain's Playwright Browser Toolkit to navigate and extract data from predefined websites. The toolkit provides capabilities for navigating to URLs, clicking elements, extracting text and hyperlinks, and more.
    * **Phone Interaction Module:** The complex module managing `scrcpy` to stream the phone screen, send screenshots to the Vision LLM for analysis, interpret LLM instructions, and translate them into `scrcpy`/ADB commands (taps, swipes, text input) to interact with social media apps. Includes fallback CV/OCR logic if needed.
    * **Data & Time Utilities:** Functions for cleaning, standardizing, deduplicating event data, and parsing/calculating dates based on user input and current time.
6.  **Memory DB (Optional - ChromaDB):** Vector store for enabling Retrieval-Augmented Generation (RAG) or storing user preferences/past results for context.
7.  **Remote Ollama Server:** A separate machine/container running Ollama with the required text and vision language models.

## 6. Key Challenges and Risks

* **Fragility of `scrcpy` Integration:** This is the highest risk. UI updates in social media apps will break the phone interaction module frequently, requiring constant maintenance and adaptation, even with Vision LLM analysis.
* **Vision LLM Reliability & Latency:** Accuracy of screen interpretation and action generation by the Vision LLM is crucial but challenging. Network latency and model processing time will impact performance. Translating high-level instructions ("tap the search icon") into precise coordinates remains difficult.
* **Task/Time Parsing Accuracy:** The LLM's ability to correctly interpret user intent, locations, and complex timeframes ("every second Tuesday") is critical.
* **Scraping Instability:** Websites frequently change layouts and implement anti-scraping measures, requiring ongoing scraper maintenance. Risk of IP blocks or CAPTCHAs.
* **Scheduler Reliability:** Ensuring background tasks run reliably and handle errors gracefully.
* **Performance:** The `scrcpy` interaction loop (screenshot -> LLM analysis -> action) can be slow and resource-intensive on the local machine.
* **Ethics & Terms of Service:** Automated interaction with websites and especially mobile apps might violate their Terms of Service. Respect `robots.txt`.
* **LLM Quality:** Overall agent performance heavily depends on the capabilities of the chosen Ollama models and the quality of prompting.
* **UI Responsiveness:** Ensuring the Streamlit interface remains responsive even when handling multiple concurrent tasks or displaying large result sets.

## 7. Conclusion

This project outlines an advanced AI agent leveraging LangChain with its Playwright Browser Toolkit, LLMs (including Vision), web scraping, and a novel `scrcpy`-based phone interaction method. The enhanced UI provides comprehensive task management capabilities, making the system more user-friendly and productive. While powerful, the reliance on scraping and especially the `scrcpy`/Vision LLM module introduces significant maintenance overhead and fragility risks. A phased approach is recommended, starting with web scraping using LangChain's Playwright integration and basic task management before tackling the phone interaction module and advanced UI features.

