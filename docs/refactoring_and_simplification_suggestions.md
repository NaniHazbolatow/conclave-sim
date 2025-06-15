# Conclave Simulator - Refactoring Plan & Status

*Updated: June 14, 2025*

This document outlines the structured refactoring plan for the Conclave simulator, tracking completed work and detailing remaining implementation tasks.

## 🎉 MAJOR SUCCESS: Full Stance Generation Restored!

**✅ STANCE GENERATION NOW FULLY FUNCTIONAL** - Agents successfully generate detailed, meaningful stances using the `generate_stance` tool from `prompts.yaml`. Recent simulation logs show complete functionality with all 15 agents generating 75-125 word detailed stances, successful multi-round progression, and no critical errors.

## ✅ COMPLETED REFACTORING (Sections I-V)

### **Core Foundation & Configuration**
- **✅ Pydantic Configuration Models**: Full validation for `config.yaml` and `prompts.yaml` structures  
- **✅ Centralized Tool Definitions**: All tools (`cast_vote`, `speak_message`, `generate_stance`) loaded from `prompts.yaml`
- **✅ Critical Tool Calling Fix**: Resolved `'ToolDefinition' object has no attribute 'get'` error by properly converting Pydantic models to OpenAI-compatible dictionaries
- **✅ Configuration Management**: `ConfigManager` and `PromptLoader` with type-safe access and validation

### **Output Organization & Logging**  
- **✅ Timestamped Simulation Directories**: Each run creates organized folders (`logs/`, `visualizations/`, `results/`, `configs_used/`)
- **✅ Three-Log System**: Separated `system.log`, `agent_conversations.log`, `llm_io.log` with proper routing
- **✅ Dynamic Log Paths**: Logs automatically directed to simulation-specific directories

### **LLM Integration & Tool Calling**
- **✅ RobustToolCaller Functional**: Handles native and prompt-based tool calling with retry mechanisms
- **✅ Centralized Agent Tools**: `Agent.cast_vote()` and `Agent.discuss()` fetch tools dynamically from `prompts.yaml`
- **✅ Error Recovery**: Automatic retry with enhanced instructions (validated in successful test runs)

### **Visualization & Testing**
- **✅ CardinalVisualizer Working**: UMAP/PCA visualization with stance progression tracking
- **✅ Successful Integration Tests**: 15-agent simulations completing without errors
- **✅ Code Cleanup**: Removed unused `prompt_manager.py`, updated all naming conventions

### **System Status: FULLY FUNCTIONAL** ✅
- All 15 agents participate in discussions and voting
- Tool calling working from centralized `prompts.yaml` definitions  
- Complete logging coverage for debugging
- Visualization generation working correctly
- Multi-round simulations completing successfully

---

## 🔄 REMAINING REFACTORING TASKS

The core system is now stable and functional. The following sections detail the remaining implementation tasks for advanced features and optimizations.

## VI. Implement Centralized Logging Strategy

Goal: Transition from ad-hoc logging and `print` statements to a structured, configurable logging system as defined in `logging_strategy.md`. This will improve debuggability and provide clear, separated logs for different aspects of the simulation.

**Key Decisions from `logging_strategy.md`:**

*   **Logger Categories:** We will implement `conclave.agents`, `conclave.llm`, and `conclave.system` loggers. Specific modules like `conclave.config`, `conclave.simulation`, `conclave.embeddings`, and `conclave.visualization` will use sub-loggers (e.g., `logging.getLogger(__name__)` which will inherit from `conclave.system` or the root logger if not explicitly configured under `conclave.system`).
*   **Configuration:** Logging will be configured via the `logging` section in `config.yaml` and parsed by `logging.config.dictConfig`.
*   **Formatters:** We will use the `simple`, `detailed`, `llm_io_format`, and `agent_conv_format` text-based formatters as defined in `logging_strategy.md`. JSON structured logging will be deferred for simplicity at this stage.
*   **Dynamic Paths:** Log file paths for file handlers (`agent_file`, `llm_file`, `system_file`) will be dynamically updated to point to the unique, timestamped simulation output directory (created in Section II).
*   **Configuration Dump:** The effective non-sensitive configuration will be logged by `conclave.system` at startup.

**Implementation Steps:**

1.  **Update Pydantic Models for Logging Configuration:**
    *   **File:** `conclave/config/config_models.py`
    *   **Action:** Define or refine Pydantic models (`LoggingConfig`, `FormatterConfig`, `HandlerConfig`, `LoggerConfig`) that precisely match the structure of the `logging` section in `config.yaml` (as outlined in `logging_strategy.md`).
    *   **Explanation:** This ensures that the logging configuration is validated upon loading and can be accessed in a type-safe manner.

2.  **Refine Logging Setup in `ConfigManager`:**
    *   **File:** `conclave/config/config_loader.py`
    *   **Method:** `setup_logging_for_run(self, logs_dir: Path, log_file_name: Optional[str] = None)` (The `log_file_name` might become redundant or used for a primary system log if not all handlers are file-based or if a general log is desired in addition to specific ones. For now, we'll assume `log_file_name` is for a general system log, and specific file handlers in the config will have their base names).
        *   The method should be renamed or adapted to more closely match `setup_logging_from_config(config_data: dict, dynamic_logs_dir: str)` from `logging_strategy.md`. Let's aim for `ConfigManager.initialize_logging(self, logs_dir: Path)`.
    *   **Action:**
        *   Retrieve the `LoggingConfig` model (or its dictionary representation) from `self.config.logging`.
        *   Iterate through the `handlers` in the logging configuration. If a handler has a `filename` attribute, prepend the `logs_dir` path to it. For example, if `filename` is `agent_conversations.log`, it becomes `Path(logs_dir) / "agent_conversations.log"`.
        *   Call `logging.config.dictConfig()` with the modified logging configuration.
        *   Add a log message using `logging.getLogger("conclave.system")` to confirm logging has been initialized and to state where log files will be stored.
        *   Log the effective (non-sensitive parts of) `RootConfig` to the `conclave.system` logger at `INFO` or `DEBUG` level for reproducibility.
    *   **Explanation:** This centralizes logging setup and ensures all file-based logs are correctly routed to the simulation-specific output directory.

3.  **Systematic Removal of Old Logging Practices:**
    *   **Action:** Conduct a codebase-wide search for:
        *   Direct calls to `logging.basicConfig()`.
        *   Manual instantiation of `logging.FileHandler`, `logging.StreamHandler` outside the new configuration.
        *   Debugging `print()` statements that should be replaced with logger calls.
    *   **Explanation:** This step is crucial to ensure the new centralized logging system is the sole authority.

4.  **Implement New Loggers Across Modules:**
    *   For each relevant module/class:
        *   Get a logger instance: `logger = logging.getLogger(__name__)`. This is generally preferred as it uses the module's path for the logger name, allowing fine-grained configuration. For specific categories like agent or LLM logs, use the designated logger names directly:
            *   `agent_logger = logging.getLogger("conclave.agents")`
            *   `llm_logger = logging.getLogger("conclave.llm")`
            *   `system_logger = logging.getLogger("conclave.system")` (or module specific like `logging.getLogger("conclave.config")`)
        *   Replace `print()` statements and old logging calls with appropriate calls to the new logger (e.g., `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`).
    *   **Specific Modules to Target:**
        *   `conclave/agents/base.py` (and other agent classes): Use `conclave.agents` for agent actions, decisions, stance changes.
        *   `conclave/llm/client.py`, `conclave/llm/robust_tools.py`: Use `conclave.llm` for prompts, responses, tool calls, errors.
        *   `conclave/config/config_loader.py`, `conclave/config/prompt_loader.py`: Use `logging.getLogger(__name__)` (which could be configured under `conclave.config` or fall under `conclave.system` or `root`) for config loading events.
        *   `conclave/environments/conclave_env.py`: Use `logging.getLogger(__name__)` for environment setup, simulation progression.
        *   `conclave/embeddings/__init__.py`: Use `logging.getLogger(__name__)` for embedding model loading, caching.
        *   `conclave/visualization/cardinal_visualizer.py`: Use `logging.getLogger(__name__)` for visualization events.
        *   Main simulation scripts (in `simulations/` or `scripts/`): Use `conclave.system` for overall simulation lifecycle events.
    *   **Explanation:** This distributes the logging responsibility according to the new strategy, making logs more organized and filterable.

5.  **Verify Main Simulation Script Initialization:**
    *   **Action:** Ensure that the main script that runs a simulation performs the following steps in order:
        1.  Loads configuration using `get_config()`.
        2.  Creates the unique timestamped `base_output_dir` and `logs_dir` (as per Section II).
        3.  Calls `config_manager.initialize_logging(logs_dir)` (the new method name from step V.2).
        4.  Proceeds with the simulation.
    *   **Explanation:** Correct initialization is key to the entire logging system functioning as intended.

**Testing:** After each significant step in implementing the logging strategy (e.g., after refactoring `ConfigManager`, or after updating a major module like `agents`), **run the smokescreen tests (`tests/test_refactoring_smoke.py`)**. While these tests might not directly verify log content, they will help catch regressions or crashes introduced during the refactoring of logging calls. Consider adding specific tests for logging configuration loading if feasible, or manually inspecting log outputs from a test run.

## VII. Refine Agent and Environment Logic (`conclave/agents/`, `conclave/environments/`)

1.  **Pydantic for States and Messages:**
    *   Define Pydantic models for agent states, inter-agent messages, and environment state representations.
2.  **Simplify Agent Role/Behavior Determination:**
    *   If complex, centralize this logic, potentially driving it from configuration.
3.  **Ensure Clear Agent Lifecycle:**
    *   The perceive -> reason/deliberate -> act cycle should be explicit.
4.  **Refine `ConclaveEnv` State Management:**
    *   Interactions modifying the environment state must be through well-defined methods.
5.  **Implement Parallel Stance Generation with Progress Visualization:**
    *   **Timing and Trigger:**
        *   **Initial Stances (Round 1):** `ConclaveEnv` will orchestrate an initial stance generation phase *before* the first round of discussions begins.
        *   **Revised Stances (Subsequent Rounds):** For all rounds after the first, `ConclaveEnv` will orchestrate a stance generation phase *after* all discussions for that round are complete and *before* the voting phase for that round begins.
    *   **Orchestration in `ConclaveEnv`:**
        *   A method like `run_stance_generation_phase(self, agents_list: list, current_round: int, is_initial_round: bool)` will be called by the main simulation loop in `ConclaveEnv` at the appropriate times.
    *   **Parallel Execution:**
        *   Inside `run_stance_generation_phase`, use `concurrent.futures.ThreadPoolExecutor` to run the `agent.generate_stance(context, is_initial_stance)` method for each agent in parallel.
        *   The `context` provided to agents for initial stances will primarily be their background and the overall scenario. For revised stances, the `context` will include summaries of the discussions from the current round.
    *   **Progress Visualization:**
        *   Wrap the parallel execution loop with `tqdm` to provide a console progress bar.
        *   Example Snippet for `ConclaveEnv.run_stance_generation_phase`:\
            ```python
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            # Assuming 'agents_list' is a list of BaseAgent instances

            def run_stance_generation_phase(self, agents_list: list, current_round: int, is_initial_stance: bool):
                phase_description = "Initial Stances" if is_initial_stance else f"Revised Stances (Round {current_round})"
                self.logger.info(f"Starting generation of {phase_description.lower()} for {len(agents_list)} agents.")
                
                with ThreadPoolExecutor(max_workers=self.config.get_max_stance_workers()) as executor: # max_workers from config
                    futures = {}
                    for agent in agents_list:
                        # The context passed to generate_stance will vary based on is_initial_stance
                        context = self.get_context_for_stance_generation(agent, current_round, is_initial_stance)
                        futures[executor.submit(agent.generate_stance, context, is_initial_stance)] = agent
                    
                    for future in tqdm(as_completed(futures), total=len(agents_list), desc=f"Generating {phase_description}"):
                        agent = futures[future]
                        try:
                            future.result() # Wait for completion and retrieve potential errors
                            self.logger.debug(f"Agent {agent.id} successfully generated stance for round {current_round} ({'initial' if is_initial_stance else 'revised'}).")
                        except Exception as e:
                            self.logger.error(f"Agent {agent.id} failed to generate stance for round {current_round} ({'initial' if is_initial_stance else 'revised'}): {e}", exc_info=True)
                            # Handle failure: agent might keep previous stance (if not initial) or a default one.
                self.logger.info(f"{phase_description} generation complete for round {current_round}.")

            def get_context_for_stance_generation(self, agent, current_round: int, is_initial_stance: bool) -> Dict:
                # This method will prepare the appropriate context.
                # For initial stances: agent's background, overall scenario.
                # For revised stances: agent's background, scenario, and discussion summaries from current_round.
                # ... implementation details ...
                pass
            ```
    *   **Configuration:** The number of parallel workers (`max_workers`) for the `ThreadPoolExecutor` should be configurable in `config.yaml`.
    *   **Agent Method `BaseAgent.generate_stance(self, context: Dict, is_initial_stance: bool)`:**
        *   This method in `BaseAgent` will perform the LLM call to formulate/revise the stance.
        *   It will use the `is_initial_stance` flag and the provided `context` to tailor the LLM prompt (e.g., different system messages or user prompts for initial vs. revised stances).
        *   The method should update the agent's internal stance attribute (e.g., `self.current_stance`).
    *   **Redundant Logic Avoidance:** By having a single `run_stance_generation_phase` in `ConclaveEnv` and a single `generate_stance` method in `BaseAgent` that adapts based on context/flags, we avoid duplicating the core stance generation logic for initial and revised stances. The differentiation happens in the data/context provided and potentially minor conditional logic within the agent's prompt formation.

## VIII. Upgrade Embedding System (`conclave/embeddings/__init__.py`)

1.  **Improve `EmbeddingClient` Robustness and Features:**
    *   **Dependency Injection:** Modify classes that use embeddings (e.g., `CardinalVisualizer`, potentially agents) to accept an `EmbeddingClient` instance via their constructor instead of calling the global `get_default_client()`. This improves testability and explicitness.
        *   Update `get_default_client()` to perhaps be a utility for scripts or top-level instantiation, but not the primary way modules get the client.
    *   **Persistent Caching (Optional but Recommended):**
        *   Offer an option for disk-based caching of embeddings (e.g., using `joblib.Memory` or a simple file store). This can be controlled via `config.yaml`.
        *   ```python
          # Example with joblib
          # from joblib import Memory
          # location = './.joblib_cache' # Configurable
          # memory = Memory(location, verbose=0)
          # @memory.cache
          # def get_embedding_with_cache(text): ...
          ```
    *   **Configuration:** Ensure embedding model names, device preferences, and cache settings are configurable in `config.yaml`.
    *   **Error Logging:** In `_load_model`, when falling back to CPU, log the specific exception `e` that occurred.
    *   **Batch Embedding Method (Efficiency):** Add a method like `get_embeddings_batch(texts: List[str]) -> List[np.ndarray]` to `EmbeddingClient` that leverages the underlying sentence-transformer's ability to encode batches for better performance.

## IX. Enhance Visualization System (`conclave/visualization/cardinal_visualizer.py`)

1.  **Decouple and Configure `CardinalVisualizer`:**
    *   **Keep Separate from Embeddings:** Reaffirm the decision: `conclave.visualization` uses `conclave.embeddings` but they remain distinct modules.
    *   **Dependency Injection:**
        *   `CardinalVisualizer` constructor should accept an `EmbeddingClient` instance.
        *   It should also accept the specific `visualizations_dir` path (from step II.1) for saving plots.
    *   **Configuration-Driven Plotting:**
        *   Move all plotting parameters (t-SNE perplexity, `tsne_random_state`, PCA components, colors, titles, labels) to a `visualization` section in `config.yaml`.
        *   `CardinalVisualizer` should load and use these settings.
        *   Define a Pydantic model for this `visualization` configuration section.
    *   **Data Input Structure:** Define Pydantic models for the data structures expected by `visualize_stance_progression` (e.g., the format of `stance_evolution`).

2.  **Modularity for Future Plots:**
    *   For now, `visualize_stance_progression` is manageable. If more plot types are added, refactor `cardinal_visualizer.py` by creating more focused helper methods or distinct classes for different visualization types.

## X. General Code Health

1.  **Reduce Code Duplication & Improve Conciseness:** Continuously look for opportunities to refactor for brevity and clarity.
2.  **Leverage Libraries:** Utilize established libraries (Pydantic, `logging`, `matplotlib`, `scikit-learn`, etc.) appropriately.

This refactoring plan provides a roadmap. Prioritize based on impact on development velocity, system stability, and overall code clarity.
