# Logging Strategy for Conclave Simulator

This document outlines a proposed logging strategy for the Conclave simulator project, aiming to provide clarity, debuggability, and insights into the simulation's behavior.

## Core Principles

1.  **Leverage Python's `logging` Module:** Utilize the standard `logging` module for its flexibility and extensive features.
2.  **Dedicated Loggers:** Employ separate logger instances for different components or aspects of the application. This allows for fine-grained control over log levels, formatting, and output handlers.
3.  **Configurability:** Manage logging setup (levels, handlers, formatters) through the main `config.yaml` file.
4.  **Structured (Optional but Recommended):** Consider structured logging (e.g., JSON format) for easier parsing and analysis by external tools, especially for LLM I/O and agent actions.

## Proposed Logger Categories

### 1. Agent Conversation Logger
Tracks the high-level interactions and decisions of agents.
*   **Logger Name:** `conclave.agents`
*   **Information Logged:**
    *   Agent taking/generating a stance (summary).
    *   Agent sending/receiving messages (summary or type).
    *   Agent casting a vote.
    *   Agent's internal state changes relevant to discussion (e.g., reflection summaries).
*   **Log Level:** INFO for key events, DEBUG for detailed internal reasoning or raw outputs before summarization.
*   **Example Usage:**
    ```python
    agent_logger = logging.getLogger("conclave.agents")
    agent_logger.info(f"Agent {agent_id} (Round {round_num}): Takes stance - Prefers Candidate X due to Y.")
    agent_logger.debug(f"Agent {agent_id}: Raw reflection output before stance: {reflection_text}")
    ```

### 2. LLM I/O Logger
Captures the direct inputs and outputs to/from Large Language Models. Crucial for debugging prompt engineering and model responses.
*   **Logger Name:** `conclave.llm`
*   **Information Logged:**
    *   Full prompt sent to the LLM.
    *   Raw response received from the LLM.
    *   Model name, parameters used (if varying).
    *   Any errors during LLM communication.
    *   Tool calls requested by LLM and parameters.
*   **Log Level:** DEBUG for full prompts/responses, INFO for summaries or successful call notifications.
*   **Example Usage:**
    ```python
    llm_logger = logging.getLogger("conclave.llm")
    llm_logger.debug(f"LLM Input for Agent {agent_id} (Action: generate_stance): {full_prompt}")
    llm_logger.info(f"LLM Output for Agent {agent_id} (Action: generate_stance): Success, {len(response_text)} chars.")
    llm_logger.debug(f"LLM Raw Output: {response_text}")
    ```

### 3. System Internals Logger
General application lifecycle, configuration, errors, and other operational messages.
*   **Logger Name:** `conclave.system` (or root logger, and specific module loggers like `conclave.config`, `conclave.simulation`)
*   **Information Logged:**
    *   Application start/stop.
    *   Configuration loading (summary, excluding secrets).
    *   Simulation setup, progression through rounds/phases.
    *   Major errors or exceptions not caught elsewhere.
    *   Resource loading (e.g., data files).
*   **Log Level:** INFO for lifecycle events, DEBUG for detailed setup or flow, ERROR/CRITICAL for issues.
*   **Example Usage:**
    ```python
    system_logger = logging.getLogger("conclave.system")
    system_logger.info("Conclave simulation started. Config loaded from 'config.yaml'.")
    system_logger.debug(f"Effective configuration: {config_summary_dict}")
    ```

## Configuration (`config.yaml`)

Manage logging via Python's `logging.config.dictConfig`.
**Note:** For file handlers, the `filename` path should be made dynamic in the simulation script to point to the correct per-simulation output subfolder. The paths shown below are illustrative placeholders.

```yaml
# In config.yaml
logging:
  version: 1
  disable_existing_loggers: False
  formatters:
    simple:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    detailed:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    llm_io_format: # Potentially structured (JSON) or just distinct
      format: '%(asctime)s - LLM_IO - %(levelname)s - %(message)s'
    agent_conv_format:
      format: '%(asctime)s - AGENT_CONV - %(levelname)s - %(message)s'

  handlers:
    console:
      class: logging.StreamHandler
      level: DEBUG # Overall verbosity for console
      formatter: simple
      stream: ext://sys.stdout

    agent_file:
      class: logging.handlers.RotatingFileHandler # Example with rotation
      filename: placeholder_logs_dir/agent_conversations.log # DYNAMICALLY SET
      maxBytes: 10485760 # 10MB
      backupCount: 3
      level: INFO
      formatter: agent_conv_format

    llm_file:
      class: logging.handlers.RotatingFileHandler
      filename: placeholder_logs_dir/llm_io.log # DYNAMICALLY SET
      maxBytes: 10485760 # 10MB
      backupCount: 3
      level: DEBUG # Capture detailed I/O
      formatter: llm_io_format

    system_file:
      class: logging.handlers.RotatingFileHandler
      filename: placeholder_logs_dir/system.log # DYNAMICALLY SET
      maxBytes: 10485760 # 10MB
      backupCount: 3
      level: DEBUG
      formatter: detailed

  loggers:
    "conclave.agents":
      handlers: [console, agent_file]
      level: INFO # Default for agents, can be overridden by console handler's level
      propagate: no
    "conclave.llm":
      handlers: [console, llm_file]
      level: DEBUG
      propagate: no
    "conclave.system":
      handlers: [console, system_file]
      level: INFO
      propagate: no
    # Add other specific loggers as needed (e.g., "conclave.config")

  root: # Default for any logger not specified above
    handlers: [console, system_file]
    level: INFO
```

## Implementation Sketch

**Setup (e.g., in `conclave/config/manager.py` or main script):**
```python
import logging.config
import yaml # or your preferred YAML loader
import os

def setup_logging_from_config(config_data: dict, dynamic_logs_dir: str):
    log_config_dict = config_data.get("logging")
    if log_config_dict:
        try:
            # Dynamically update log file paths
            for handler_name, handler_config in log_config_dict.get("handlers", {}).items():
                if "filename" in handler_config: # Check if it's a file handler
                    original_filename = os.path.basename(handler_config["filename"])
                    handler_config["filename"] = os.path.join(dynamic_logs_dir, original_filename)
            
            logging.config.dictConfig(log_config_dict)
            logging.getLogger("conclave.system").info(f"Logging configured from dictConfig. Logs in: {dynamic_logs_dir}")
        except Exception as e:
            # Fallback to basic config if dictConfig fails
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).error(f"Failed to configure logging from dict: {e}. Using basicConfig.", exc_info=True)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("conclave.system").info("No 'logging' section in config. Using basicConfig.")

# In your application entry point or config loading:
# config = load_config_from_yaml("config.yaml") # Assuming this loads the full config dict
# simulation_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# base_output_dir = os.path.join("output", f"simulation_{simulation_timestamp}")
# logs_dir = os.path.join(base_output_dir, "logs")
# os.makedirs(logs_dir, exist_ok=True)
# setup_logging_from_config(config, logs_dir)
```

## Additional Logging Features to Consider

*   **Performance/Timing Logs:**
    *   Logger: `conclave.performance`
    *   Log duration of critical operations (LLM calls, simulation steps).
*   **Structured Logging (JSON):**
    *   Use libraries like `python-json-logger` for formatters.
    *   Makes logs machine-readable for analysis tools.
*   **Correlation IDs:**
    *   Trace a single "session" or "round" across multiple log messages and files.
    *   Can be managed with `threading.local()` or passed explicitly.
*   **Configuration Dump:**
    *   Log the effective configuration (excluding secrets) at startup to `conclave.system`.
*   **Per-Agent Debug Mode:**
    *   Allow enabling verbose logging for specific agent IDs via configuration for targeted debugging.

This logging strategy aims to provide a robust foundation for understanding and debugging the Conclave simulator.
