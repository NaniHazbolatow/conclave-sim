# ConclaveSim: A Multi-Agent Simulation of the Papal Election

## Introduction

ConclaveSim is a sophisticated, multi-agent simulation framework designed to model the intricate dynamics of the papal election process. Leveraging large language models (LLMs), this project provides a unique platform for exploring the strategic interactions, discussions, and voting patterns that characterize a papal conclave. Each cardinal is represented by an autonomous agent, endowed with a distinct personality, background, and set of priorities, all derived from real-world data.

The simulation allows for a deep dive into the complex decision-making processes of the College of Cardinals, offering insights into how consensus is built, how factions emerge, and how a new Pope is ultimately elected. By providing a configurable and extensible environment, ConclaveSim serves as a powerful tool for researchers, historians, and anyone interested in the intersection of artificial intelligence, social dynamics, and religious studies.

## System Architecture

ConclaveSim is built on a modular and extensible architecture, designed to facilitate experimentation and future development. The key components of the system are:

### Core Components

*   **`ConclaveEnv`**: The central environment that orchestrates the simulation. It is responsible for managing the lifecycle of the agents, running the voting and discussion rounds, and collecting data.
*   **`Agent`**: The fundamental unit of the simulation, representing a single cardinal. The `Agent` class is designed with a modular, mixin-based architecture, allowing for the flexible composition of behaviors such as voting, discussion, and reflection.
*   **`LLMClientManager`**: A centralized manager for handling interactions with the LLM backend. This component ensures that all LLM clients are created and managed consistently, improving the maintainability and scalability of the system.
*   **`UnifiedPromptVariableGenerator`**: A modular and extensible prompt generator that creates the prompts used to elicit responses from the LLM agents. This component is designed to be easily adaptable to new scenarios and prompt structures.

### Data and Configuration

*   **`config.yaml`**: The main configuration file for the simulation. It allows for the configuration of the simulation parameters, LLM settings, and other key aspects of the environment.
*   **`prompts.yaml`**: A file containing the prompt templates used to generate the prompts for the LLM agents.
*   **`cardinals_master_data.csv`**: The primary data source for the simulation, containing detailed information about each cardinal, including their background, views, and priorities.

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   `uv` (to install dependencies)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ConclaveSim.git
    cd ConclaveSim
    ```
2.  Install the dependencies:
    ```bash
    uv sync
    ```

### Configuration

1.  Create a `config.yaml` file in the root directory of the project. You can use the `config.yaml.example` file as a template.
2.  Configure your LLM backend in the `config.yaml` file. You can choose between a local or remote backend.
3.  If you are using a remote backend, make sure to set your API key in the `config.yaml` file.

### Running the Simulation

To run the simulation, you can use the `run.py` script in the `simulations` directory:

```bash
uv run python simulations/run.py
```

The simulation results will be saved in the `simulations/outputs` directory.

## Simulation Modes

ConclaveSim supports several simulation modes, each designed to explore different aspects of the papal election process:

*   **Single Round**: A single voting round without any discussion.
*   **Multi-Round**: Multiple voting rounds, with the results of each round being shared with the agents before the next round begins.
*   **Discussion-Based**: Multiple voting rounds, with discussion periods between each round. During the discussion periods, the agents can interact with each other and try to build consensus.

## Future Work

ConclaveSim is an ongoing project, and there are many opportunities for future development and improvement. Some potential areas for future work include:

*   **Experimenting with different LLMs**: The simulation can be extended to support a wider range of LLMs, allowing for a comparative analysis of their performance.
*   **Improving the agent models**: The agent models can be made more sophisticated by incorporating more detailed psychological and sociological models of human behavior.
*   **Adding new simulation modes**: New simulation modes can be added to explore different aspects of the papal election process, such as the role of the media or the influence of external events.
*   **Developing a graphical user interface**: A graphical user interface (GUI) could be developed to make the simulation more accessible to a wider audience.
