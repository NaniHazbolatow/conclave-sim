# ConclaveSim: A Multi-Agent Simulation of the Papal Election. [Extended]

ConclaveSim is an agent-based simulation framework that models the dynamics of papal elections using large language model (LLM) agents. Each cardinal is represented as a generative agent with a unique profile, ideological stance, and conversational memory. Agents participate in group discussions, update their internal stances via LLM-based reflection, and vote in sequential rounds until a two-thirds majority is achieved. The framework enables the study of how informal discussions, strategic alliances, and decentralized negotiation shape formal outcomes in highly constrained institutional settings.

**This repository extends the original ConclaveSim platform by introducing:**
- A stance-based embedding mechanism to track ideological convergence and polarization.
- A group assignment function that blends random allocation with utility-based optimization, based on social ties and ideological proximity.
- Flexible control over simulation parameters such as LLM temperature, group size, and rationality, allowing for systematic exploration of consensus formation, gridlock, and the emergence of dominant candidates.
- Support for both local and distributed (Snellius cluster) execution, with automated scripts for large-scale parameter sweeps and result collection.
- Usage of meta-llama/Llama-3.1-8B-Instruct compared to claude 3.7s sonnet

By building on the foundation of ConclaveSim, this project provides a powerful tool for researchers interested in the intersection of AI, social dynamics, and institutional decision-making, and demonstrates how LLMs can serve as generative social agents in complex simulations.

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

#### Command-Line Flags

You can customize the simulation run with the following flags:

- `--group {small,medium,large,xlarge,full}`: Select the active predefined group (overrides config.yaml groups.active)
- `--max-rounds N`: Set the maximum number of simulation rounds (1-50)
- `--parallel` / `--no-parallel`: Enable or disable parallel processing
- `--rationality R`: Set agent rationality (0.0=random to 1.0=fully rational)
- `--temperature T`: Set LLM temperature (0.0=deterministic to 2.0=creative)
- `--discussion-size N`: Number of agents per discussion group (2-20)
- `--output-dir DIR`: Custom output directory (default: auto-generated timestamp)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging level (overrides config.yaml)
- `--no-viz`: Disable visualization generation

**Examples:**

- Run with medium group and 10 rounds:
  ```bash
  uv run python simulations/run.py --group medium --max-rounds 10
  ```
- Run with custom rationality and temperature:
  ```bash
  uv run python simulations/run.py --rationality 0.9 --temperature 0.5
  ```
- Run with parallel processing disabled:
  ```bash
  uv run python simulations/run.py --no-parallel
  ```
- Run with custom discussion group size:
  ```bash
  uv run python simulations/run.py --discussion-size 5
  ```

**Local/Remote**
To run the simulation locally or remotely, the ```config.yaml``` file must be manually editted and set to either *remote* or *local*. The remote model requires the reation of a ```.env``` file with your openrouter api key set to ```OPENROUTER_API_KEY```

### Running a Local LLM or on Snellius

Running a local LLM requires a powerful GPU and substantial RAM. For large-scale or distributed runs, you can use the Snellius cluster. The workflow is as follows:

1. **Edit Job Templates**  
   - Navigate to `scripts/snellius/distribute/templates/`.
   - Modify `batch_script.sh` to set the parameters for a single simulation run (e.g., resources, environment variables, and command-line flags).
   - Other template files in this directory help automate job submission across multiple folders and parameter sets.

2. **Generate Directory Structure for Batch Runs**  
   - Use the `generate_tree.sh` script in `scripts/snellius/distribute/` to automatically create a directory tree for your batch jobs.
   - This script sets up folders for different parameter combinations and copies the necessary SLURM scripts into each.

   ```bash
   bash scripts/snellius/distribute/generate_tree.sh
   ```

3. **Submit Jobs to Snellius**  
   - After generating the directory structure, use the provided submission scripts (e.g., `submit_all.sh` or `submit_batch.sh` in the same directory) to submit your jobs to the cluster.

4. **Collect Results**  
   - Use the scripts in `scripts/snellius/collect/` to gather and organize results from completed runs.


> **Note:** This project uses Hugging Face for model loading. On Snellius, you must have your Hugging Face account configured (e.g., via `huggingface-cli login`) and have access to the required model (such as meta-llama/Llama-3.1-8B-Instruct) before launching jobs.
> Make sure to adjust resource requests and parameters in `batch_script.sh` according to your experiment needs and Snellius policies.
> For more details, see the comments in each script and the `scripts/snellius/` directory.

## Project Structure
Folders and important files in this repository:

```
conclave-sim/
├── analysis/                           # Jupyter notebooks for analysis
│   ├── conclave_analysis_new.ipynb
│   └── special_analysis.ipynb
├── conclave/                           # Core simulation code
│   ├── agents/                         # Agent classes and mixins
│   ├── config/                         # Configuration management
│   ├── embeddings/                     # Embedding utilities
│   ├── environments/                   # Simulation environments
│   ├── llm/                            # LLM management and tools
│   ├── network/                        # Network management
│   ├── prompting/                      # Prompt generation and templates
│   ├── utils/                          # Utility functions
│   └── visualization/                  # Visualization tools
├── config/                             # Additional config files and scripts
│   ├── groups.yaml
│   └── scripts/                        # Helper scripts for configuration and data loading
├── data/                               # Input data files (e.g., cardinals, networks)
│   ├── cardinals_master_data.csv
│   └── network/
├── results/                            # Results from 2500 snellius runs with 25 params
├── scripts/                            # Various scripts for data processing, analysis, and Snellius cluster
│   ├── analysis/                       # Voting analysis scripts
│   ├── data_processing/                # Data processing scripts
│   ├── network/                        # Network generation and processing
│   ├── results/                        # Results combination scripts
│   └── snellius/                       # Scripts for running/distributing jobs on Snellius cluster      
│       ├── distribute/
│       │   ├── templates/              # Contains files to submit jobs to snellius among ways to automate this
│       │   │   └── batch_script.sh     # Main snellius job script
│       │   └── generate_tree.sh        # Used to automatically generate a directory tree folder with chosen parameters and SLURM scripts
│       └──  collect/                   
├── config.yaml                         # Main configuration file
├── pyproject.toml                      # Python project metadata
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── uv.lock                             # Lock file for uv
├── simulations/                        # Simulation runner
│   └── run.py                          # Main entry point for running simulations
├── config.yaml                         # Main configuration file
├── pyproject.toml                      # Python project metadata
├── requirements.txt                    # Python dependencies
├── uv.lock                             # Lock file for uv
```
