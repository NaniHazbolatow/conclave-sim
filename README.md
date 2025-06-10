# ConclaveSim: Simulating the 2025 Papal Election with LLM Agents

## Contents

- [What is This?](#what-is-this)
- [Context: The Papal Election Process](#context-the-papal-election-process)
  - [The Conclave System](#the-conclave-system)
- [Experiment Setup](#experiment-setup)
  - [1. Environment](#1-environment-conclaveenv)
  - [2. Cardinal Agents](#2-cardinal-agents)
  - [3. Simulation Modes](#3-simulation-modes)
  - [4. Data Sources](#4-data-sources)
- [Running Experiments](#running-experiments)
- [Future work](#future-work)

## What is This?

ConclaveSim is a simulation framework that models the papal election process using Large Language Model (LLM) agents. Each cardinal is represented by an autonomous agent powered by Claude 3.7 Sonnet, with unique backgrounds and perspectives derived from real-world data about the College of Cardinals.

The simulation allows researchers and observers to explore the dynamics of papal elections by having AI agents engage in discussions and cast votes according to their simulated beliefs and priorities. The framework supports different modes of interaction including single voting rounds, multi-round elections, and discussion-based proceedings.

## Context: The Papal Election Process

### The Conclave System

When a Pope dies or resigns, the College of Cardinals gathers in the Sistine Chapel for a conclave to elect his successor. The term "conclave" comes from the Latin "cum clave" meaning "with a key," referring to the tradition of locking the cardinals inside until they reach a decision.

Key aspects of the papal conclave:

1. **Participants**: Only cardinals under 80 years of age on the day the Holy See becomes vacant may vote. These cardinals are known as "cardinal electors." There are **133 cardinal electors** for the 2025 election.

2. **Secrecy**: The conclave is conducted in absolute secrecy. The cardinals take an oath to maintain confidentiality about the proceedings.

3. **Voting Process**:
   - Cardinals cast paper ballots for their preferred candidate
   - A two-thirds supermajority is required to elect a new Pope
   - Voting occurs in multiple rounds until a Pope is elected
   - After each unsuccessful round, the ballots are burned with chemicals that produce black smoke
   - When a Pope is elected, white smoke signals the decision

4. **Discussion Periods**: Between voting sessions, cardinals engage in informal discussions where they can advocate for candidates and build consensus.

5. **Duration**: Conclaves can last from a few days to several weeks, depending on how quickly consensus forms around a candidate.

## Experiment Setup

This simulation framework features several components:

### 1. Environment (`ConclaveEnv`)

The central simulation environment that manages:
- Instantiating cardinal agents
- Executing voting rounds and tallying results
- Executing discussion rounds where cardinals can speak
- Recording history of voting results and discussions

### 2. Cardinal Agents

Each cardinal is represented by an LLM-powered agent with:
- Real-world identity based on actual cardinals
- Background information describing their views and priorities
- Ability to vote for candidates
- Ability to participate in discussions
- Memory of previous votes and discussions

### 3. Simulation Modes

The framework supports several simulation modes:

- **Single Round** (`single_round.py`): Runs a single voting round without discussion
- **Multi-Round** (`multi_round.py`): Conducts multiple voting rounds until a winner is elected. The only information shared among agents is the ballot results after each round.
- **Discussion-Based** (`discussion_round.py`): Conducts multiple voting rounds until a winner is elected.

### 4. Data Sources

Cardinal information is sourced from:
- `cardinal_electors_2025.csv`: Contains data on all eligible cardinal electors including their name, country, office, age, and background
- This data is generated using `generate_cardinals_csv.py`, which scrapes Wikipedia for the elector list and enhances each cardinal's information with GPT4.1 + Web Search

### Running Experiments

To run a simulation:

1. Install dependencies: `uv sync`
2. Configure your LLM backend in `config.yaml`:
   - For local models: Set `backend: "local"` and choose a model
   - For remote models: Set `backend: "remote"`, choose a model, and set your API key
3. Choose a simulation mode:
   ```
   uv run single_round.py    # For a single voting round
   uv run multi_round.py     # For multiple voting rounds
   uv run discussion_round.py  # For voting rounds with discussion periods
   ```

3. Results are logged to the `logs/` directory

### Future work

The simulation can be extended by:
- Experimenting with different LLMs
- Implementing different discussion or voting mechanics
- Creating new analysis tools to interpret simulation results
