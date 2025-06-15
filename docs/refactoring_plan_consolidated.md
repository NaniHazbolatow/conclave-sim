# Conclave Simulator - Refactoring Plan & Status

*Updated: June 15, 2025*

## 🎉 MAJOR SUCCESS: Full System Refactoring Complete!

**✅ STANCE GENERATION NOW FULLY FUNCTIONAL** - Agents successfully generate detailed, meaningful stances using the `generate_stance` tool from `prompts.yaml`. Recent simulation logs show complete functionality with all 15 agents generating 75-125 word detailed stances, successful multi-round progression, and no critical errors.

**🎉 CONFIGURATION SYSTEM FULLY REFACTORED** - Successfully split monolithic configuration into modular YAML files and separated prompt management into dedicated `conclave/prompting/` module. System now fully operational with improved maintainability and clear separation of concerns.

## ✅ COMPLETED REFACTORING (Sections I-V)

### **Core Foundation & Configuration**
- **✅ Pydantic Configuration Models**: Full validation for all configuration structures  
- **✅ Centralized Tool Definitions**: All tools (`cast_vote`, `speak_message`, `generate_stance`) loaded from `prompts.yaml`
- **✅ Critical Tool Calling Fix**: Resolved `'ToolDefinition' object has no attribute 'get'` error by properly converting Pydantic models to OpenAI-compatible dictionaries
- **✅ Configuration Management**: Complete modular configuration system with type-safe access and validation

### **✅ Configuration System Refactoring - COMPLETED**

**Goal:** Split monolithic `config.yaml` into focused configuration files for better maintainability.

**✅ STATUS: FULLY IMPLEMENTED** - Configuration system successfully split into modular YAML files with proper separation of concerns.

#### **✅ A. New Configuration Structure - IMPLEMENTED**
```
config/
├── agent.yaml        # Agent behavior, LLM settings, tool configurations  
├── simulation.yaml   # Voting rounds, discussion settings, candidate selection
├── output.yaml       # Logging handlers, visualization, file paths
└── testing_groups.yaml # Testing overrides and group configurations
```

**✅ Additional Achievement:** Separated prompt management into dedicated module:
```
conclave/prompting/    # Renamed from conclave/config/
├── prompts.yaml      # Moved from data/ - all prompt templates and tools
├── prompt_loader.py  # Centralized prompt loading and validation
├── prompt_models.py  # Pydantic models for prompt validation
└── prompt_variable_generator.py # Dynamic variable generation for templates
```

#### **✅ B. Configuration Components - IMPLEMENTED**

**✅ Actual Configuration Files Created:**

1. **Agent Configuration (`agent.yaml`)**
   ```yaml
   llm:
     backend: "local"  # or "remote"
     local:
       model_name: "microsoft/DialoGPT-medium"
       device: "auto"
     remote:
       model_name: "gpt-4"
       api_key: "${OPENAI_API_KEY}"
   
   tool_calling:
     max_retries: 3
     retry_delay: 1.0
   ```

2. **Simulation Configuration (`simulation.yaml`)**
   ```yaml
   num_cardinals: 15
   max_election_rounds: 10
   supermajority_threshold: 0.67
   discussion_group_size: 3
   
   discussion_length:
     min_words: 100
     max_words: 200
   
   stance_generation:
     min_words: 75
     max_words: 125
   ```

3. **Output Configuration (`output.yaml`)**
   ```yaml
   logging:
     handlers:
       system:
         level: "INFO"
       agent:
         level: "DEBUG"
       llm:
         level: "DEBUG"
   
   visualization:
     tsne:
       perplexity: 30
       learning_rate: 200
     umap:
       n_neighbors: 15
       min_dist: 0.1
     pca:
       n_components: 2
   ```

4. **Testing Groups Configuration (`testing_groups.yaml`)**
   ```yaml
   enabled: true
   overrides:
     num_cardinals: 5
     candidate_ids: [0, 1, 2, 3, 4]
     discussion_group_size: 2
   ```

#### **✅ C. Implementation Tasks - COMPLETED**

**✅ 1. Configuration Structure**
- ✅ Created modular configuration files (`agent.yaml`, `simulation.yaml`, `output.yaml`, `testing_groups.yaml`)
- ✅ Implemented `ConfigAdapter` for unified configuration access
- ✅ Added configuration file inheritance and override system
- ✅ Separated prompt management into dedicated `conclave/prompting/` module

**✅ 2. Pydantic Models**
- ✅ Created comprehensive validation models in `config/scripts/models.py`
- ✅ Implemented type-safe configuration loading with validation
- ✅ Added default values and environment variable support
- ✅ Created specialized models for tool definitions and prompt templates

**✅ 3. Migration & Integration**
- ✅ Successfully migrated from monolithic configuration
- ✅ Updated all imports from `conclave.config` to `conclave.prompting`
- ✅ Moved `data/prompts.yaml` to `conclave/prompting/prompts.yaml`
- ✅ Integrated with existing simulation infrastructure

#### **✅ D. Testing Results - VERIFIED**

**✅ 1. Configuration Loading**
- ✅ Verified multi-file loading with proper precedence
- ✅ Tested environment variable overrides
- ✅ Validated default value handling
- ✅ Confirmed error handling for malformed configurations

**✅ 2. Pydantic Models**
- ✅ Tested comprehensive model validation
- ✅ Verified automatic type conversion
- ✅ Validated required field enforcement
- ✅ Tested custom validators for complex configurations

**✅ 3. Migration Success**
- ✅ Backward compatibility maintained during transition
- ✅ All existing functionality preserved
- ✅ New modular structure fully operational
- ✅ Complete simulation runs successful with new configuration system

**🎉 ACHIEVEMENT: Configuration system refactoring completed successfully after intensive debugging session to resolve import path issues and Python bytecode cache conflicts.**

### **Output Organization & Logging**  
- **✅ Timestamped Simulation Directories**: Each run creates organized folders (`logs/`, `visualizations/`, `results/`, `configs_used/`)
- **✅ Three-Log System**: Separated `system.log`, `agent_conversations.log`, `llm_io.log` with proper routing
- **✅ Dynamic Log Paths**: Logs automatically directed to simulation-specific directories

### **LLM Integration & Tool Calling**
- **✅ RobustToolCaller Functional**: Handles native and prompt-based tool calling with retry mechanisms
- **✅ Centralized Agent Tools**: `Agent.cast_vote()` and `Agent.discuss()` fetch tools dynamically from `prompts.yaml`
- **✅ CRITICAL: Stance Generation Restored**: Full LLM-based stance generation implemented in `BaseAgent.generate_internal_stance()` using `PromptVariableGenerator`, `PromptLoader`, and `RobustToolCaller` with comprehensive error handling
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
- **🎉 Modular configuration system fully operational** with clean separation between `agent.yaml`, `simulation.yaml`, `output.yaml`, and `testing_groups.yaml`
- **✅ Prompts properly isolated** in `conclave/prompting/` module with `prompts.yaml` co-located with processing logic

**📝 Note on Configuration Refactoring:** This task required extensive debugging to resolve import path issues and Python bytecode cache conflicts when renaming `conclave.config` to `conclave.prompting`. All import statements, file paths, and cached bytecode were systematically updated to ensure full compatibility.

---

## 🔄 REMAINING REFACTORING TASKS

## VI. Enhanced Agent and Environment Logic

**Goal:** Address LLM output quality issues, improve discussion workflows, and implement sophisticated agent behavior patterns.

### **A. LLM Output Quality Improvements**

**Current Issues:**
- Short and incomplete LLM outputs due to over-querying
- Agents posing questions in non-interactive discussion format

**Solutions:**
- **Rate Limiting**: Configurable delays between LLM calls
- **Quality Monitoring**: Retry with enhanced context if responses are too short
- **Context Optimization**: Reduce unnecessary prompt content
- **Temperature Control**: Configurable settings for response quality

### **B. Discussion Prompt Redesign**

**Strategy Changes:**
- **Rhetorical Questions Only**: Eliminate direct questions expecting responses
- **Statement-Based Communication**: Emphasize declarative statements
- **Memory-Based References**: Reference previous round concerns
- **Addressable Concerns**: Focus on addressing rather than asking

### **C. Advanced Agent Workflows**

**Parallel Stance Generation:**
- Orchestrated by `ConclaveEnv` before first round and after discussions
- Uses `ThreadPoolExecutor` with `tqdm` progress visualization
- Context varies by round (initial vs. revised stances)

**Post-Round Embedding Updates:**
- Re-compute embeddings after each voting phase
- Voting-based color coding for visualization
- Dynamic alliance tracking through color evolution

## VII. Discussion Reflection System

**Goal:** Implement complete discussion workflow: Discussion → Analysis → Reflection → Stance Update.

### **A. Core Components**

**Critical Variables:**
- `group_summary` (JSON): Generated using `discussion_analyzer`
- `agent_last_utterance` (str): Agent's most recent speech
- `reflection_digest` (str): Generated using `discussion_reflection`

### **B. Workflow Integration**

**Enhanced Discussion Round Process:**
1. **Discussion Phase**: Small group discussions
2. **Analysis Phase**: Generate `group_summary`
3. **Reflection Phase**: Generate `reflection_digest`
4. **Stance Update Phase**: Update stances with reflection
5. **Voting Phase**: Cast votes with enhanced context

### **C. Implementation Requirements**

**Required Methods:**
- `ConclaveEnv.analyze_discussion_round()`: Generate JSON group summaries
- `BaseAgent.reflect_on_discussion()`: Generate reflection digests
- Enhanced stance generation incorporating reflection context
- Helper methods for data extraction and validation

## VIII. Embedding and Visualization System Upgrades

**Goal:** Improve robustness and implement advanced dimensionality reduction techniques.

### **A. Embedding System Enhancements**

**Improvements:**
- **Dependency Injection**: Pass `EmbeddingClient` instances instead of global singleton
- **Persistent Caching**: Disk-based embedding cache using `joblib.Memory`
- **Batch Processing**: `get_embeddings_batch()` method for improved performance
- **Configuration-Driven**: All settings in `config.yaml`

### **B. Advanced Visualization Implementation**

**Dimensionality Reduction Techniques:**
- **Primary Method**: UMAP for optimal balance of local/global structure
- **Comparative Analysis**: PCA baseline and t-SNE investigation
- **Temporal Consistency**: Procrustes alignment for trajectory tracking

**Enhanced Visualization Features:**
- **Multi-Method Dashboard**: Side-by-side comparison of techniques
- **Real-Time Updates**: UMAP embedding updates with temporal alignment
- **Validation Framework**: Quantitative metrics and qualitative review

## IX. Enhanced Logging Implementation

**Goal:** Implement structured, configurable logging system for improved debuggability and monitoring.

### **A. Core Logging Structure**

**Logger Categories:**
- `conclave.agents`: Agent actions, decisions, stance changes
- `conclave.llm`: Prompts, responses, tool calls, errors
- `conclave.system`: Simulation lifecycle, configuration loading
- `conclave.performance`: Timing measurements and metrics

**Implementation Components:**
1. **Pydantic Models**: Define `LoggingConfig`, `FormatterConfig`, `HandlerConfig`, `LoggerConfig` in `config_models.py`
2. **ConfigManager Enhancement**: Implement `initialize_logging(logs_dir: Path)` method
3. **Module Integration**: Update all modules to use appropriate loggers
4. **Performance Monitoring**: Add timing decorators for key operations

### **B. Detailed Logging Structure**

**Directory Structure:**
```
simulation_output/
├── logs/
│   ├── detailed_discussions.log    # Every utterance and voting result
│   ├── llm_io.log                  # All LLM interactions
│   └── system.log                  # System events and errors
├── reports/
│   ├── discussion_summaries/       # Per-discussion group summaries
│   ├── round_summaries/           # Complete election round summaries
│   └── conclave_summary.md        # Complete narrative summary
└── results/
    ├── voting_history.json        # Complete voting record
    └── stance_evolution.json      # Stance changes over time
```

**Log Types and Content:**
1. **Detailed Discussion Logs**
   - Every agent utterance
   - Voting results
   - Stance changes
   - No truncation of LLM I/O

2. **Discussion Summaries**
   - Generated using `discussion_analyzer`
   - Key points and themes
   - Group dynamics
   - JSON format for analysis

3. **Round Summaries**
   - Complete election round overview
   - Voting patterns
   - Stance evolution
   - Key decision points

4. **Conclave Summary**
   - Complete narrative of the conclave
   - Major turning points
   - Final outcomes
   - Markdown format for readability

### **C. Implementation Tasks**

1. **Update Configuration Models**
   - Define logging configuration models in `config_models.py`
   - Ensure validation matches `config.yaml` structure

2. **Enhance ConfigManager**
   - Implement `initialize_logging` method
   - Handle dynamic log paths
   - Configure formatters and handlers

3. **Module Integration**
   - Update agent classes to use `conclave.agents` logger
   - Update LLM modules to use `conclave.llm` logger
   - Update environment to use `conclave.system` logger
   - Add performance logging to key operations

4. **Summary Generation**
   - Implement `discussion_analyzer` for group summaries
   - Create round summary generator
   - Develop conclave narrative generator
   - Add JSON/Markdown export capabilities

### **D. Testing Requirements**

**After Each Implementation Phase:**
1. **Smoke Tests**
   - Run `tests/test_refactoring_smoke.py`
   - Verify all 17 glossary variables are correctly generated
   - Check log file creation and content
   - Validate JSON/Markdown output formats

2. **Variable Validation**
   - Test all critical variables from glossary:
     - `agent_name`, `biography`, `persona_internal`
     - `profile_public`, `group_profiles`, `participant_relation_list`
     - `discussion_round`, `group_transcript`, `group_summary`
     - `agent_last_utterance`, `reflection_digest`, `compact_scoreboard`
     - `visible_candidates`, `threshold`, `voting_round`
     - `role_tag`, `stance_digest`

3. **Logging Verification**
   - Check log file permissions and rotation
   - Verify log content formatting
   - Test log level filtering
   - Validate performance metrics

4. **Summary Generation Tests**
   - Verify discussion summary JSON format
   - Check round summary completeness
   - Validate conclave narrative structure
   - Test export functionality

## X. Parameter Sweep Infrastructure

**Goal:** Implement systematic parameter testing and result analysis framework.

### **A. Parameter Sweep Architecture**

**Components:**
- **Sweep Configuration**: Define parameter ranges in `config/parameter_sweeps/`
- **Sweep Runner**: Execute simulations with different parameters
- **Result Aggregator**: Collect and organize results
- **Analysis Engine**: Compare outcomes across variations

### **B. Implementation Features**

**Sweep Configuration:**
```
config/parameter_sweeps/
├── grid_search.yaml      # Systematic grid search
├── random_search.yaml    # Random parameter sampling
└── custom_sweeps/        # Custom combinations
```

**Sweep Runner Capabilities:**
- Parallel execution of parameter combinations
- Progress tracking and visualization
- Automatic result collection
- Error handling and recovery

**Analysis Tools:**
- Statistical comparison of outcomes
- Visualization of parameter effects
- Automated report generation
- Parameter importance analysis

### **C. Implementation Tasks**

1. **Sweep Configuration System**
   - Create Pydantic models for sweep configurations
   - Define parameter ranges and combinations
   - Implement validation and defaults

2. **Sweep Runner**
   - Implement parallel execution
   - Add progress tracking
   - Handle result collection
   - Implement error recovery

3. **Result Analysis**
   - Develop statistical comparison tools
   - Create visualization framework
   - Implement automated reporting
   - Add parameter importance analysis

### **D. Testing Requirements**

**After Each Implementation Phase:**
1. **Smoke Tests**
   - Run `tests/test_refactoring_smoke.py`
   - Verify parameter sweep configuration loading
   - Check parallel execution functionality
   - Validate result collection

2. **Parameter Validation**
   - Test parameter range definitions
   - Verify combination generation
   - Check parameter validation
   - Test default values

3. **Result Analysis Tests**
   - Verify statistical comparison tools
   - Check visualization generation
   - Validate automated reporting
   - Test parameter importance analysis

4. **Performance Tests**
   - Measure parallel execution efficiency
   - Check memory usage during sweeps
   - Validate error recovery
   - Test progress tracking

## XI. Implementation Timeline and Priorities

[Previous timeline sections remain unchanged]

### **Testing Strategy**

**For Each Phase:**
1. **Unit Tests**
   - Test individual components
   - Verify variable generation
   - Check logging functionality
   - Validate parameter handling

2. **Integration Tests**
   - Test component interactions
   - Verify data flow
   - Check error handling
   - Validate logging integration

3. **Smoke Tests**
   - Run full simulation
   - Verify all variables
   - Check output generation
   - Validate performance

4. **Regression Tests**
   - Verify existing functionality
   - Check backward compatibility
   - Validate error recovery
   - Test fallback mechanisms