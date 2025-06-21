"""
Refactored configuration models using Pydantic.
Only includes configuration parameters that are actually used in the codebase.
Split into focused modules: agent, simulation, output, and testing_groups.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Union, Any
import yaml
from pathlib import Path


# ===================================================================
# AGENT CONFIGURATION MODELS (LLM + Embeddings)
# ===================================================================

class ToolCallingConfig(BaseModel):
    """Tool calling configuration for robust_tools module."""
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay in seconds between retries")
    retry_backoff_sec: int = Field(default=2, description="Exponential backoff multiplier")
    enable_fallback: bool = Field(default=True, description="Enable fallback to prompt-based tool calling")


class LocalLLMConfig(BaseModel):
    """Local LLM configuration (HuggingFace models)."""
    model_name: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    use_quantization: bool = Field(default=False)
    device: str = Field(default="auto")
    cache_dir: str = Field(default="~/.cache/huggingface")
    do_sample: bool = Field(default=True)
    top_p: float = Field(default=0.9)
    repetition_penalty: float = Field(default=1.1)


class RemoteLLMConfig(BaseModel):
    """Remote LLM configuration (API models)."""
    model_name: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    api_key_env: str = Field(default="OPENROUTER_API_KEY")
    top_p: float = Field(default=0.9)
    frequency_penalty: float = Field(default=0.1)
    presence_penalty: float = Field(default=0.1)


class LLMConfig(BaseModel):
    """Main LLM configuration."""
    backend: str = Field(default="remote", description="Backend type: 'local' or 'remote'")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_tokens: int = Field(default=500, description="Maximum tokens per response")
    tool_calling: ToolCallingConfig = Field(default_factory=ToolCallingConfig)
    local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    remote: RemoteLLMConfig = Field(default_factory=RemoteLLMConfig)


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration - simplified to only used parameters."""
    model_name: str = Field(default="intfloat/e5-large-v2")
    device: str = Field(default="auto")
    batch_size: int = Field(default=32)
    enable_caching: bool = Field(default=True)
    cache_dir: str = Field(default="~/.cache/embeddings")
    similarity_threshold: float = Field(default=0.7)
    max_search_results: int = Field(default=10)


class AgentConfig(BaseModel):
    """Agent configuration (LLM + Embeddings)."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    allow_reflection_without_summary: bool = Field(default=True, description="Allow agents to reflect even if no group analysis summary is available") # Added field


# ===================================================================
# SIMULATION CONFIGURATION MODELS
# ===================================================================

class DiscussionLengthConfig(BaseModel):
    """Discussion word count constraints."""
    min_words: int = Field(default=50, description="Minimum word count for discussions")
    max_words: int = Field(default=100, description="Maximum word count for discussions")


class VotingConfig(BaseModel):
    """Voting rules and thresholds."""
    require_reasoning: bool = Field(default=True, description="Cardinals must explain their vote")
    supermajority_threshold: float = Field(default=0.667, description="Threshold for electing Pope (2/3)")


class UtilityWeightsConfig(BaseModel):
    """Utility weights for network-based grouping."""
    connection: float = Field(default=1.0, description="Connection strength weight")
    ideology: float = Field(default=0.5, description="Ideological proximity weight")
    influence: float = Field(default=0.2, description="Influenceability weight")
    interaction: float = Field(default=0.0, description="Interaction term weight")


class GroupingConfig(BaseModel):
    """Network-based grouping configuration."""
    rationality: float = Field(default=1.0, description="Balance between utility-based (1.0) and random (0.0) selection")
    penalty_weight: float = Field(default=0.1, description="Weight for avoiding repeated pairings")
    utility_weights: UtilityWeightsConfig = Field(default_factory=UtilityWeightsConfig)


class SimulationConfig(BaseModel):
    """Core simulation parameters."""
    max_rounds: int = Field(default=5, description="Maximum simulation rounds before stopping (Range: 1-50)")
    enable_parallel: bool = Field(default=True, description="Enable parallel processing for agent actions")
    rationality: float = Field(default=0.8, description="Agent rationality (Range: 0.0=random to 1.0=fully rational)")
    temperature: float = Field(default=0.7, description="LLM temperature (Range: 0.0=deterministic to 2.0=creative)")
    discussion_group_size: int = Field(default=3, description="Number of agents per discussion group")
    discussion_length: DiscussionLengthConfig = Field(default_factory=DiscussionLengthConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig, description="Network-based grouping configuration")
    
    # Legacy aliases for backward compatibility
    @property
    def max_election_rounds(self) -> int:
        return self.max_rounds
    
    @property 
    def enable_parallel_processing(self) -> bool:
        return self.enable_parallel


# ===================================================================
# OUTPUT CONFIGURATION MODELS
# ===================================================================

class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_level: str = Field(default="INFO")
    performance_logging: bool = Field(default=True)
    disable_tqdm: bool = Field(default=False, description="Disable tqdm progress bars globally")


class ResultsConfig(BaseModel):
    """Results saving configuration."""
    save_individual_votes: bool = Field(default=True)
    save_discussion_transcripts: bool = Field(default=True)
    save_agent_states: bool = Field(default=False)


class TSNEConfig(BaseModel):
    """t-SNE configuration."""
    perplexity: float = Field(default=30.0, description="t-SNE perplexity")
    n_iter: int = Field(default=1000, description="Number of iterations for t-SNE")
    random_state: Optional[int] = Field(default=None, description="Random state for t-SNE")


class UMAPConfig(BaseModel):
    """UMAP configuration."""
    n_neighbors: int = Field(default=15, description="Number of neighbors for UMAP")
    min_dist: float = Field(default=0.1, description="Minimum distance between points in UMAP")
    metric: str = Field(default='euclidean', description="Distance metric for UMAP")
    random_state: Optional[int] = Field(default=None, description="Random state for UMAP")


class PCAConfig(BaseModel):
    """PCA configuration."""
    random_state: Optional[int] = Field(default=None, description="Random state for PCA")


class PlotConfig(BaseModel):
    """Plotting configuration for visualizations."""
    target_type: str = Field(default='stance', description="Target type for plotting (stance or ideology)")
    show_names: bool = Field(default=True, description="Show agent/candidate names on plots")
    show_initial_stances: bool = Field(default=True, description="Show initial stances in plots")
    show_final_stances: bool = Field(default=True, description="Show final stances in plots")
    show_paths: bool = Field(default=True, description="Show paths between initial and final stances")
    show_group_colors: bool = Field(default=True, description="Color by group in plots")
    font_size: int = Field(default=8, description="Font size for plot labels")
    figure_size: List[float] = Field(default_factory=lambda: [10.0, 8.0], description="Size of the plot figure")
    colormap: str = Field(default='viridis', description="Colormap for plots")
    path_alpha: float = Field(default=0.5, description="Transparency of the path lines")
    path_linewidth: float = Field(default=1.0, description="Line width of the path lines")
    point_size: int = Field(default=50, description="Size of the points in the plot")
    initial_stance_marker: str = Field(default='o', description="Marker for initial stances")
    final_stance_marker: str = Field(default='X', description="Marker for final stances")
    intermediate_stance_marker: str = Field(default='.', description="Marker for intermediate stances")
    candidate_marker: str = Field(default='*', description="Marker for candidates")
    non_candidate_marker: str = Field(default='.', description="Marker for non-candidates")
    candidate_color: str = Field(default='red', description="Color for candidate points")
    non_candidate_color: str = Field(default='blue', description="Color for non-candidate points")
    group_palette: str = Field(default='Set2', description="Color palette for groups")
    stance_history_limit: Optional[int] = Field(default=None, description="Limit for stance history points")


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    auto_generate: bool = Field(default=True)
    include_network_graphs: bool = Field(default=True)
    reduction_method: str = Field(default="pca", description="Dimensionality reduction method for embeddings (pca or tsne)")
    output_format: str = Field(default="png", description="Output format for plots (png, svg, pdf)")
    dpi: int = Field(default=300, description="DPI for saved plots")
    tsne: TSNEConfig = Field(default_factory=TSNEConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    pca: PCAConfig = Field(default_factory=PCAConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)


class OutputConfig(BaseModel):
    """Output configuration (logging, results, visualization)."""
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    # Add a direct tqdm attribute for easier access, mirroring the structure of other configs
    @property
    def tqdm(self) -> LoggingConfig: # Reuse LoggingConfig for simplicity or create a new TqdmConfig
        return self.logging


# ===================================================================
# PREDEFINED GROUPS CONFIGURATION MODELS
# ===================================================================

class IdeologyBalance(BaseModel):
    """Ideology balance for a predefined group."""
    progressive: int = 0
    conservative: int = 0
    moderate: int = 0

class OverrideSettings(BaseModel):
    """Override settings for a predefined group."""
    num_cardinals: int
    discussion_group_size: int
    max_election_rounds: int
    supermajority_threshold: float

class PredefinedGroup(BaseModel):
    """Settings for a predefined group."""
    total_cardinals: int
    cardinal_ids: List[int]
    candidate_ids: List[int]
    elector_ids: List[int]
    description: str
    ideology_balance: IdeologyBalance
    override_settings: Optional[OverrideSettings] = None

class PredefinedGroupsConfig(BaseModel):
    """Configuration for all predefined groups."""
    small: PredefinedGroup
    medium: PredefinedGroup
    large: PredefinedGroup
    xlarge: PredefinedGroup
    full: PredefinedGroup


# ===================================================================
# SIMULATION SETTINGS CONFIGURATION MODELS
# ===================================================================

class SimulationSettings(BaseModel):
    """Settings to override default simulation parameters."""
    group: str
    group_size: int
    rationality: float
    temperature: float


# ===================================================================
# GROUPS CONFIGURATION (Optional)
# ===================================================================

class GroupsConfig(BaseModel):
    """Optional groups configuration for using predefined groups."""
    active: str = Field(default="medium", description="Active predefined group: small, medium, large, xlarge, full")


# ===================================================================
# ROOT CONFIGURATION MODEL
# ===================================================================

class RefactoredConfig(BaseModel):
    """Root configuration model with consolidated settings."""
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    models: AgentConfig = Field(default_factory=AgentConfig, description="LLM and embeddings configuration")
    output: OutputConfig = Field(default_factory=OutputConfig)
    agents: Optional[Dict[str, Any]] = Field(default={"allow_reflection_without_summary": True}, description="Agent behavior settings")
    groups: Optional[GroupsConfig] = Field(default=None, description="Optional predefined groups configuration")
    
    # Legacy support for old structure (deprecated)
    agent: Optional[AgentConfig] = Field(default=None, description="DEPRECATED: Use 'models' instead")
    predefined_groups: Optional[PredefinedGroupsConfig] = Field(default=None, description="DEPRECATED: Use external groups.yaml")
    simulation_settings: Optional[SimulationSettings] = Field(default=None, description="DEPRECATED: Settings moved to simulation")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True
        
    @model_validator(mode='after')
    def handle_legacy_fields(self) -> 'RefactoredConfig':
        """Handle legacy field mapping after model initialization."""
        # Map legacy 'agent' to 'models' if present
        if self.agent is not None:
            self.models = self.agent
            
        # Apply simulation_settings overrides if present  
        if self.simulation_settings is not None:
            if hasattr(self.simulation_settings, 'rationality'):
                self.simulation.rationality = self.simulation_settings.rationality
            if hasattr(self.simulation_settings, 'temperature'):
                self.simulation.temperature = self.simulation_settings.temperature
        
        return self

    def to_yaml(self, include_cli_overrides_note: bool = False) -> str:
        """Convert config to YAML string for saving to file.
        
        Args:
            include_cli_overrides_note: If True, add comment about CLI overrides
            
        Returns:
            YAML string representation of the configuration
        """
        # If we have CLI overrides, we want to preserve the original format
        # but with the overridden values and groups section removed.
        try:
            # Read the original config.yaml to preserve structure and comments
            from pathlib import Path
            original_config_path = Path(__file__).parent.parent.parent / "config.yaml"
            if original_config_path.exists():
                with open(original_config_path, 'r') as f:
                    original_content = f.read()
                
                # Remove the groups section entirely and replace with CLI override if present
                import re
                # Remove groups section (including comments and the section itself)
                # Use a more comprehensive regex to catch all variations
                content = re.sub(
                    r'# Active predefined group[^#]*?groups:\s*\n\s*active:[^\n]*\n\n',
                    '',
                    original_content,
                    flags=re.DOTALL
                )
                
                # If we have a groups config with CLI overrides, add it back
                if self.groups and self.groups.active:
                    # Add the groups section at the beginning of the config content
                    groups_section = f"# Active predefined group (CLI override applied)\ngroups:\n  active: {self.groups.active}\n\n"
                    # Find where to insert - after the main header comment
                    insert_pos = content.find("simulation:")
                    if insert_pos > 0:
                        content = content[:insert_pos] + groups_section + content[insert_pos:]
                
                # Update specific values while preserving comments and structure
                # Update simulation values if they differ from defaults
                if hasattr(self.simulation, 'max_rounds'):
                    content = re.sub(
                        r'(\s+max_rounds:\s+)\d+', 
                        rf'\g<1>{self.simulation.max_rounds}',
                        content
                    )
                
                if hasattr(self.simulation, 'rationality'):
                    content = re.sub(
                        r'(\s+rationality:\s+)[\d.]+', 
                        rf'\g<1>{self.simulation.rationality}',
                        content
                    )
                
                if hasattr(self.simulation, 'temperature'):
                    content = re.sub(
                        r'(\s+temperature:\s+)[\d.]+', 
                        rf'\g<1>{self.simulation.temperature}',
                        content
                    )
                
                if hasattr(self.simulation, 'enable_parallel'):
                    content = re.sub(
                        r'(\s+enable_parallel:\s+)(true|false)', 
                        rf'\g<1>{str(self.simulation.enable_parallel).lower()}',
                        content
                    )
                
                # Add header if CLI overrides
                if include_cli_overrides_note:
                    header = "# Configuration used for this simulation run\n"
                    header += "# This file includes any CLI overrides that were applied\n"
                    header += f"# Generated on: {self._get_timestamp()}\n"
                    header += "# Groups configuration is loaded separately from groups.yaml\n\n"
                    
                    # Replace the first line with our header
                    lines = content.split('\n')
                    if lines and lines[0].startswith('#'):
                        lines[0] = header.rstrip()
                        content = '\n'.join(lines)
                    else:
                        content = header + content
                
                return content
        
        except Exception as e:
            # Fallback to standard YAML generation if file reading fails
            print(f"Warning: Could not preserve original format: {e}")
        
        # Fallback: Convert to dict and generate standard YAML
        config_dict = self.model_dump(
            exclude_none=True,
            exclude={
                'agent',  # Deprecated field
                'predefined_groups',  # External file
                'simulation_settings',  # Deprecated field
                'groups'  # External file - groups.yaml should remain separate
            }
        )
        
        # Create YAML string with nice formatting
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
            width=100
        )
        
        # Add header comment
        header = "# Configuration used for this simulation run\n"
        if include_cli_overrides_note:
            header += "# This file includes any CLI overrides that were applied\n"
            header += f"# Generated on: {self._get_timestamp()}\n"
            header += "# Groups configuration is loaded separately from groups.yaml\n"
        header += "\n"
        
        return header + yaml_str
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for config generation."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def save_to_file(self, file_path: Union[str, Path], include_cli_overrides_note: bool = False):
        """Save configuration to YAML file.
        
        Args:
            file_path: Path to save the configuration file
            include_cli_overrides_note: If True, add comment about CLI overrides
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml(include_cli_overrides_note=include_cli_overrides_note))
