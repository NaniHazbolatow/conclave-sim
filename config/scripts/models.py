"""
Refactored configuration models using Pydantic.
Only includes configuration parameters that are actually used in the codebase.
Split into focused modules: agent, simulation, output, and testing_groups.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union


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


class SimulationConfig(BaseModel):
    """Core simulation parameters."""
    num_cardinals: int = Field(default=5, description="Number of cardinals in simulation")
    discussion_group_size: int = Field(default=3, description="Number of agents per discussion group")
    max_election_rounds: int = Field(default=5, description="Maximum election rounds before stopping")
    discussion_length: DiscussionLengthConfig = Field(default_factory=DiscussionLengthConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)


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
# TESTING GROUPS CONFIGURATION MODELS
# ===================================================================

class TestingGroupSettings(BaseModel):
    """Override settings for a specific testing group."""
    num_cardinals: int
    discussion_group_size: int
    max_election_rounds: int
    supermajority_threshold: float


class TestingGroup(BaseModel):
    """Individual testing group configuration."""
    total_cardinals: int
    cardinal_ids: List[int]
    candidate_ids: List[int]
    description: str
    override_settings: TestingGroupSettings


class TestingGroupsConfig(BaseModel):
    """Testing groups configuration."""
    enabled: bool = Field(default=True)
    active_group: str = Field(default="medium")
    small: TestingGroup
    medium: TestingGroup
    large: TestingGroup


# ===================================================================
# ROOT CONFIGURATION MODEL
# ===================================================================

class RefactoredConfig(BaseModel):
    """Root configuration model with focused component configurations."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    testing_groups: Optional[TestingGroupsConfig] = Field(default=None, description="Optional testing groups configuration")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent extra fields to keep config clean
        validate_assignment = True
