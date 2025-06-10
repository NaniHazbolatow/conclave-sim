#!/usr/bin/env python3
"""
Cardinal visualization module for the conclave simulation.

This module provides comprehensive visualization tools for displaying cardinal positions,
ideological clusters, and embedding spaces using PCA and other dimensionality reduction techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CardinalVisualizer:
    """
    Visualizes cardinal positions and ideological clusters in 2D space.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.reduction_method = self.viz_config.get('reduction_method', 'pca')
        self.plot_config = self.viz_config.get('plot', {})
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def reduce_dimensions(self, embeddings: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Reduce high-dimensional embeddings to 2D coordinates.
        
        Args:
            embeddings: High-dimensional embedding vectors
            method: Reduction method ('pca', 'tsne', 'umap'). Uses config default if None.
            
        Returns:
            2D coordinates array
        """
        if method is None:
            method = self.reduction_method
            
        n_samples = len(embeddings)
        
        if method == 'pca':
            pca_config = self.viz_config.get('pca', {})
            reducer = PCA(
                n_components=2,
                random_state=pca_config.get('random_state', 42)
            )
            
        elif method == 'tsne':
            tsne_config = self.viz_config.get('tsne', {})
            reducer = TSNE(
                n_components=2,
                random_state=tsne_config.get('random_state', 42),
                perplexity=min(tsne_config.get('perplexity', 5), n_samples - 1),
                n_iter=tsne_config.get('n_iter', 1000)
            )
            
        elif method == 'umap':
            umap_config = self.viz_config.get('umap', {})
            reducer = umap.UMAP(
                n_components=2,
                random_state=umap_config.get('random_state', 42),
                n_neighbors=min(umap_config.get('n_neighbors', 5), n_samples - 1),
                min_dist=umap_config.get('min_dist', 0.1)
            )
            
        else:
            raise ValueError(f"Unknown reduction method: {method}")
            
        logger.info(f"Applying {method.upper()} dimensionality reduction...")
        coords_2d = reducer.fit_transform(embeddings)
        
        return coords_2d
    
    def normalize_coordinates(self, coords_2d: np.ndarray) -> np.ndarray:
        """
        Normalize 2D coordinates to 0-100 grid scale.
        
        Args:
            coords_2d: Raw 2D coordinates
            
        Returns:
            Normalized coordinates in 0-100 range
        """
        grid_config = self.viz_config.get('grid', {})
        grid_size = grid_config.get('size', 100)
        jitter = grid_config.get('jitter', 2.0)
        
        # Normalize to 0-grid_size range
        x_coords = coords_2d[:, 0]
        y_coords = coords_2d[:, 1]
        
        x_norm = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())) * grid_size
        y_norm = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())) * grid_size
        
        # Add small jitter to avoid overlaps
        if jitter > 0:
            x_norm += np.random.normal(0, jitter, len(x_norm))
            y_norm += np.random.normal(0, jitter, len(y_norm))
            
            # Clamp to valid range
            x_norm = np.clip(x_norm, 0, grid_size)
            y_norm = np.clip(y_norm, 0, grid_size)
        
        return np.column_stack([x_norm, y_norm])
    
    def create_ideological_visualization(
        self, 
        embeddings: np.ndarray, 
        cardinal_names: List[str],
        cardinal_categories: Optional[Dict[str, str]] = None,
        title: str = "Cardinal Ideological Positioning"
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of cardinal ideological positions.
        
        Args:
            embeddings: High-dimensional embedding vectors
            cardinal_names: List of cardinal names
            cardinal_categories: Optional mapping of cardinal names to categories
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        # Reduce dimensions
        coords_2d = self.reduce_dimensions(embeddings)
        coords_norm = self.normalize_coordinates(coords_2d)
        
        # Set up plot configuration
        fig_size = self.plot_config.get('figure_size', [12, 8])
        point_size = self.plot_config.get('point_size', 120)
        font_size = self.plot_config.get('font_size', 10)
        alpha = self.plot_config.get('alpha', 0.8)
        colormap = self.plot_config.get('colormap', 'tab10')
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Color points by category if provided
        if cardinal_categories:
            categories = [cardinal_categories.get(name, 'Unknown') for name in cardinal_names]
            unique_categories = list(set(categories))
            color_map = plt.cm.get_cmap(colormap)
            colors = [color_map(i / len(unique_categories)) for i in range(len(unique_categories))]
            
            for i, category in enumerate(unique_categories):
                mask = [cat == category for cat in categories]
                x_cat = coords_norm[mask, 0]
                y_cat = coords_norm[mask, 1]
                names_cat = [name for name, cat in zip(cardinal_names, categories) if cat == category]
                
                scatter = ax.scatter(
                    x_cat, y_cat, 
                    c=[colors[i]] * len(x_cat),
                    s=point_size, 
                    alpha=alpha,
                    label=category,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add labels
                for j, name in enumerate(names_cat):
                    ax.annotate(
                        name, 
                        (x_cat[j], y_cat[j]), 
                        xytext=(5, 5),
                        textcoords='offset points', 
                        fontsize=font_size, 
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3)
                    )
            
            ax.legend(title="Cardinal Categories", loc='upper right')
            
        else:
            # Simple scatter plot without categories
            scatter = ax.scatter(
                coords_norm[:, 0], coords_norm[:, 1],
                c=range(len(cardinal_names)),
                cmap=colormap,
                s=point_size,
                alpha=alpha,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add labels
            for i, name in enumerate(cardinal_names):
                ax.annotate(
                    name, 
                    (coords_norm[i, 0], coords_norm[i, 1]), 
                    xytext=(5, 5),
                    textcoords='offset points', 
                    fontsize=font_size, 
                    alpha=0.9
                )
        
        # Customize plot
        ax.set_title(f'{title} ({self.reduction_method.upper()})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Ideological Dimension 1', fontsize=12)
        ax.set_ylabel('Ideological Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        
        # Add explained variance for PCA
        if self.reduction_method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            pca.fit(embeddings)
            variance_ratio = pca.explained_variance_ratio_
            ax.text(
                0.02, 0.98, 
                f'PC1: {variance_ratio[0]:.1%}\nPC2: {variance_ratio[1]:.1%}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        return fig
    
    def analyze_clusters(
        self, 
        embeddings: np.ndarray, 
        cardinal_names: List[str],
        cardinal_categories: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze ideological clusters and their separation.
        
        Args:
            embeddings: High-dimensional embedding vectors
            cardinal_names: List of cardinal names
            cardinal_categories: Mapping of cardinal names to categories
            
        Returns:
            Dictionary containing cluster analysis results
        """
        # Reduce dimensions
        coords_2d = self.reduce_dimensions(embeddings)
        coords_norm = self.normalize_coordinates(coords_2d)
        
        # Group by categories
        category_positions = {}
        for name, category in cardinal_categories.items():
            if name in cardinal_names:
                idx = cardinal_names.index(name)
                if category not in category_positions:
                    category_positions[category] = []
                category_positions[category].append(coords_norm[idx])
        
        # Calculate cluster centers
        cluster_centers = {}
        for category, positions in category_positions.items():
            if positions:
                positions_array = np.array(positions)
                center = np.mean(positions_array, axis=0)
                cluster_centers[category] = center
        
        # Calculate pairwise distances between cluster centers
        cluster_distances = {}
        categories = list(cluster_centers.keys())
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                center1 = cluster_centers[cat1]
                center2 = cluster_centers[cat2]
                distance = np.linalg.norm(center1 - center2)
                cluster_distances[f"{cat1} - {cat2}"] = distance
        
        # Analysis results
        analysis_config = self.viz_config.get('analysis', {})
        threshold = analysis_config.get('cluster_distance_threshold', 30)
        
        return {
            'cluster_centers': cluster_centers,
            'cluster_distances': cluster_distances,
            'good_separation': all(d > threshold for d in cluster_distances.values()),
            'coordinates': {name: coords_norm[i].tolist() for i, name in enumerate(cardinal_names)}
        }
    
    def create_comparison_plot(
        self, 
        embeddings: np.ndarray, 
        cardinal_names: List[str],
        cardinal_categories: Optional[Dict[str, str]] = None
    ) -> plt.Figure:
        """
        Create a comparison plot showing all three reduction methods.
        
        Args:
            embeddings: High-dimensional embedding vectors
            cardinal_names: List of cardinal names
            cardinal_categories: Optional mapping of cardinal names to categories
            
        Returns:
            Matplotlib figure object with subplots
        """
        methods = ['pca', 'tsne', 'umap']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, method in enumerate(methods):
            # Reduce dimensions with specific method
            coords_2d = self.reduce_dimensions(embeddings, method)
            coords_norm = self.normalize_coordinates(coords_2d)
            
            ax = axes[idx]
            
            # Color points by category if provided
            if cardinal_categories:
                categories = [cardinal_categories.get(name, 'Unknown') for name in cardinal_names]
                unique_categories = list(set(categories))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
                
                for i, category in enumerate(unique_categories):
                    mask = [cat == category for cat in categories]
                    x_cat = coords_norm[mask, 0]
                    y_cat = coords_norm[mask, 1]
                    
                    ax.scatter(
                        x_cat, y_cat,
                        c=[colors[i]] * len(x_cat),
                        s=100,
                        alpha=0.7,
                        label=category,
                        edgecolors='black',
                        linewidth=0.5
                    )
            else:
                ax.scatter(
                    coords_norm[:, 0], coords_norm[:, 1],
                    c=range(len(cardinal_names)),
                    cmap='tab10',
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Add labels
            for i, name in enumerate(cardinal_names):
                ax.annotate(
                    name, 
                    (coords_norm[i, 0], coords_norm[i, 1]), 
                    xytext=(3, 3),
                    textcoords='offset points', 
                    fontsize=8
                )
            
            ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            
            if idx == 0 and cardinal_categories:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Cardinal Positioning Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
