# plotting_theme.py
# A centralized file to define a consistent plotting style for the ConclaveSim report.

import matplotlib.pyplot as plt
import seaborn as sns

print("Applying custom plotting theme...")

# 1. Define the style parameters for fonts and sizes
plot_style = {
    # Using a serif font to match academic paper styles
    "font.family": "serif", 
    # You can specify a font like "Times New Roman", "Georgia", or "Computer Modern Roman"
    # Ensure the font is installed on your system.
    "font.serif": "Georgia", 
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "figure.facecolor": "white", # Ensure figure background is white
    "axes.facecolor": "white"    # Ensure plot background is white
}

# 2. Apply the styles using seaborn's set_theme function
sns.set_theme(
    context="paper", 
    style="darkgrid", 
    rc=plot_style
)

# 3. Define and set the custom "papal" color palette
# Colors inspired by Vatican flag (gold, white/silver) and cardinal vestments (red)
papal_colors = [
    "#FFD700", # Gold
    "#C41E3A", # Cardinal Red
    "#800000", # Maroon
    "#A9A9A9", # Dark Gray (for silver/stone)
    "#4169E1", # Royal Blue (for contrast)
    "#555555"  # A darker gray for additional categories if needed
]
color_palette = sns.color_palette(papal_colors)
sns.set_palette(color_palette)

print("Custom plotting theme applied successfully.")

