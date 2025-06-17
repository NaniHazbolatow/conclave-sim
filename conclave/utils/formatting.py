"""
Common formatting utilities for the Conclave simulation.

This module provides standardized formatting functions for displaying
agent information, discussion transcripts, and other simulation data.
"""

from typing import List, Dict, Any, Optional


def format_candidate_info(
    agent: Any, 
    include_cardinal_id: bool = False, 
    include_votes: bool = False,
    include_stance: bool = False
) -> str:
    """
    Format agent information for display as a candidate.
    
    Args:
        agent: The agent object to format
        include_cardinal_id: Whether to include the cardinal ID
        include_votes: Whether to include vote count
        include_stance: Whether to include current stance
        
    Returns:
        Formatted string representation of the candidate
    """
    if not agent:
        return "Unknown Agent"
    
    # Start with basic name and ID
    parts = [f"{agent.name} (ID: {agent.id})"]
    
    # Add cardinal ID if requested and available
    if include_cardinal_id and hasattr(agent, 'cardinal_id'):
        parts.append(f"Cardinal ID: {agent.cardinal_id}")
    
    # Add vote count if requested and available
    if include_votes and hasattr(agent, 'votes_received'):
        parts.append(f"Votes: {agent.votes_received}")
    
    # Add stance if requested and available
    if include_stance and hasattr(agent, 'current_stance'):
        stance_preview = agent.current_stance[:50] + "..." if len(agent.current_stance) > 50 else agent.current_stance
        parts.append(f"Stance: {stance_preview}")
    
    return " | ".join(parts)


def format_agent_list(
    agents: List[Any], 
    format_style: str = "simple",
    include_indices: bool = False,
    separator: str = "\n"
) -> str:
    """
    Format a list of agents for display.
    
    Args:
        agents: List of agent objects
        format_style: Style of formatting ('simple', 'detailed', 'candidate')
        include_indices: Whether to include numeric indices
        separator: String to separate agent entries
        
    Returns:
        Formatted string representation of the agent list
    """
    if not agents:
        return "No agents available"
    
    formatted_agents = []
    
    for i, agent in enumerate(agents):
        if format_style == "simple":
            entry = f"{agent.name} (ID: {agent.id})"
        elif format_style == "detailed":
            entry = format_candidate_info(agent, include_cardinal_id=True, include_votes=True)
        elif format_style == "candidate":
            entry = format_candidate_info(agent, include_cardinal_id=True, include_stance=True)
        else:
            entry = str(agent)
        
        if include_indices:
            entry = f"{i+1}. {entry}"
        
        formatted_agents.append(entry)
    
    return separator.join(formatted_agents)


def format_discussion_transcript(
    transcript: List[Dict[str, Any]], 
    include_timestamps: bool = False,
    include_agent_ids: bool = False,
    max_message_length: Optional[int] = None
) -> str:
    """
    Format a discussion transcript for display.
    
    Args:
        transcript: List of message dictionaries
        include_timestamps: Whether to include timestamps
        include_agent_ids: Whether to include agent IDs
        max_message_length: Maximum length for messages (truncates if longer)
        
    Returns:
        Formatted transcript string
    """
    if not transcript:
        return "No discussion messages"
    
    formatted_messages = []
    
    for msg in transcript:
        parts = []
        
        # Add timestamp if requested
        if include_timestamps and 'timestamp' in msg:
            parts.append(f"[{msg['timestamp']}]")
        
        # Add speaker info
        speaker_info = msg.get('speaker_name', 'Unknown')
        if include_agent_ids and 'speaker_id' in msg:
            speaker_info += f" (ID: {msg['speaker_id']})"
        parts.append(f"{speaker_info}:")
        
        # Add message content
        message = msg.get('message', '')
        if max_message_length and len(message) > max_message_length:
            message = message[:max_message_length] + "..."
        parts.append(message)
        
        formatted_messages.append(" ".join(parts))
    
    return "\n".join(formatted_messages)


def format_voting_results(
    results: Dict[str, Any],
    include_percentages: bool = True,
    sort_by_votes: bool = True
) -> str:
    """
    Format voting results for display.
    
    Args:
        results: Dictionary containing voting results
        include_percentages: Whether to show vote percentages
        sort_by_votes: Whether to sort candidates by vote count
        
    Returns:
        Formatted voting results string
    """
    if not results or 'votes' not in results:
        return "No voting results available"
    
    votes = results['votes']
    total_votes = sum(votes.values()) if votes else 0
    
    if total_votes == 0:
        return "No votes cast"
    
    # Convert to list of tuples for sorting
    vote_items = list(votes.items())
    
    if sort_by_votes:
        vote_items.sort(key=lambda x: x[1], reverse=True)
    
    formatted_results = []
    
    for candidate, vote_count in vote_items:
        line = f"{candidate}: {vote_count} vote{'s' if vote_count != 1 else ''}"
        
        if include_percentages:
            percentage = (vote_count / total_votes) * 100
            line += f" ({percentage:.1f}%)"
        
        formatted_results.append(line)
    
    # Add total
    formatted_results.append(f"\nTotal votes cast: {total_votes}")
    
    # Add winner if available
    if 'winner' in results and results['winner']:
        formatted_results.append(f"Winner: {results['winner']}")
    
    return "\n".join(formatted_results)


def format_stance_history(
    stance_history: List[str],
    max_entries: Optional[int] = None,
    show_evolution: bool = True
) -> str:
    """
    Format an agent's stance history for display.
    
    Args:
        stance_history: List of stance strings
        max_entries: Maximum number of entries to show (None for all)
        show_evolution: Whether to show how stance evolved
        
    Returns:
        Formatted stance history string
    """
    if not stance_history:
        return "No stance history available"
    
    # Limit entries if requested
    if max_entries and len(stance_history) > max_entries:
        stance_history = stance_history[-max_entries:]
    
    if not show_evolution:
        return f"Current stance: {stance_history[-1]}"
    
    formatted_history = []
    
    for i, stance in enumerate(stance_history, 1):
        prefix = f"Stance {i}:" if len(stance_history) > 1 else "Stance:"
        formatted_history.append(f"{prefix} {stance}")
    
    return "\n".join(formatted_history)


def format_summary_statistics(
    stats: Dict[str, Any],
    title: str = "Statistics"
) -> str:
    """
    Format summary statistics for display.
    
    Args:
        stats: Dictionary of statistics
        title: Title for the statistics section
        
    Returns:
        Formatted statistics string
    """
    if not stats:
        return f"{title}: No statistics available"
    
    lines = [f"=== {title} ==="]
    
    for key, value in stats.items():
        # Format key (convert from snake_case to readable)
        readable_key = key.replace('_', ' ').title()
        
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = str(value)
        elif isinstance(value, (list, tuple)):
            formatted_value = f"{len(value)} items"
        else:
            formatted_value = str(value)
        
        lines.append(f"{readable_key}: {formatted_value}")
    
    return "\n".join(lines)
