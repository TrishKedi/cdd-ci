"""Rich console-based visualization for code duplication results.

This module provides a Visualizer class that formats and displays
code duplication matches using Rich library components.
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text

logger = logging.getLogger(__name__)


class Visualizer:
    """Handles rich console visualization of duplicate code matches.
    
    This class provides methods to display code similarity results in a
    formatted, interactive way using Rich library components including
    syntax highlighting, clickable links, and structured tables.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the visualizer with display options.
        
        Args:
            **kwargs: Configuration options including:
                - reasoning: Whether to show LLM reasoning in output
                - verbose: Whether to show detailed code snippets
                - less: Whether to use less pager for output
        """
        self.reasoning: bool = kwargs.get("reasoning", False)
        self.verbose: bool = kwargs.get("verbose", False)
        self.less: bool = kwargs.get("less", False)
        self.console = Console()

    def _short(self, code: str, max_lines: int = 8) -> str:
        """Truncate code to specified number of lines for display.
        
        Args:
            code: The source code to truncate
            max_lines: Maximum number of lines to display
            
        Returns:
            Truncated code string with ellipsis if truncated
        """
        lines = code.splitlines()
        return "\n".join(lines[:max_lines] + (["…"] if len(lines) > max_lines else []))

    def _score_style(self, score: float) -> str:
        """Get Rich style for similarity score based on value.
        
        Args:
            score: Similarity score between 0.0 and 1.0
            
        Returns:
            Rich style string for coloring the score
        """
        if score >= 0.90:
            return "bold green"
        if score >= 0.80:
            return "yellow"
        return "red"

    def _link_text(self, path: Path, label: str, style: str, line: Optional[int] = None) -> Text:
        """Create clickable text for file paths in supporting terminals.
        
        Args:
            path: File path to link to
            label: Display text for the link
            style: Rich style to apply
            line: Optional line number to jump to
            
        Returns:
            Rich Text object with clickable file link
        """
        # Create styled text with VS Code-compatible file link
        text = Text(label, style=style)
        uri = f"vscode://file/{path.resolve().as_posix()}"
        if line is not None:
            uri += f":{line}"
        text.stylize(f"link {uri}")
        return text

    def display_matches(self, matches: Dict[str, Any]) -> None:
        """Display matches if any exist.
        
        Args:
            matches: Dictionary containing match data and metadata
        """
        if matches:
            if self.less:
                self._display_with_pager(matches)
            else:
                self.print_matches(matches)
    
    def _display_with_pager(self, matches: Dict[str, Any]) -> None:
        """Display matches using less pager to prevent truncation.
        
        Args:
            matches: Dictionary containing match data and metadata
        """
        try:
            # Capture console output to string
            with self.console.capture() as capture:
                self.print_matches(matches)
            
            # Get the captured output
            output = capture.get()
            
            # Use less pager to display output
            self._pipe_to_less(output)
            
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback to normal display if less is not available
            self.console.print("[yellow]Warning: 'less' command not found, displaying normally[/yellow]")
            self.print_matches(matches)
    
    def _pipe_to_less(self, content: str) -> None:
        """Pipe content to the less pager.
        
        Args:
            content: Text content to display in pager
        """
        try:
            # Start less process with useful options:
            # -R: interpret ANSI color escape sequences
            # -S: chop long lines (don't wrap)
            # -F: quit if entire file fits on first screen
            # -X: don't clear screen after quitting
            process = subprocess.Popen(
                ['less', '-R', '-S', '-F', '-X'],
                stdin=subprocess.PIPE,
                text=True
            )
            
            # Send content to less and wait for completion
            process.communicate(input=content)
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # If less fails, fallback to direct output
            sys.stdout.write(content)
            logger.error(f"Failed to pipe to less: {e}")

    def print_matches(self, matches: Dict[str, Any]) -> None:
        """Print formatted duplicate matches with Rich console output.
        
        Displays source file information and a table of similar code matches
        with similarity scores, file paths, and optionally code snippets.
        
        Args:
            matches: Dictionary containing:
                - path: Source file path
                - code: Source code snippet
                - start: Starting line number
                - matches: List of similar code blocks with scores
        """
        # Extract match data
        src_label = matches.get("path", "<?>")
        src_path = Path(src_label).resolve()
        rows = matches.get("matches") or []


        if rows:
            # Display source file header
            start = matches.get('start', 1)
            clickable_file_text = self._link_text(
                src_path,
                f"File: {src_label}:{start}",
                "bold cyan",
                start,
            )
            self.console.rule(clickable_file_text)

            # Display source code if verbose mode
            if self.verbose:
                src_code = matches.get("code") or ""
                if src_code:
                    syntax = Syntax(
                        self._short(src_code), 
                        "javascript", 
                        theme="monokai", 
                        word_wrap=True
                    )
                    self.console.print(Panel(syntax, title="Code", border_style="cyan"))

            # Create matches table
            table = self._create_matches_table()

            # Add sorted matches to table
            for match in sorted(rows, key=lambda x: x.get("score", 0), reverse=True):
                self._add_match_row(table, match)
                
            # Display the completed table
            self.console.print(Panel(table, title="Matches", border_style="magenta"))
    
    def _create_matches_table(self) -> Table:
        """Create and configure the matches display table.
        
        Returns:
            Configured Rich Table for displaying matches
        """
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Score", justify="right", width=6)
        table.add_column("File", overflow="fold")

        if self.verbose:
            table.add_column("Snippet", overflow="fold")

        if self.reasoning:
            table.add_column("Reasoning", overflow="fold")
            
        return table
    
    def _add_match_row(self, table: Table, match: Dict[str, Any]) -> None:
        """Add a single match row to the table.
        
        Args:
            table: Rich Table to add row to
            match: Match data dictionary
        """
        # Format similarity score
        score = float(match.get("score", 0))
        score_text = Text(f"{score:.2f}", style=self._score_style(score))

        # Create clickable file link
        tgt_path = Path(match.get("path", "")).resolve()
        tgt_label = match.get("path", "<?>")
        start_line = match.get("start")
        label_with_line = f"{tgt_label}:{start_line}" if start_line is not None else str(tgt_label)
        clickable = self._link_text(tgt_path, label_with_line, "white", start_line)

        # Build row data
        row_data = [score_text, clickable]

        # Add code snippet if verbose
        if self.verbose:
            tgt_code = match.get("code") or ""
            syntax = Syntax(
                self._short(tgt_code), 
                "javascript", 
                theme="monokai", 
                word_wrap=True
            )
            row_data.append(syntax)

        # Add reasoning if enabled
        if self.reasoning:
            llm_reasoning = match.get("llm_reasoning", "")
            row_data.append(llm_reasoning)
        
        # Add completed row to table
        table.add_row(*row_data)
    
    @staticmethod
    def _is_pager_available() -> bool:
        """Check if less pager is available on the system.
        
        Returns:
            True if less command is available, False otherwise
        """
        try:
            subprocess.run(['which', 'less'], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
