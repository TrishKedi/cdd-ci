"""Export functionality for code duplication detection results.

This module provides utilities to export duplicate detection results
to various formats including JSON and JSONL files with streaming support
for lazy match generation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from config.settings import json_export_file, jsonl_export_file, export_dir


class Exporter:
    """Handles exporting of duplicate code detection results to files.
    
    Supports streaming export for lazy match generation, writing matches
    one at a time as they are discovered. Supports both JSON arrays and
    newline-delimited JSON (JSONL) formats with enhanced user experience.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the exporter with configuration options.
        
        Args:
            **kwargs: Configuration options including:
                - auto_open: Whether to automatically open exported files
                - export_dir: Custom export directory (defaults to Downloads)
                - verbose: Whether to show detailed export information
        """
        self.auto_open: bool = kwargs.get("auto_open", False)
        self.export_dir: Path = Path(kwargs.get("export_dir", export_dir))
        self.verbose: bool = kwargs.get("verbose", False)
        self.console = Console()
        
        # Track export state
        self.json_file_path: Optional[Path] =  self.export_dir / "diagnostics.json"
        self.jsonl_file_path: Optional[Path] = None
        self.match_count: int = 0
        self.is_json_initialized: bool = False
        
        # Ensure export directory exists
        self._ensure_export_dir()

    def _get_downloads_dir(self) -> Path:
        """Get the user's Downloads directory.
        
        Returns:
            Path to the Downloads directory, falls back to current dir if not found
        """
        # Try to get Downloads directory for different operating systems
        downloads_paths = [
            Path.home() / "Downloads",  # macOS/Linux
            Path.home() / "downloads",  # Alternative case
            Path.home() / "Desktop",    # Fallback
            Path.cwd() / "exports",     # Final fallback
        ]
        
        for downloads_path in downloads_paths:
            if downloads_path.exists() and downloads_path.is_dir():
                return downloads_path
        
        # Create exports directory if none found
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(exist_ok=True)
        return exports_dir

    def _ensure_export_dir(self) -> None:
        """Ensure the export directory exists and is writable."""
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.export_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
        except (OSError, PermissionError) as e:
            # Fallback to current directory if Downloads not writable
            # self.console.print(f"[yellow]Warning: Cannot write to {self.export_dir}, using current directory[/yellow]")
            self.export_dir = Path.cwd() / "exports"
            self.export_dir.mkdir(exist_ok=True)

    def _generate_filename(self, base_name: str, extension: str) -> str:
        """Generate a unique filename to avoid overwriting existing files.
        
        Args:
            base_name: Base name for the file
            extension: File extension (without dot)
            
        Returns:
            Unique filename with timestamp if needed
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.{extension}"
        
        # Ensure filename is unique
        full_path = self.export_dir / filename
        counter = 1
        while full_path.exists():
            filename = f"{base_name}_{timestamp}_{counter}.{extension}"
            full_path = self.export_dir / filename
            counter += 1
            
        return filename

    def initialize_export(self) -> None:
        """Initialize the JSON array export file with opening bracket.
        
        This should be called once before any export_json calls.
        """
        if not self.is_json_initialized:
            
            with open(self.json_file_path, "w", encoding="utf-8") as f:
                header = '''{"source": {"name": "find-duplicates"}\n
                "duplicates": [
                \n
                '''
                f.write(header)
            
            self.is_json_initialized = True
            
       
    def export_json(self, matches: Dict[str, Any]) -> None:
        """Append match results to JSON array file.
        
        Args:
            matches: Dictionary containing match results to export
        """
        if not self.is_json_initialized:
            self.initialize_export()
        
        # Add comma before new entry if not first match
        prefix = ",\n" if self.match_count > 0 else ""
        
        with open(self.json_file_path, "a", encoding="utf-8") as f:
            f.write(prefix)
            json.dump(matches, f, indent=2, ensure_ascii=False, default=str)
            self.match_count += 1

    def finalize_json_export(self) -> None:
        """Close the JSON array export file with closing bracket.
        
        This should be called once after all export_json calls are complete.
        """
        if self.is_json_initialized and self.json_file_path:
            with open(self.json_file_path, "a", encoding="utf-8") as f:
                f.write("\n]\n}")
            
            self._notify_export_success(self.json_file_path, "JSON")

    def export_jsonl(self, matches: Dict[str, Any]) -> None:
        """Append match results to JSONL (newline-delimited JSON) file.
        
        Args:
            matches: Dictionary containing match results to export
        """
        # Initialize JSONL file path if not set
        if not self.jsonl_file_path:
            filename = self._generate_filename("duplicate_matches", "jsonl")
            self.jsonl_file_path = self.export_dir / filename
            
         

        with open(self.jsonl_file_path, "a", encoding="utf-8") as f:
            json.dump(matches, f, separators=(',', ':'), ensure_ascii=False, default=str)
            f.write("\n")

    def export(self, matches: Dict[str, Any]) -> None:
        """Export match results to both JSONL and JSON formats.
        
        Args:
            matches: Dictionary containing match results to export
        """
        # self.export_jsonl(matches)
        self.export_json(matches)
        self.match_count += 1

    def finalize_export(self) -> None:
        """Finalize all export operations and notify user.
        
        Should be called after all matches have been exported.
        """
        # Finalize JSON export
        self.finalize_json_export()
        
        # Notify about JSONL export
        if self.jsonl_file_path:
            self._notify_export_success(self.jsonl_file_path, "JSONL")
        

    def _notify_export_success(self, file_path: Path, format_type: str) -> None:
        """Notify user of successful export and optionally open file.
        
        Args:
            file_path: Path to the exported file
            format_type: Type of format exported (JSON/JSONL)
        """
        # Calculate file size for user info
        if file_path.exists():
            file_size = self._format_file_size(file_path.stat().st_size)
            
  
            
            # Create clickable file path for supported terminals
            clickable_path = f"[link=file://{file_path.resolve()}]{file_path}[/link]"
            # self.console.print(f"[cyan] Location: {clickable_path}[/cyan]")
           
            
            # Auto-open file if requested
            if self.auto_open:
                self._open_file(file_path)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string (e.g., "1.2 KB", "3.4 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _open_file(self, file_path: Path) -> None:
        """Attempt to open the exported file with the system default application.
        
        Args:
            file_path: Path to the file to open
        """
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(file_path)], check=True)
            elif sys.platform == "win32":  # Windows
                os.startfile(str(file_path))
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", str(file_path)], check=True)
                            
        except (subprocess.SubprocessError, OSError, AttributeError) as e:
            pass
            # self.console.print(f"[yellow]Could not auto-open file: {e}[/yellow]")
            # self.console.print(f"[dim]You can manually open: {file_path}[/dim]")

    def cleanup(self) -> None:
        """Clean up any temporary resources if needed."""
        # Reset state for potential reuse
        self.match_count = 0
        self.is_json_initialized = False
        self.json_file_path = None
        self.jsonl_file_path = None

    def export_rdjson(self, search_results):
        json.dump(search_results, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")

    def export_diagonistics(self, search_results):
          with open(self.json_file_path, "w", encoding="utf-8") as f:
            json.dump(search_results, f, indent=2, ensure_ascii=False, default=str)

    def stream_diagonistics(self, search_results):
        for result in search_results:

            diagnostics = {
                "source": {"name": "find-duplicates"},
                "diagnostic": result
            }


            sys.stdout.write(json.dumps(diagnostics, ensure_ascii=False) + "\n")
            sys.stdout.flush()