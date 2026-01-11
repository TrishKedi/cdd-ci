"""Repository management for local directories and remote git cloning.

This module handles validation and preparation of code locations,
including cloning remote repositories and managing temporary directories.
"""

import os
import shutil
import subprocess
from typing import List, Optional, Dict
from rich.console import Console
from config.settings import tmp_dir
from pathlib import Path


class RepositoryManager:
    """Manages code repository access and temporary directory lifecycle.
    
    This class handles both local directory validation and remote repository
    cloning, ensuring that code locations are properly prepared for analysis.
    """

    def __init__(self) -> None:
        """Initialize the repository manager with console output."""
        self.console = Console()


    def _prepare_code_locations(
        self, 
        candidate_repo: str, 
        changed_files: str
    ) -> Dict[str, str]:
        """Prepare and validate code locations for processing.
        
        Validates input parameters and either returns local directories
        or clones remote repositories as needed.
        
        Args:
            candidate_repo: List of local directory paths
            clone: List of git repository URLs to clone
            
        Returns:
            List of prepared directory paths, or None if validation fails
        """
        # Validate input parameters
        validation_result = self._validate_candidate_repo(candidate_repo, changed_files)
        if validation_result is None:
            return None
        
        # Process local directories
        return {
            "candidate_repo": candidate_repo,
            "changed_files": changed_files
        }
        
            
    def _validate_candidate_repo(
        self, 
        candidate_repo: str, 
        changed_files: str
    ) -> Optional[bool]:
        """Validate that exactly one input method is provided.
        
        Args:
            candidate_repo: List of local directory paths
            clone: List of git repository URLs
            
        Returns:
            True if validation passes, None if validation fails
        """
        if not candidate_repo and not changed_files:
            self.console.print(
                " You must provide candidate repository with --candidate-repo and changed files with --changed-files", 
                style="red"
            )

            return None
            
        return True

    def extract_candidate_files(self, code_dir:str)->List[Path]:
        directory = Path(code_dir)
        
        if not directory.exists() or not directory.is_dir():
            # print(f"❌ {directory} is not a valid directory.")
            # raise typer.Exit(code=1)
            return

        # Get list of all JS files and filter them
        return list(directory.rglob("*.js"))

    def get_changed_files(self, changed_files_path):
        # print("Get changed files")
        # print(changed_files_path)
        files = [
            "repos/code-samples-ui/Cart.controller.js",
            "repos/code-samples-ui/Category.controller.js",
            "repos/code-samples-ui/Checkout.controller.js"
        ]

        return [Path(file) for file in files]
      
        # with open('changed-files.txt', 'r', encoding='utf-8') as f:
        #     files = f.read()
        #     changed_files = [Path(file) for file in files.splitlines('\n')]
        #     print(changed_files)

        #     return changed_files

    def _clean_up(self, path) -> None:
        if os.path.exists(path):
            os.remove(path)
        
