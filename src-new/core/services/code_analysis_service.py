"""Code analysis service for extracting and processing code fragments.

This module provides comprehensive code analysis capabilities including:
- Tree-sitter based syntax parsing for JavaScript and other languages
- Intelligent code fragment extraction (functions, methods, classes)
- File caching system to avoid reprocessing unchanged files
- Multi-processing support for efficient large codebase analysis
- Smart filtering of test files, minified code, and generated files
"""

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set, Iterator

import numpy as np
import typer
from tree_sitter import Language, Parser, Query, Tree, Node
import tree_sitter_javascript as tsjs

from config.settings import cache_file

@dataclass
class FileCache:
    """Cache entry for a processed file."""
    mtime: float           # File's modification time
    size: int             # File size
    fragments: List[Dict]  # Extracted code fragments
    hash: str             # Content hash

class FileProcessingCache:
    """Manages caching of processed files to avoid reprocessing unchanged files."""
    
    def __init__(self):
        self.cache_file = cache_file
        self.cache: Dict[str, FileCache] = {}
        self.load_cache()

    def save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                cache_dict = {
                    path: {
                        'mtime': entry.mtime,
                        'size': entry.size,
                        'fragments': entry.fragments,
                        'hash': entry.hash
                    }
                    for path, entry in self.cache.items()
                }
                json.dump(cache_dict, f)

        except Exception as e:
            typer.echo(f"⚠️ Cache saving failed: {e}")
    
    def load_cache(self) -> None:
        """Load cache from disk if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = {
                        path: FileCache(**entry)
                        for path, entry in data.items()
                    }
        except Exception as e:
            typer.echo(f"⚠️ Cache loading failed: {e}")
            self.cache = {}
    

# Default patterns for files to exclude
DEFAULT_EXCLUDE_PATTERNS = {
    # Build and output directories
    '**/dist/**',
    '**/build/**',
    '**/out/**',
    
    # Dependencies
    '**/node_modules/**',
    '**/bower_components/**',
    '**/vendor/**',
    
    # Test files
    '**/test/**',
    '**/tests/**',
    '**/_tests/**',
    '**/scenarios/**',
    '**/scenario_tests/**',
    '**/*.test.js',
    '**/*.spec.js',
    '**/qmate_tests/**',
    '**/localService/**',
    
    # Minified files
    '**/*.min.js',
    '**/*-min.js',
    '**/*bundle*.js',
    
    # Generated files
    '**/*.generated.js',
    '**/*.g.js',
    
    # Configuration files
    '**/webpack.config.js',
    '**/rollup.config.js',
    '**/jest.config.js',
    '**/babel.config.js',
    '**/*.config.js',
    '**/*.conf.js'
    
    # Source maps
    '**/*.js.map',

    # Configuration files
    '**/eslint.config.js',
    '**/lint-staged.config.js',
    '**/prettier.config.js',
    '**/ui5lint.config.js',

    # Temporary and cache files
    '**/.tmp/',
    '**/.eslintcache',
    '**/.prettiercache',

    # Test and Boilerplate folders/files
    
  
}


# Patterns for detecting minified JavaScript files
MINIFIED_JS_INDICATORS: List[str] = [
    r'sourceMappingURL',      # Source map reference
    r'\.min\.js$',           # .min.js extension
    r'^[^\\n]{500,}$',       # Long lines (typical in minified code)
    r'\]{3,}',              # Multiple closing brackets in sequence
    r'\}{3,}'               # Multiple closing braces in sequence
]
class CodeBaseProcessor:
    """Processes codebases to extract meaningful code fragments.
    
    This class handles the analysis of source code files using tree-sitter
    for syntax parsing and intelligent extraction of functions, methods,
    and other code constructs for similarity analysis.
    
    Attributes:
        code_blocks: List of extracted code blocks with metadata
        exclude_patterns: Set of glob patterns for files to exclude
        minified_patterns: List of compiled regex patterns to detect minified files
        cache: File processing cache to avoid reprocessing unchanged files
        cache_hits: Counter for cache hit statistics
        cache_misses: Counter for cache miss statistics
    """

    def __init__(self) -> None:
        """Initialize the code processor with default configuration."""
        # Core data structures
        self.code_blocks: List[Dict[str, Any]] = []
        self.exclude_patterns: Set[str] = DEFAULT_EXCLUDE_PATTERNS
        self.minified_patterns: List[re.Pattern[str]] = [
            re.compile(pattern) for pattern in MINIFIED_JS_INDICATORS
        ]
        
        # Cache management
        self.cache: FileProcessingCache = FileProcessingCache()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Internal counter for ID assignment
        self._id_counter: int = 0
        
        # Tree-sitter query patterns for different JavaScript constructs
        self.CONTROLLER_QUERY: str = r"""
        (call_expression
        function: (member_expression
            property: (property_identifier) @callee_name
        )
        arguments: (arguments
            (_)
            (object) @controller_obj
            .
        )
        (#eq? @callee_name "extend")
        )"""
        
        # Query patterns for extracting different method definition styles
        self.METHODS_QUERY: str = r"""
        ; 1) Method shorthand: onInit() { ... }
        (object
        (method_definition
            name: (property_identifier) @method_name
        )
        ) @m1

        ; 2) Pair with function value: onInit: function (...) { ... }
        (object
        (pair
            key: (property_identifier) @method_name
            value: (function) @fn
        )
        ) @m2

        ; 3) Pair with arrow function value: onSomething: (...) => { ... }
        (object
        (pair
            key: (property_identifier) @method_name
            value: (arrow_function) @afn
        )
        ) @m3

        ; (optional) quoted keys: "onInit": function () {}
        (object
        (pair
            key: (string) @method_name_str
            value: (function) @fn2
        )
        ) @m4

        (object
        (pair
            key: (string) @method_name_str
            value: (arrow_function) @afn2
        )
        ) @m5
        """

        # Query pattern for arrow functions and method definitions
        self.ARROW_FUNCS: str = r"""
        (object 
        (pair 
            key: (property_identifier) @method_name
            value: (arrow_function) @afn
        )
        )
      
        (object
        (method_definition
            name: (property_identifier) @method_name
        )
        ) @afn

        (pair
            key: (property_identifier) @method_name
            value: (function_expression) @afn
        )

        (method_definition
            name: (property_identifier) @method_name
        ) @afn
        """

        # Query pattern for various function declaration styles
        self.FUNCTION_QUERY: str = r"""
        (function_declaration) @func
        (lexical_declaration
        (variable_declarator
            name: (identifier) @var_name
            value: [(arrow_function) (function)] @func
        )
        )
        (expression_statement
        (assignment_expression
            left: (member_expression) @member
            right: [(arrow_function) (function)] @func
        )
        )
        (method_definition) @func
        (pair
        key: (property_identifier) @prop
        value: [(arrow_function) (function)] @func
        )
        """

        

        self.OBJECT_METHODS_AND_PAIRS = r"""
        ; method shorthand: onInit() { ... } / render() { ... }
        (method_definition
        name: (property_identifier) @prop_name
        ) @mdef

        ; key: function(...) { ... }  or  key: (...) => { ... }
        (pair
        key: (property_identifier) @prop_name
        value: (choice (function) (function_expression) (arrow_function))
        ) @mpair

        ; allow quoted keys: "onInit": function(){}, "render": ()=>{}
        (pair
        key: (string) @prop_name_str
        value: (choice (function) (function_expression) (arrow_function))
        ) @mpair_str
        """

    def load_code_blocks(self, code_blocks_list: List[Dict[str, Any]]) -> None:
        """Load code blocks from an external list.
        
        Args:
            code_blocks_list: List of code block dictionaries to load
        """
        self.code_blocks = code_blocks_list

    def set_all_embeddings(self, embeddings: np.ndarray) -> None:
        """Set embeddings for all loaded code blocks.
        
        Args:
            embeddings: NumPy array of embeddings, one per code block
        """
        for index, cb in enumerate(self.code_blocks):
            cb['embeddings'] = embeddings[index].tolist()
            
    def set_matches(self, block_id: str, matches: List[Dict[str, Any]]) -> None:
        """Set similarity matches for a specific code block.
        
        Args:
            block_id: Unique identifier of the code block
            matches: List of matching code blocks with similarity scores
        """
        # Find the block with the specified ID and update its matches
        for block in self.code_blocks:
            if block.get('id') == block_id:
                block['matches'] = matches
                break

    def get_code_blocks(self) -> List[Dict[str, Any]]:
        """Get the list of loaded code blocks.
        
        Returns:
            List of code block dictionaries with metadata
        """
        return self.code_blocks

    def get_changed_files(self):
        print("Get changed files")
        parser = argparse.ArgumentParser()
        parser.add_argument('--changed-files', required=True)
        args = parser.parse_args()

        with open(args.changed_files, 'r', encoding='utf-8') as f:
            files = f.read()
            changed_files = [Path(file) for file in files.splitlines('\n')]

            return changed_files

    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single JavaScript file to extract code blocks.
        
        This is a helper method designed to run in a separate process.
        
        Args:
            file_path: Path to the JavaScript file to process
            
        Returns:
            List of extracted code blocks with their metadata
        """
        try:
            fragments = self.fragments_from_file(file_path)
       
            return fragments
        except Exception as e:
            typer.echo(f"⚠️ Skipping {file_path}: {e}")
            return []

    def extract_candidate_files(self, code_dir:str)->List[Path]:
        directory = Path(code_dir)
        
        if not directory.exists() or not directory.is_dir():
            print(f"❌ {directory} is not a valid directory.")
            # raise typer.Exit(code=1)
            return

        # Get list of all JS files and filter them
        return list(directory.rglob("*.js"))

    def extract_raw_code(self, code_dir: str, batch_size: int = 50) -> Iterator[Dict[str, Any]]:
        """Extract code blocks from JavaScript files in a directory using parallel processing.
        
        This method combines parallel processing, lazy loading, and file caching to optimize
        both CPU and memory usage while avoiding reprocessing unchanged files. Instead of
        loading all files at once, it processes them in batches while yielding results
        incrementally.
        
        Args:
            code_dir: Path to the directory containing JavaScript files
            batch_size: Number of files to process in each batch
            
        Yields:
            Dict[str, Any]: Code blocks with their metadata, yielded as they're processed
            
        Raises:
            typer.Exit: If the directory does not exist or is not valid
        """
        self._id_counter = 0
        directory = Path(code_dir)
        
        if not directory.exists() or not directory.is_dir():
            typer.echo(f"❌ {directory} is not a valid directory.")
            raise typer.Exit(code=1)

        # Get list of all JS files and filter them
        all_js_files = list(directory.rglob("*.js"))
        js_files = [f for f in all_js_files if self.should_process_file(f)]
      
        
        
        if not js_files:
            typer.echo("No suitable JavaScript files found to process.")
            return

        # First pass: check cache and collect files needing processing
        files_to_process = []
        for file_path in js_files:

            cached_fragments = self.check_cache(file_path)
            if cached_fragments is not None:
                # Yield cached fragments immediately
                for fragment in cached_fragments:
                    fragment['id'] = self._id_counter
                    self._id_counter += 1
                    yield fragment
            else:
                files_to_process.append(file_path)

        if not files_to_process:
            typer.echo("✨ All files loaded from cache!")
            return

        # Calculate optimal number of processes for remaining files
        num_processes = min(cpu_count(), batch_size, len(files_to_process))
        total_files = len(files_to_process)
        processed_files = 0
        
        typer.echo(f"Processing {total_files} uncached files")

        # Process remaining files in batches
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            typer.echo(f"\nProcessing batch {(i//batch_size) + 1} ({len(batch)} files)")
            
            # Process batch in parallel
            with Pool(processes=num_processes) as pool:
                fragment_lists = pool.map(self._process_single_file, batch)
                
            # Update cache and yield fragments as they're processed
            for file_path, fragments in zip(batch, fragment_lists):
                if fragments:  # Only cache successful processing
                    self.update_cache(file_path, fragments)
                    for fragment in fragments:
                        fragment['id'] = self._id_counter
                        self._id_counter += 1
                        yield fragment
                    
            processed_files += len(batch)
            typer.echo(f"Progress: {processed_files}/{total_files} files processed")
        

        # Cache statistics are tracked in self.cache_hits and self.cache_misses
        # but not displayed to reduce verbosity

    def extract_code_chunks(self, files:List[str], is_candidate:bool, chunk_size:int=50, batch_size:int=50 ) -> Iterator[Dict[str, Any]]:
    
        if not files:
            return
            
        valid_files = [ file_path for file_path in files if self.should_process_file(file_path) ]

        if valid_files:
   
            num_processes = min(cpu_count(), batch_size, len(valid_files))

            total_files = len(valid_files)
            processed_files = 0
            
            typer.echo(f"Processing {total_files} files")

            # Process remaining files in batches
            for i in range(0, len(valid_files), batch_size):
                batch = valid_files[i:i + batch_size]
                typer.echo(f"\nProcessing batch {(i//batch_size) + 1} ({len(batch)} files)")
                
                # Process batch in parallel
                with Pool(processes=num_processes) as pool:
                    fragment_lists = pool.map(self._process_single_file, batch)
                    
                # Update cache and yield fragments as they're processed
                for fragments in  fragment_lists:
                
                    for fragment in fragments:

                        fragment['id'] = self._id_counter
                        self._id_counter += 1

                        if is_candidate:
                            with open("chunks.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(fragment, ensure_ascii=False) + "\n")
                        
                        yield fragment
                        
                processed_files += len(batch)
                typer.echo(f"Progress: {processed_files}/{total_files} files processed")

               

    def assign_ids(self, all_fragments: List[Dict[str, Any]]) -> None:
        """Assign unique integer IDs to code fragments.
        
        Args:
            all_fragments: List of code fragment dictionaries to assign IDs to
        """
        for frag_key, frag in enumerate(all_fragments):
            frag['id'] = frag_key

    def update_cache(self, file_path: Path, fragments: List[Dict[str, Any]]) -> None:
        """Update the cache with new file fragments.
        
        Args:
            file_path: Path to the processed file
            fragments: List of extracted code fragments
        """
        try:
            stat = file_path.stat()
            self.cache.cache[str(file_path)] = FileCache(
                mtime=stat.st_mtime,
                size=stat.st_size,
                fragments=fragments,
                hash=hashlib.sha256(file_path.read_bytes()).hexdigest()
            )
            self.cache.save_cache()  # Persist cache to disk
        except Exception as e:
            typer.echo(f"⚠️ Cache update failed for {file_path}: {e}")
            
    def check_cache(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Check if a file is in the cache and return its fragments if valid.
        
        This method:
        1. Checks if the file exists in the cache
        2. Validates file metadata (size and mtime) hasn't changed
        3. Returns cached fragments if valid, None otherwise
        
        Args:
            file_path: Path to the file to check in cache
            
        Returns:
            List of cached fragments if valid cache exists, None otherwise
        """
        try:
            stat = file_path.stat()
            path_str = str(file_path)
            
            # Check if file is in cache and metadata matches
            if path_str in self.cache.cache:  # Use FileProcessingCache's cache dict
                cached = self.cache.cache[path_str]
                if (cached.mtime == stat.st_mtime and 
                    cached.size == stat.st_size):
                    self.cache_hits += 1
                    return cached.fragments
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            typer.echo(f"⚠️ Cache check failed for {file_path}: {e}")
            return None
        

    def parse_js(self, code: str) -> Tree:
        """Parse JavaScript code using tree-sitter parser.
        
        Note: This method expects self.parser to be initialized with a JavaScript language.
        
        Args:
            code: JavaScript source code string
            
        Returns:
            Tree-sitter Tree object representing the parsed AST
        """
        return self.parser.parse(bytes(code, "utf-8"))

    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed based on filtering rules.
        
        This method applies several filters to determine if a file should be processed:
        1. Checks against exclude patterns (test files, minified files, etc.)
        2. Analyzes file content for minification
        3. Validates file size
        
        Args:
            file_path: Path to the JavaScript file
            
        Returns:
            bool: True if the file should be processed, False if it should be skipped
        """
        try:
            if not self._passes_exclude_patterns(file_path):
                return False
                
            if not self._passes_size_check(file_path):
                return False
                
            if not self._passes_minification_check(file_path):
                return False
            
            return True
            
        except Exception as e:
            typer.echo(f"⚠️ Error checking file {file_path}: {e}")
            return False
            
    def _passes_exclude_patterns(self, file_path: Path) -> bool:
        """Check if file passes exclude pattern filters."""
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return False

            if pattern.startswith('**/') and pattern.endswith('/**'):
                # Pattern like '**/node_modules/**' - check if directory is in path parts
                folder_name = pattern[3:-3]  # Remove '**/' and '/**'
                if folder_name in file_path.parts:
                    return False
        return True
        
    def _passes_size_check(self, file_path: Path) -> bool:
        """Check if file size is within acceptable limits."""
        return file_path.stat().st_size <= 1_000_000  # Skip files larger than 1MB
        
    def _passes_minification_check(self, file_path: Path) -> bool:
        """Check if file appears to be minified based on content analysis."""
        sample = file_path.read_text(encoding='utf-8', errors='ignore')[:5000]
        
        # Check for minification indicators
        for pattern in self.minified_patterns:
            if pattern.search(sample):
                return False
        
        # Check for suspiciously long lines (typical in minified code)
        max_line_length = max((len(line) for line in sample.splitlines()[:10]), default=0)
        return max_line_length <= 500


    def iter_captures(self, query: Query, root: Node) -> Iterator[Tuple[Node, str]]:
        """Normalize Tree-sitter captures to (node, capture_name) across versions.
        
        Different versions of tree-sitter return captures in different formats:
        - Some return (node, name), others (node, index, pattern_index)
        This method normalizes the format to always return (node, name).
        
        Args:
            query: Tree-sitter query object
            root: Root AST node to search in
            
        Yields:
            Tuple of (node, capture_name) for each match
        """
        for cap in query.captures(root):
            node = cap[0]
            second = cap[1]
            if isinstance(second, str):
                name = second
            else:
                name = query.capture_names[second]
            yield node, name

    def get_controller_nodes(self, tree: Tree, language: Optional[Language] = None) -> List[Node]:
        """Extract controller object nodes from a JavaScript AST.
        
        Looks for patterns matching controller objects in Angular-style code,
        specifically objects passed to .extend() calls.
        
        Args:
            tree: The parsed tree-sitter AST to search
            language: Optional tree-sitter language instance for the query
            
        Returns:
            List of tree-sitter nodes representing controller objects
            
        Raises:
            ValueError: If language is None and no default language is available
        """
        if language is None:
            raise ValueError("Language parameter is required for controller node extraction")
            
        controller_q = Query(language, self.CONTROLLER_QUERY)
        controller_q_result = controller_q.captures(tree.root_node)
        
        # Extract controller objects from query results
        controller_objs = controller_q_result.get('controller_obj', [])
        return controller_objs if isinstance(controller_objs, list) else [controller_objs]

    def extract_function_nodes(self, root_node: Node, code_bytes: bytes, language: Optional[Language] = None) -> List[Dict[str, Any]]:
        """Extract function-like nodes from a JavaScript AST.
        
        This method identifies and extracts various types of function definitions
        including function declarations, expressions, arrow functions, and
        method definitions.
        
        Args:
            root_node: The root node of the AST to search
            code_bytes: The raw bytes of the source code
            language: Optional tree-sitter language instance for the query
            
        Returns:
            List of dictionaries containing extracted function information with
            keys: 'code', 'start', 'end', and optional metadata
            
        Raises:
            ValueError: If language is None
        """
        if language is None:
            raise ValueError("Language parameter is required for function node extraction")
            
        query = Query(language, self.ARROW_FUNCS)
        
        # Valid function-like node types
        FUNCTION_TYPES = {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
        }
        
        fragments: List[Dict[str, Any]] = []
        func_captures = query.captures(root_node)
        
        # Extract function nodes from query results
        nodes = func_captures.get('afn', [])
        if not isinstance(nodes, list):
            nodes = [nodes] if nodes else []
         
        for node in nodes:
            if node.type in FUNCTION_TYPES:
                start_byte, end_byte = node.start_byte, node.end_byte
                fragment_code = code_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")
                
                # Calculate line numbers more efficiently
                start_line = code_bytes[:start_byte].count(b'\n') + 1
                end_line = code_bytes[:end_byte].count(b'\n') + 1

                fragments.append({
                    "code": fragment_code,
                    "start": start_line,
                    "end": end_line,
                })

        return fragments


    def normalize_js(self, snippet: str) -> str:
        """Normalize JavaScript code by removing comments and semicolons, and compacting whitespace.
        
        This is a lighter normalization that preserves variable names and structure
        while removing syntax noise for similarity comparison.
        
        Args:
            snippet: JavaScript code string to normalize
            
        Returns:
            Normalized JavaScript code string
        """
        # Regex patterns for different code elements
        _space = re.compile(r"\s+")
        
        # Remove comments
        snippet = re.sub(r"//.*$", "", snippet, flags=re.M)
        snippet = re.sub(r"/\*.*?\*/", "", snippet, flags=re.S)
        
        # Normalize whitespace and remove semicolons
        snippet = snippet.replace(";", "")
        snippet = _space.sub(" ", snippet).strip()


        return snippet

    def fragments_from_file(self, file_path: Path, parser: Optional[Parser] = None, language: Optional[Language] = None) -> List[Dict[str, Any]]:
        """Extract and process code fragments from a JavaScript file.
        
        This method:
        1. Parses the JavaScript file
        2. Detects if it contains an Angular-style controller
        3. Extracts function-like nodes
        4. Normalizes the extracted code
        
        Args:
            file_path: Path to the JavaScript file to process
            parser: Optional tree-sitter parser instance. If not provided, creates a new one
            language: Optional tree-sitter language instance. If not provided, creates a new one
            
        Returns:
            List of dictionaries containing code blocks with metadata:
            - code: Original code fragment
            - processedCode: Normalized version of the code
            - path: Source file path
            - start: Starting line number
            - end: Ending line number
            
        Raises:
            UnicodeDecodeError: If file cannot be read as UTF-8
            tree_sitter.LanguageError: If parsing fails
        """
        # Initialize or use provided parser/language
        if language is None:
            language = Language(tsjs.language())
        if parser is None:
            parser = Parser(language)
            
        # Read and parse the file
        code = file_path.read_text(encoding="utf-8")
        tree = parser.parse(bytes(code, "utf-8"))
        root_node = tree.root_node
        
        # Check for SAP UI5 controller pattern
        ctr_node = self.get_controller_nodes(tree, language)
        if ctr_node:

            root_node = ctr_node[0]
        
        # Extract function nodes using existing method
        code_bytes = code.encode("utf-8")
        fragments = self.extract_function_nodes(root_node, code_bytes, language)
        
        # Process and normalize fragments
        processed_fragments = [
            {
                **fragment,
                "processedCode": self.normalize_js(fragment.get('code')),
                'path': str(file_path)
            }
            for fragment in fragments
        ]
        
        return processed_fragments

        
    

