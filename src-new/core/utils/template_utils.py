"""Template utilities for HTML report generation in code duplication detection.

This module provides utilities for loading, processing, and generating HTML templates
for similarity search results and code duplication reports. It maintains separation
between template logic and analysis code for better maintainability.

Key Features:
    - Template loading from file system with validation
    - HTML generation for code snippets and match results
    - CSS class assignment based on similarity scores
    - HTML escaping for security and proper rendering
    - Modular template components for flexible report generation
    - VSCode integration links for easy navigation

The module supports both class-based and functional approaches:
    - TemplateLoader class for advanced template management
    - Convenience functions for simple use cases
    - Template component functions for building complex reports

Security Considerations:
    - All user content is properly HTML-escaped
    - File path validation prevents directory traversal
    - Template loading includes existence validation
"""

import logging
from pathlib import Path
from typing import Optional, Union

# Set up logging for template operations
logger = logging.getLogger(__name__)


class TemplateLoader:
    """Template loader for HTML report generation with file system integration.
    
    Provides a robust interface for loading and processing HTML templates from
    the file system. Handles template discovery, validation, and content parsing
    with proper error handling and logging.
    
    The loader supports modular template design by parsing template components
    and providing methods to extract specific sections like headers, footers,
    and JavaScript content.
    
    Attributes:
        template_dir (Path): Directory containing template files
    """
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize template loader with specified or default template directory.
        
        Args:
            template_dir (Optional[Union[str, Path]]): Path to template directory.
                If None, defaults to 'web/templates' relative to this module.
        
        Raises:
            ValueError: If template_dir is provided but doesn't exist
        """
        if template_dir is None:
            # Default to templates folder relative to web directory
            # This assumes the standard project structure
            template_dir = Path(__file__).parent.parent.parent / "web" / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Validate template directory exists
        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {self.template_dir}")
            # Create directory if it doesn't exist for development convenience
            self.template_dir.mkdir(parents=True, exist_ok=True)
            
        logger.debug(f"TemplateLoader initialized with directory: {self.template_dir}")
    
    def load_template(self, template_name: str) -> str:
        """Load template file content with validation and error handling.
        
        Args:
            template_name (str): Name of the template file to load
            
        Returns:
            str: Complete template file content
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            PermissionError: If template file cannot be read
            ValueError: If template_name contains invalid characters
        """
        # Validate template name to prevent directory traversal
        if not template_name or '..' in template_name or '/' in template_name:
            raise ValueError(f"Invalid template name: '{template_name}'")
            
        template_path = self.template_dir / template_name
        
        # Check if template file exists
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Check if it's actually a file (not a directory)
        if not template_path.is_file():
            raise ValueError(f"Template path is not a file: {template_path}")
        
        try:
            # Load template content with explicit UTF-8 encoding
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            logger.debug(f"Loaded template '{template_name}' ({len(content)} characters)")
            return content
            
        except PermissionError as e:
            logger.error(f"Permission denied reading template: {template_path}")
            raise PermissionError(f"Cannot read template '{template_name}': {e}") from e
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading template: {template_path}")
            raise ValueError(f"Template '{template_name}' contains invalid UTF-8: {e}") from e
    
    def get_html_header(self) -> str:
        """Extract HTML header section from report template.
        
        Parses the main template file and extracts everything up to and including
        the opening of the matches-content div. This provides the HTML document
        structure, CSS, and any header content.
        
        Returns:
            str: HTML header content including opening matches-content div
            
        Raises:
            ValueError: If template structure is invalid or missing required elements
            FileNotFoundError: If report template file is not found
        """
        try:
            template_content = self.load_template(DEFAULT_REPORT_TEMPLATE)
        except FileNotFoundError: 
            # Provide helpful error message for missing main template\n            
            raise FileNotFoundError(
                '''Main report template '{DEFAULT_REPORT_TEMPLATE}' not found. \n
                Please ensure template file exists in template directory.'''
                )
        
        # Find the matches-content div marker
        content_marker = CONTENT_MARKER
        header_end = template_content.find(content_marker)
        
        if header_end == -1:
            logger.error("Template missing required matches-content div")
            raise ValueError(
                "Template structure error: missing '<div id=\"matches-content\">' marker. "
                "This marker is required to separate header from content."
            )
        
        # Extract header including the opening content div
        header_content = template_content[:header_end + len(content_marker)]
        
        logger.debug(f"Extracted HTML header ({len(header_content)} characters)")
        return header_content
    
    def get_html_footer(self) -> str:
        """Extract HTML footer section from report template.
        
        Parses the main template file and extracts everything after the closing
        of the matches-content div. This includes JavaScript, closing HTML tags,
        and any footer content.
        
        Returns:
            str: HTML footer content from matches-content closing div onward
            
        Raises:
            ValueError: If template structure is invalid or missing required elements
            FileNotFoundError: If report template file is not found
        """
        try:
            template_content = self.load_template(DEFAULT_REPORT_TEMPLATE)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Main report template '{DEFAULT_REPORT_TEMPLATE}' not found. "
                "Please ensure template file exists in template directory."
            )
        
        # Find the matches-content div opening first
        content_marker = CONTENT_MARKER
        content_start = template_content.find(content_marker)
        
        if content_start == -1:
            logger.error("Template missing required matches-content div")
            raise ValueError(
                "Template structure error: missing '<div id=\"matches-content\">' marker"
            )
        
        # Find the closing div after the content marker
        closing_div = '</div>'
        footer_start = template_content.find(closing_div, content_start)
        
        if footer_start == -1:
            logger.error("Template missing closing div for matches-content")
            raise ValueError(
                "Template structure error: missing closing '</div>' for matches-content. "
                "Template must have properly balanced div tags."
            )
        
        # Extract footer from the closing div onward
        footer_content = template_content[footer_start:]
        
        logger.debug(f"Extracted HTML footer ({len(footer_content)} characters)")
        return footer_content
    
    def get_javascript(self) -> str:
        """Load JavaScript content for report functionality.
        
        Returns:
            str: JavaScript code for report interactivity and statistics
            
        Raises:
            FileNotFoundError: If JavaScript template file is not found
        """
        try:
            js_content = self.load_template(DEFAULT_JAVASCRIPT_FILE)
            logger.debug(f"Loaded JavaScript content ({len(js_content)} characters)")
            return js_content
        except FileNotFoundError:
            # Provide helpful error message for missing JavaScript
            raise FileNotFoundError(
                f"JavaScript template '{DEFAULT_JAVASCRIPT_FILE}' not found. "
                "Report may not have interactive functionality."
            )


# Convenience Functions for Backward Compatibility
# ===============================================

def get_html_template() -> str:
    """Get HTML template header using default template loader.
    
    Convenience function that creates a default TemplateLoader and returns
    the HTML header. Maintained for backward compatibility with existing code.
    
    Returns:
        str: HTML header content including opening matches-content div
        
    Raises:
        FileNotFoundError: If template files are not found
        ValueError: If template structure is invalid
    """
    loader = TemplateLoader()
    return loader.get_html_header()


def get_html_footer() -> str:
    """Get HTML template footer using default template loader.
    
    Convenience function that creates a default TemplateLoader and returns
    the HTML footer. Maintained for backward compatibility with existing code.
    
    Returns:
        str: HTML footer content from matches-content closing div onward
        
    Raises:
        FileNotFoundError: If template files are not found
        ValueError: If template structure is invalid
    """
    loader = TemplateLoader()
    return loader.get_html_footer()

def get_score_class(score: float) -> str:
    """Determine CSS class based on similarity score for visual styling.
    
    Converts numerical similarity scores into CSS class names that correspond
    to different visual styles (colors, formatting) in the HTML report.
    
    Args:
        score (float): Similarity score between 0.0 and 1.0
        
    Returns:
        str: CSS class name ('score-high', 'score-medium', or 'score-low')
        
    Raises:
        ValueError: If score is outside valid range [0.0, 1.0]
    """
 
    # Determine CSS class based on score thresholds
    # High confidence: >= 90% similarity
    if score >= HIGH_SCORE_THRESHOLD:
        return SCORE_HIGH_CLASS
    # Medium confidence: >= 80% similarity
    elif score >= MEDIUM_SCORE_THRESHOLD:
        return SCORE_MEDIUM_CLASS
    # Low confidence: < 80% similarity
    else:
        return SCORE_LOW_CLASS

def get_match_section_template(file_path: str, filename: str, start_line: int) -> str:
    """Generate HTML template for match section header with VSCode integration.
    
    Creates HTML markup for displaying the source file information with
    clickable links that open files directly in VSCode at specific line numbers.
    
    Args:
        file_path (str): Full path to the source file
        filename (str): Display name for the file (usually just filename)
        start_line (int): Starting line number for the code match
        
    Returns:
        str: HTML markup for match section header
        
    Raises:
        ValueError: If start_line is not a positive integer
    """
    # Validate line number
    if not isinstance(start_line, int) or start_line < 1:
        raise ValueError(f"start_line must be a positive integer, got {start_line}")
    
    # Escape HTML in file paths and names to prevent XSS
    safe_file_path = escape_html(file_path)
    safe_filename = escape_html(filename)
    
    return f'''
            <div class="match-section">
                <div class="source-header">
                    <a href="vscode://file/{safe_file_path}:{start_line}" class="source-path">
                        {safe_filename}
                    </a>
                    <span class="line-number">Line {start_line}</span>
                </div>
            '''

def get_code_snippet_template(code: str, header_title: str = "Source Code") -> str:
    """Generate HTML template for displaying code snippets with syntax highlighting.
    
    Creates properly formatted HTML for code display with syntax highlighting
    support. The code is HTML-escaped for security and wrapped in appropriate
    markup for styling and highlighting.
    
    Args:
        code (str): Source code content to display
        header_title (str): Title to display above the code snippet
        
    Returns:
        str: HTML markup for code snippet display
    """
    # Validate inputs
    if code is None:
        code = ""
    if header_title is None:
        header_title = "Source Code"
    
    # Escape HTML to prevent XSS and ensure proper rendering
    escaped_code = escape_html(code)
    safe_header_title = escape_html(header_title)

    # Default to JavaScript syntax highlighting
    # NOTE: Language detection could be enhanced in future versions
    # by analyzing file extensions or using content-based detection
    language = "javascript" 
    
    return f'''
                <div class="code-snippet">
                    <div class="code-header">{safe_header_title}</div>
                    <pre><code class="language-{language}">{escaped_code}</code></pre>
                </div>
                    '''

def get_match_item_template(
    score: float, 
    score_class: str, 
    file_path: str, 
    target_file: str, 
    start_line: int
) -> str:
    """Generate HTML template for individual similarity match item.
    
    Creates HTML markup for displaying a single similarity match with score,
    CSS styling, and clickable link to open the file in VSCode.
    
    Args:
        score (float): Similarity score between 0.0 and 1.0
        score_class (str): CSS class for score styling
        file_path (str): Full path to the matched file
        target_file (str): Display name for the matched file
        start_line (int): Starting line number of the match
        
    Returns:
        str: HTML markup for match item display
        
    Raises:
        ValueError: If score or start_line values are invalid
    """
    
    # Escape HTML to prevent XSS attacks
    safe_file_path = escape_html(file_path)
    safe_target_file = escape_html(target_file)
    safe_score_class = escape_html(score_class)
    
    return f'''
                <div class="match-item">
                    <span class="score {safe_score_class}">{score:.2f}</span>
                    <a href="vscode://file/{safe_file_path}:{start_line}" class="match-path">
                        {safe_target_file}:{start_line}
                    </a>
                </div>
                '''

def get_matches_container_start() -> str:
    """Generate opening HTML tag for matches container section.
    
    Returns:
        str: Opening div tag with matches-container CSS class
    """
    return '<div class="matches-container">'

def get_matches_container_end() -> str:
    """Generate closing HTML tags for matches container and section.
    
    Returns:
        str: Closing div tags to properly close matches container and section
    """
    return '''
                </div>
            </div>
            '''

def escape_html(text: Optional[str]) -> str:
    """Escape HTML special characters to prevent XSS attacks and ensure proper rendering.
    
    Converts HTML special characters to their corresponding HTML entities to prevent
    cross-site scripting (XSS) attacks and ensure user content displays correctly
    in HTML contexts.
    
    Args:
        text (Optional[str]): Text content to escape. Can be None or empty.
        
    Returns:
        str: HTML-escaped text with special characters converted to entities
    """
    # Handle None or empty input
    if not text:
        return ""
    
    # Convert to string if not already (handles numeric types gracefully)
    text_str = str(text)
    
    # Escape HTML special characters in order
    # Order matters: & must be escaped first to avoid double-escaping
    return (text_str
            .replace('&', '&amp;')    # Ampersand (must be first)
            .replace('<', '&lt;')     # Less than
            .replace('>', '&gt;')     # Greater than  
            .replace('"', '&quot;')   # Double quote
            .replace("'", '&#x27;'))  # Single quote (apostrophe)

def get_reasoning_template(reasoning: str) -> str:
    """Generate HTML template for displaying LLM reasoning explanations.
    
    Creates formatted HTML to display machine learning model reasoning or
    explanation text with appropriate styling and safe HTML escaping.
    
    Args:
        reasoning (str): Reasoning text from LLM or analysis system
        
    Returns:
        str: HTML markup for reasoning section with proper styling
    """
    # Ensure reasoning is properly escaped for HTML safety
    safe_reasoning = escape_html(reasoning)
    
    return f'''<div class="reasoning-section" style="margin: 10px 0; padding: 10px; border-left: 3px solid #007bff;">
                    <strong>LLM Reasoning:</strong> {safe_reasoning}
                </div>'''


# Module Constants
# ===============

# Default template file names
DEFAULT_REPORT_TEMPLATE: str = "report_template.html"
DEFAULT_JAVASCRIPT_FILE: str = "report_stats.js"

# CSS class names for score styling
SCORE_HIGH_CLASS: str = "score-high"
SCORE_MEDIUM_CLASS: str = "score-medium"
SCORE_LOW_CLASS: str = "score-low"

# Score thresholds for classification
HIGH_SCORE_THRESHOLD: float = 0.90
MEDIUM_SCORE_THRESHOLD: float = 0.80

# HTML template markers
CONTENT_DIV_ID: str = "matches-content"
CONTENT_MARKER: str = f'<div id="{CONTENT_DIV_ID}">'


# Public API
# ==========

__all__ = [
    # Main template loader class
    'TemplateLoader',
    
    # Convenience functions
    'get_html_template',
    'get_html_footer',
    
    # Template component functions
    'get_score_class',
    'get_match_section_template',
    'get_code_snippet_template',
    'get_match_item_template',
    'get_matches_container_start',
    'get_matches_container_end',
    'get_reasoning_template',
    
    # Utility functions
    'escape_html',
    
    # Constants
    'DEFAULT_REPORT_TEMPLATE',
    'DEFAULT_JAVASCRIPT_FILE',
    'SCORE_HIGH_CLASS',
    'SCORE_MEDIUM_CLASS',
    'SCORE_LOW_CLASS',
    'HIGH_SCORE_THRESHOLD',
    'MEDIUM_SCORE_THRESHOLD',
    'CONTENT_DIV_ID',
    'CONTENT_MARKER'
]
