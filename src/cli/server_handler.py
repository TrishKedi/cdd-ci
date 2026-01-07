"""Web server management for real-time duplication detection results.

This module provides a handler class for managing the web server lifecycle
and streaming real-time results to connected clients.
"""

import time
from typing import Optional
from threading import Thread
from rich.console import Console
from web.server import stream_completion_to_web, initialize_web_server, start_web_server


class ServerHandler:
    """Handles web server initialization and lifecycle management.
    
    This class manages the web server that provides real-time streaming of
    duplicate detection results to browser clients via WebSocket connections.
    """
    
    def __init__(
        self, 
        port: int = 8080, 
        verbose: bool = False, 
        reasoning: bool = False, 
        auto_open: bool = True
    ) -> None:
        """Initialize the server handler with configuration options.
        
        Args:
            port: Port number for the web server (default: 8080)
            verbose: Whether to show detailed output in web interface
            reasoning: Whether to include LLM reasoning in results
            auto_open: Whether to automatically open browser on server start
        """
        self.port = port
        self.verbose = verbose
        self.reasoning = reasoning
        self.auto_open = auto_open
        self.console = Console()

    def start_server(self) -> Thread:
        """Initialize and start the web server in a separate thread.
        
        Returns:
            Thread object for the running web server
        """
        # Initialize server configuration
        initialize_web_server(self.port, self.verbose, self.reasoning)
        
        # Start server in background thread
        server_thread = start_web_server(self.auto_open)
        
        # Display success message
        self.console.print(
            f" Web server started at http://localhost:{self.port}", 
            style="bold green"
        )
        
        return server_thread

    async def keep_running(self, serve_enabled: bool, server_thread: Optional[Thread]) -> None:
        """Keep the web server running and handle graceful shutdown.
        
        This method maintains the server thread and handles keyboard interrupts
        for clean shutdown of the web server.
        
        Args:
            serve_enabled: Whether web server mode is active
            server_thread: The server thread to monitor, if any
        """
        if serve_enabled and server_thread:
            # Signal completion to web clients
            await stream_completion_to_web()
            
            # Display instructions to user
            self.console.print("Press Ctrl+C to stop the web server", style="dim")
            
            try:
                # Keep server alive until interrupted
                while server_thread.is_alive():
                    time.sleep(1)  # Check server status every second
            except KeyboardInterrupt:
                # Handle graceful shutdown
                self.console.print("\n Web server stopped", style="yellow")