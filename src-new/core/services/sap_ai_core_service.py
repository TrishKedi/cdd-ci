"""SAP AI Core service for centralized AI model interactions.

This module provides a unified interface for interacting with various AI models
including OpenAI embeddings and Claude for code similarity evaluation.
Handles authentication, token management, and API communication.
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from dotenv import load_dotenv, set_key

# Constants
CONTENT_TYPE_JSON = "application/json"
JSON_CODE_BLOCK_MARKER = "```json"

# Custom exceptions for better error handling
class AuthenticationError(Exception):
    """Raised when authentication with SAP AI Core fails."""
    pass

class TokenError(Exception):
    """Raised when token-related operations fail."""
    pass

class APIError(Exception):
    """Raised when API calls to AI services fail."""
    pass

# Set up logging
logger = logging.getLogger(__name__)


class SapAiCore:
    """Centralized service for handling API calls to different AI models.
    
    This class provides a unified interface for interacting with SAP AI Core services,
    specifically designed for code duplication detection workflows. It handles authentication,
    token management, and provides methods for generating embeddings and evaluating code similarity.
    
    Key Features:
        - OpenAI/Azure OpenAI embedding generation for code vectorization
        - Claude-based semantic code similarity evaluation
        - Automatic OAuth2 token management with caching
        - Batch processing capabilities for efficient API usage
        - Robust error handling and retry mechanisms
    
    Attributes:
        client_id (Optional[str]): OAuth2 client identifier for authentication
        client_secret (Optional[str]): OAuth2 client secret for authentication
        auth_url (Optional[str]): Authentication endpoint URL
        deployment_url (Optional[str]): OpenAI deployment endpoint URL
        claude_deployment_url (Optional[str]): Claude deployment endpoint URL
        auth_token (Optional[str]): Current authentication token
        token_expires_at (Optional[float]): Token expiration timestamp
        token_buffer_seconds (int): Seconds before expiry to refresh token
        env_file (str): Path to token cache file
    """
    
    def __init__(self) -> None:
        """Initialize the SAP AI Core service with configuration from environment.
        
        Loads configuration from environment variables and sets up token management.
        Automatically attempts to load cached authentication tokens to avoid
        unnecessary API calls during initialization.
        
        Environment Variables Required:
            CLIENT_ID: OAuth2 client identifier
            CLIENT_SECRET: OAuth2 client secret
            AUTH_URL: Authentication endpoint URL
            DEPLOYMENT_URL: OpenAI/Azure OpenAI deployment endpoint
            CLAUDE_DEP_URL: Claude deployment endpoint URL
        """
        # OpenAI/Azure OpenAI configuration - loaded from environment variables
        self.client_id: Optional[str] = os.environ.get('CLIENT_ID')
        self.client_secret: Optional[str] = os.environ.get('CLIENT_SECRET')
        self.auth_url: Optional[str] = os.environ.get('AUTH_URL')
        self.auth_token: Optional[str] = None
        self.deployment_url: Optional[str] = os.environ.get('DEPLOYMENT_URL')
        self.claude_deployment_url: Optional[str] = os.environ.get('CLAUDE_DEP_URL')
        
        # Token management with automatic refresh
        self.token_expires_at: Optional[float] = None
        self.token_buffer_seconds: int = 300  # Refresh token 5 minutes before expiry
        
        # Token persistence - separate .env file for tokens (gitignored for security)
        self.env_file: str = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env.tokens')
        
        # Claude configuration (reserved for future use)
        self.claude_client: Optional[Any] = None
        
        # Load cached token on initialization to avoid unnecessary auth calls
        self._load_cached_token() 


    def get_auth_token(self) -> str:
        """Fetch a new authentication token from SAP AI Core.
        
        Performs OAuth2 client credentials flow to obtain an access token.
        The token is cached both in memory and to disk for efficient reuse.
        Implements comprehensive error handling for various failure scenarios.
        
        Returns:
            str: Valid authentication token for API requests
            
        Raises:
            Exception: If authentication fails due to:
                - Network connectivity issues
                - Invalid credentials
                - Malformed server response
                - Missing access_token in response
                - JSON parsing errors
        """
        logger.info("Requesting new access token from SAP AI Core")
        
        # OAuth2 client credentials payload as per RFC 6749
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        # Form-encoded content type required for OAuth2 token requests
        headers = {"content-type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(self.auth_url, data=payload, headers=headers)
            
            # Validate HTTP response status
            if response.status_code != 200:
                logger.error(f"Authentication failed with HTTP status {response.status_code}")
                raise AuthenticationError(f"Auth request failed with status {response.status_code}: {response.text}")
            
            # Ensure response body is not empty
            if not response.text.strip():
                raise AuthenticationError("Auth endpoint returned empty response")
            
            # Basic JSON format validation before parsing
            response_text = response.text.strip()
            if not (response_text.startswith('{') and response_text.endswith('}')):
                raise AuthenticationError(f"Auth endpoint returned non-JSON response: {response_text}")
            
            # Parse JSON response safely
            token_data = response.json()
            
            auth_token = token_data.get("access_token")
            if not auth_token:
                raise TokenError(f"No access_token in response: {token_data}")
            
            # Track token expiration (typically expires_in is in seconds)
            expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour if not provided
            self.token_expires_at = time.time() + expires_in
            
            # Store the token in the instance for persistence
            self.auth_token = auth_token
            
            # Cache the token to .env file for future sessions
            self._save_token_to_env(auth_token, self.token_expires_at)
            
            logger.info(f"Token successfully fetched and cached. Expires at: {time.ctime(self.token_expires_at)}")
            
            return auth_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during authentication: {e}")
            raise AuthenticationError(f"Network error during authentication: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Auth endpoint returned invalid JSON: {response.text}")
            raise AuthenticationError(f"Auth endpoint returned invalid JSON: {response.text}") from e
        except (AuthenticationError, TokenError):
            # Re-raise our custom exceptions without modification
            raise
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            raise AuthenticationError(f"Unexpected authentication error: {e}") from e

    def _save_token_to_env(self, token: str, expires_at: float) -> None:
        """Save authentication token to persistent cache file.
        
        Stores the authentication token, expiration time, and cache timestamp
        to a .env.tokens file for reuse across application sessions. The file
        is created with restrictive permissions (0o600) for security.
        
        Args:
            token (str): The authentication token to cache
            expires_at (float): Unix timestamp when the token expires
        """
        try:
            # Ensure the cache directory structure exists
            os.makedirs(os.path.dirname(self.env_file), exist_ok=True)
            
            # Store token data using python-dotenv for consistent format
            set_key(self.env_file, "CACHED_ACCESS_TOKEN", token)
            set_key(self.env_file, "TOKEN_EXPIRES_AT", str(expires_at))
            set_key(self.env_file, "TOKEN_CACHED_AT", str(time.time()))  # For debugging/monitoring
            
            # Set restrictive file permissions - readable/writable only by owner (security best practice)
            os.chmod(self.env_file, 0o600)
            
            logger.debug(f"Token successfully cached to {self.env_file}")
            
        except Exception:
            # Silent failure for caching - authentication should not fail due to cache issues
            logger.debug(f"Failed to cache token to {self.env_file}", exc_info=True)

    def _load_cached_token(self) -> bool:
        """Load and validate cached token from .env.tokens file.
        
        Checks if a previously cached authentication token is still valid
        and loads it for use, avoiding unnecessary API calls.
        
        Returns:
            True if a valid cached token was loaded, False otherwise
        """
        try:
            if not os.path.exists(self.env_file):
                return False
            
            # Load the .env.tokens file
            load_dotenv(self.env_file)
            
            cached_token = os.getenv("CACHED_ACCESS_TOKEN")
            expires_at_str = os.getenv("TOKEN_EXPIRES_AT")
            
            if not cached_token or not expires_at_str:
                return False
            
            cached_expires_at = float(expires_at_str)
            
            # Check if token is still valid (with buffer)
            current_time = time.time()
            if current_time >= (cached_expires_at - self.token_buffer_seconds):
                self._clear_token_cache()
                return False
            
            # Token is valid, use it
            self.auth_token = cached_token
            self.token_expires_at = cached_expires_at
            
            return True
            
        except Exception:
            self._clear_token_cache()
            return False

    def _clear_token_cache(self) -> None:
        """Remove cached token file and clear related environment variables.
        
        Performs complete cleanup of cached authentication data, including
        both the persistent file cache and any loaded environment variables.
        This method is used during token invalidation and error recovery.
        """
        try:
            # Remove the cache file if it exists
            if os.path.exists(self.env_file):
                os.remove(self.env_file)
                logger.debug(f"Removed token cache file: {self.env_file}")
            
            # Clear token-related variables from current environment
            # This prevents stale cached values from being used
            token_env_vars = ["CACHED_ACCESS_TOKEN", "TOKEN_EXPIRES_AT", "TOKEN_CACHED_AT"]
            for key in token_env_vars:
                if key in os.environ:
                    del os.environ[key]
                    
        except Exception:
            # Silent failure for cache cleanup - should not break authentication flow
            logger.debug(f"Failed to clear token cache from {self.env_file}", exc_info=True)

    def clear_token_cache(self) -> None:
        """Public method to clear cached token (useful for debugging/testing).
        
        Clears both in-memory and cached tokens, forcing fresh authentication
        on next API call. Useful for testing and troubleshooting.
        """
        self.auth_token = None
        self.token_expires_at = None
        self._clear_token_cache()

    def debug_token_status(self) -> Dict[str, Any]:
        """Retrieve comprehensive token status information for debugging.
        
        Provides detailed information about the current token state,
        including in-memory token data, cache file status, and expiration timing.
        Useful for troubleshooting authentication issues and monitoring token lifecycle.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - has_token (bool): Whether an in-memory token exists
                - token_preview (Optional[str]): First 10 characters of token (for identification)
                - expires_at (Optional[str]): Human-readable expiration timestamp
                - minutes_until_expiry (Optional[float]): Minutes until token expires
                - is_expired (bool): Whether the token is currently expired or expiring soon
                - cache_file_exists (bool): Whether the cache file exists on disk
                - cache_file_path (str): Path to the cache file
                - cache_info (Optional[Dict]): Cache file contents and metadata
        """
        cache_exists = os.path.exists(self.env_file)
        cache_info = None
        
        if cache_exists:
            try:
                load_dotenv(self.env_file)
                cache_info = {
                    "cached_token_exists": bool(os.getenv("CACHED_ACCESS_TOKEN")),
                    "cache_expires_at": time.ctime(float(os.getenv("TOKEN_EXPIRES_AT", "0"))),
                    "cache_created_at": time.ctime(float(os.getenv("TOKEN_CACHED_AT", "0"))),
                }
            except Exception:
                cache_info = {"error": "Failed to read cache"}
        
        return {
            "has_token": self.auth_token is not None,
            "token_preview": self.auth_token[:10] + "..." if self.auth_token else None,
            "expires_at": time.ctime(self.token_expires_at) if self.token_expires_at else None,
            "minutes_until_expiry": round((self.token_expires_at - time.time()) / 60, 1) if self.token_expires_at else None,
            "is_expired": self._is_token_expired(),
            "cache_file_exists": cache_exists,
            "cache_file_path": self.env_file,
            "cache_info": cache_info
        }
    
    def _is_token_expired(self) -> bool:
        """Check if the current authentication token is expired or expiring soon.
        
        Uses the configured buffer time to determine if a token should be
        considered expired before its actual expiration time. This prevents
        API calls with tokens that might expire during the request.
        
        Returns:
            bool: True if token is None, expired, or expiring within buffer time
        """
        if not self.token_expires_at:
            return True
        return time.time() >= (self.token_expires_at - self.token_buffer_seconds)
    
    def _ensure_valid_token(self) -> str:
        """Ensure a valid authentication token is available, refreshing if needed.
        
        Checks the current token status and automatically fetches a new token
        if none exists or if the current token is expired/expiring soon.
        This method provides automatic token management for API calls.
        
        Returns:
            str: Valid authentication token ready for API use
            
        Raises:
            Exception: If token acquisition fails (propagated from get_auth_token)
        """
        # Fetch new token if none exists or current token is expired/expiring
        if not self.auth_token or self._is_token_expired():
            self.auth_token = self.get_auth_token()
        
        return self.auth_token

    def generate_openai_embeddings(self, code_list: List[str]) -> np.ndarray:
        """Generate vector embeddings for code snippets using OpenAI/Azure OpenAI API.
        
        Takes a list of code strings and converts them into high-dimensional
        vector representations suitable for similarity calculations and clustering.
        Handles authentication automatically and includes retry logic for token expiration.
        
        Args:
            code_list (List[str]): List of code snippets to generate embeddings for.
                Each string should contain the code to be vectorized.
        
        Returns:
            np.ndarray: 2D numpy array where each row is an embedding vector.
                Shape: (len(code_list), embedding_dimension)
                Data type: float32 for memory efficiency
        
        Raises:
            requests.HTTPError: If the API request fails after token retry
            Exception: If token acquisition fails or response format is invalid
        """
        # Construct API endpoint for embeddings
        url = f"{self.deployment_url}/embeddings"
        
        # Use latest API version for OpenAI embeddings
        querystring = {"api-version": "2024-10-21"}
        payload = {"input": code_list}

        # Ensure we have a valid authentication token
        valid_token = self._ensure_valid_token()

        # Set up request headers with authentication and resource group
        headers = {
            "ai-resource-group": "default",  # SAP AI Core resource group
            "content-type": CONTENT_TYPE_JSON,
            "Authorization": f"Bearer {valid_token}"
        }
    
        response = requests.post(url, json=payload, headers=headers, params=querystring)
        
        # Handle potential token expiration with automatic retry
        if response.status_code == 401:
            # Force fresh token acquisition by clearing cached expiration
            self.token_expires_at = None
            valid_token = self._ensure_valid_token()
            headers["Authorization"] = f"Bearer {valid_token}"
            # Retry the request with fresh token
            response = requests.post(url, json=payload, headers=headers, params=querystring)
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Extract embeddings from API response and convert to numpy array
        # Stack embeddings vertically to create 2D array (samples x dimensions)
        embeddings = np.vstack([
            np.asarray(d.get('embedding'), dtype="float32") 
            for d in response.json().get('data')
        ])
        
        return embeddings
        
 
    async def evaluate_code_similarity_claude(self, query_code: str, candidate_code: str) -> Dict[str, Any]:
        """Evaluate semantic similarity between two code snippets using Claude AI.
        
        Uses Claude's natural language understanding to determine if two code
        snippets could be refactored into the same reusable UI library function.
        Focuses on semantic similarity rather than syntactic differences.
        
        Args:
            query_code (str): The reference code snippet to compare against
            candidate_code (str): The candidate code snippet for comparison
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - is_similar (bool): Whether the codes could use the same library function
                - confidence (float): Confidence score between 0.0 and 1.0
                - reasoning (str, optional): Explanation of the similarity assessment
        
        Raises:
            aiohttp.ClientError: If the HTTP request fails
            json.JSONDecodeError: If the response is not valid JSON
        """
        prompt = f"""
        You are a SAP UI5 refactoring expert. Your task is to determine if these two JavaScript code snippets could be replaced by the SAME reusable UI library function.

        KEY QUESTION: Could both code snippets be refactored to use the same utility function?

        LOOK FOR:
        1. **Similar business logic or problem-solving patterns** (even if implemented differently)
        2. **Common UI patterns** (navigation, data transformation, validation, etc.)
        3. **Same conceptual operations** (filtering, mapping, formatting, etc.)
        4. **Similar workflow steps** (get data → transform → display/navigate)

        EXAMPLES OF REFACTORABLE PAIRS:
        - One uses map(), other uses for-loop (both transforming data)
        - One uses if-else, other uses switch (both handling conditions)
        - Different navigation patterns but same routing logic
        - Different validation approaches but same validation rules

        IGNORE THESE DIFFERENCES:
        - Implementation method (map vs forEach vs for-loop)
        - Variable names and string literals
        - Parameter structures and object properties
        - Specific UI5 control types
        - Minor conditional variations

        Query Code:
        ```javascript
        {query_code}
        ```

        Candidate Code:
        ```javascript
        {candidate_code}
        ```

        Think: "Could I write ONE library function that handles both scenarios?"

        Respond with JSON only:
        {{
            "is_similar": boolean,
            "confidence": float (0.0-1.0),
        }}"""

        try:
            url = f"{self.claude_deployment_url}/converse"
            
            payload = {
         
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0.1
                }
            }

            # Ensure we have a valid token
            valid_token = self._ensure_valid_token()

            headers = {
                "ai-resource-group": "default",
                "content-type": CONTENT_TYPE_JSON,
                "Authorization": f"Bearer {valid_token}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    # If we get a 401, the token might be expired despite our checks
                    if response.status == 401:
                        # Force a fresh token by clearing expiration time
                        self.token_expires_at = None
                        valid_token = self._ensure_valid_token()
                        headers["Authorization"] = f"Bearer {valid_token}"
                        # Retry with fresh token
                        async with session.post(url, json=payload, headers=headers) as retry_response:
                            retry_response.raise_for_status()
                            response_data = await retry_response.json()
                    else:
                        response.raise_for_status()
                        response_data = await response.json()
            
            # Extract content from Converse API response
            content = response_data['output']['message']['content'][0]['text'].strip()

            # print(f"=====CLAUDE RESPONSE: {content}======")
            
            # Extract JSON from response
            if JSON_CODE_BLOCK_MARKER in content:
                json_start = content.find(JSON_CODE_BLOCK_MARKER) + len(JSON_CODE_BLOCK_MARKER)
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "{" in content and "}" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                content = content[json_start:json_end]
            
            return json.loads(content)
            
        except Exception:
            logger.warning("Claude similarity evaluation failed, returning conservative result", exc_info=True)
            return {"is_similar": True, "confidence": 0.5, "reasoning": "API error - keeping match"}
    
    def evaluate_code_similarity_claude_batch(self, query_candidate_pairs: List[Tuple[str, str]], model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0") -> List[Dict[str, Any]]:
        """Evaluate multiple code similarity pairs in a single API call for efficiency.
        
        Processes multiple query-candidate code pairs in a single request to Claude,
        significantly reducing API calls and improving performance for batch operations.
        Each pair is evaluated independently with consistent scoring criteria.
        
        Args:
            query_candidate_pairs (List[Tuple[str, str]]): List of (query_code, candidate_code) tuples
                to evaluate. Each tuple contains two code snippets for comparison.
            model (str, optional): Claude model identifier. Defaults to Claude 3.5 Sonnet.
        
        Returns:
            List[Dict[str, Any]]: List of evaluation results, one per input pair.
                Each dictionary contains:
                - pair_id (int): Sequential ID (1-based) matching input order
                - is_similar (bool): Whether codes could use same library function
                - confidence (float): Confidence score between 0.0 and 1.0
                - reasoning (str): Explanation of the similarity assessment
        
        Raises:
            requests.HTTPError: If the API request fails after token retry
            json.JSONDecodeError: If the response cannot be parsed as JSON
        """
        # Return empty list for empty input
        if not query_candidate_pairs:
            return []
        
        # Build formatted sections for each code pair in the batch prompt
        batch_sections = []
        for idx, (query_code, candidate_code) in enumerate(query_candidate_pairs, 1):
            batch_sections.append(f"""
            Pair {idx}:
            Query Code:
            ```javascript
            {query_code}
            ```

            Candidate Code:
            ```javascript
            {candidate_code}
            ```""")
                    
        # Construct comprehensive prompt for batch evaluation
        prompt = f"""You are a SAP UI5 refactoring expert. For each pair below, determine if both code snippets could be refactored to use the same reusable UI library function.

            KEY QUESTION: Could both code snippets be replaced by the SAME utility function?

            LOOK FOR:
            1. **Similar business logic or problem-solving patterns** (even if implemented differently)
            2. **Common UI patterns** (navigation, data transformation, validation, etc.)
            3. **Same conceptual operations** (filtering, mapping, formatting, etc.)
            4. **Similar workflow steps** (get data → transform → display/navigate)

            EXAMPLES OF REFACTORABLE PAIRS:
            - One uses map(), other uses for-loop (both transforming data)
            - One uses if-else, other uses switch (both handling conditions)
            - Different navigation patterns but same routing logic
            - Different validation approaches but same validation rules

            IGNORE THESE DIFFERENCES:
            - Implementation method (map vs forEach vs for-loop)
            - Variable names and string literals
            - Parameter structures and object properties
            - Specific UI5 control types
            - Minor conditional variations

            {chr(10).join(batch_sections)}

            Respond with JSON array only (no other text):
            [
                {{"pair_id": 1, "is_similar": boolean, "confidence": float (0.0-1.0), "reasoning": "Explain reusable UI library function"}},
                {{"pair_id": 2, "is_similar": boolean, "confidence": float (0.0-1.0), "reasoning": "Explain reusable UI library function"}},
                {{"pair_id": 3, "is_similar": boolean, "confidence": float (0.0-1.0), "reasoning": "Explain reusable UI library function"}}
            ]"""

        try:
            # Use Bedrock Converse API endpoint
            url = f"{self.claude_deployment_url}/converse"
            
            # Converse API payload format
            payload = {
                "modelId": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 1000,  # Increased for batch responses
                    "temperature": 0.1
                }
            }
            
            # Ensure we have a valid token
            valid_token = self._ensure_valid_token()
            
            headers = {
                "ai-resource-group": "default",
                "content-type": "application/json",
                "Authorization": f"Bearer {valid_token}"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            # If we get a 401, the token might be expired despite our checks
            if response.status_code == 401:
                # Force a fresh token by clearing expiration time
                self.token_expires_at = None
                valid_token = self._ensure_valid_token()
                headers["Authorization"] = f"Bearer {valid_token}"
                response = requests.post(url, json=payload, headers=headers)
            
            response.raise_for_status()  # Raise exception for other HTTP errors
            response_data = response.json()
            
            # Extract content from Converse API response
            content = response_data['output']['message']['content'][0]['text'].strip()
            
            # Extract JSON array from response
            if JSON_CODE_BLOCK_MARKER in content:
                json_start = content.find(JSON_CODE_BLOCK_MARKER) + len(JSON_CODE_BLOCK_MARKER)
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "[" in content and "]" in content:
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                content = content[json_start:json_end]
            
            batch_results = json.loads(content)
            
            # Ensure we have results for all pairs
            if len(batch_results) != len(query_candidate_pairs):
                # Pad with default results if needed
                while len(batch_results) < len(query_candidate_pairs):
                    batch_results.append({
                        "pair_id": len(batch_results) + 1,
                        "is_similar": False,
                        "confidence": 0.0,
                        "reasoning": "Missing result from batch response"
                    })
            
            return batch_results
            
        except Exception:
            # Fallback to individual error responses
            logger.warning("Claude batch evaluation failed, returning conservative results", exc_info=True)
            return [
                {"is_similar": True, "confidence": 0.5, "reasoning": "API error - keeping match"}
                for _ in query_candidate_pairs
            ]

    def generate_embeddings(self, code_blocks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Generate embeddings for a list of code block objects.
        
        Convenience method that extracts processed code from code block objects
        and generates embeddings using the OpenAI API. Handles errors gracefully
        with logging for debugging purposes.
        
        Args:
            code_blocks (List[Dict[str, Any]]): List of code block dictionaries,
                each expected to have a 'processedCode' key with the code string
        
        Returns:
            Optional[np.ndarray]: 2D numpy array of embeddings if successful,
                None if an error occurs during processing
        """
        try:
            # Extract processed code strings from code block objects
            code_list = [code_block.get('processedCode') for code_block in code_blocks]
            embeddings = self.generate_openai_embeddings(code_list)
            return embeddings
        except Exception:
            # Log error but don't raise to prevent pipeline failures
            logger.error('Error generating embeddings for code blocks', exc_info=True)
            return None
