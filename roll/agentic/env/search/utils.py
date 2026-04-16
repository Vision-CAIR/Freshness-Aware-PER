import re
import json
import requests
import time
from typing import Dict, Any, Callable, Optional


def parse_action_content(action: str, config) -> Dict[str, Any]:
    """
    Parse action content to extract think, code, and answer components.
    
    Args:
        action: The action string from the model
        config: SearchEnvConfig object
        
    Returns:
        Dictionary with parsed components
    """
    result = {
        "is_valid": False,
        "think": "",
        "code": "",
        "answer": "",
        "raw_action": action
    }
    
    # Extract think content
    think_match = re.search(config.think_pattern, action, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()
    
    # Extract code content
    code_match = re.search(config.code_pattern, action, re.DOTALL)
    if code_match:
        result["code"] = code_match.group(1).strip()
    
    # Extract answer content
    answer_match = re.search(config.answer_pattern, action, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    # Action is valid if it contains at least one of: think, code, or answer
    result["is_valid"] = bool(result["think"] or result["code"] or result["answer"])
    
    return result


def execute_search_code(code: str, config, increment_search_count: Callable) -> str:
    """
    Execute search code using remote HTTP service or mock functions.

    Args:
        code: Python code to execute
        config: SearchEnvConfig object
        increment_search_count: Callback to increment search count

    Returns:
        Execution result as string
    """

    # Check if we should use remote HTTP service
    if hasattr(config, 'use_remote_service') and config.use_remote_service:
        return execute_code_via_http(code, config, increment_search_count)

    # Otherwise use local execution with mock or real APIs
    def web_search(keywords: str) -> str:
        """Search the web with given keywords"""
        increment_search_count()

        # Check if we should use mock mode (for testing)
        if hasattr(config, 'use_mock_api') and config.use_mock_api:
            return mock_web_search(keywords)

        try:
            # Real API implementation
            response = requests.post(
                config.search_api_url,
                json={"query": keywords},
                headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
                timeout=30
            )
            if response.status_code == 200:
                return response.text
            else:
                return f"Search API error: {response.status_code}"
        except Exception as e:
            # Fallback to mock if real API fails
            return mock_web_search(keywords)

    def web_parse(link: str, query: str) -> str:
        """Parse web page content"""
        increment_search_count()

        # Check if we should use mock mode (for testing)
        if hasattr(config, 'use_mock_api') and config.use_mock_api:
            return mock_web_parse(link, query)

        try:
            # Real API implementation
            response = requests.post(
                config.parse_api_url,
                json={"url": link, "query": query},
                headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
                timeout=30
            )
            if response.status_code == 200:
                return response.text
            else:
                return f"Parse API error: {response.status_code}"
        except Exception as e:
            # Fallback to mock if real API fails
            return mock_web_parse(link, query)

    def parse_img(link: str, query: str) -> str:
        """Parse image content"""
        increment_search_count()

        # Check if we should use mock mode (for testing)
        if hasattr(config, 'use_mock_api') and config.use_mock_api:
            return mock_parse_img(link, query)

        try:
            # Real API implementation
            response = requests.post(
                config.img_parse_api_url,
                json={"image_url": link, "query": query},
                headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
                timeout=30
            )
            if response.status_code == 200:
                return response.text
            else:
                return f"Image parse API error: {response.status_code}"
        except Exception as e:
            # Fallback to mock if real API fails
            return mock_parse_img(link, query)
    
    # Create execution environment with available functions
    exec_globals = {
        "web_search": web_search,
        "web_parse": web_parse,
        "parse_img": parse_img,
        "print": lambda *args: "\n".join(str(arg) for arg in args),
        "__builtins__": {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "max": max,
            "min": min,
            "sum": sum,
            "abs": abs,
            "round": round,
        }
    }
    
    exec_locals = {}
    output_lines = []
    
    # Capture print output
    def capture_print(*args):
        output_lines.append(" ".join(str(arg) for arg in args))
    
    exec_globals["print"] = capture_print
    
    try:
        # Execute the code
        exec(code, exec_globals, exec_locals)
        
        # Return captured output
        if output_lines:
            return "\n".join(output_lines)
        else:
            return "Code executed successfully (no output)"
            
    except Exception as e:
        return f"Execution error: {str(e)}"


def mock_web_search(keywords: str) -> str:
    """Mock web search function for testing"""
    # Provide more realistic mock responses based on keywords
    mock_responses = {
        "railway yard paddington": """
Search Results:
1. Old Oak Common Yard - Wikipedia
   Old Oak Common Yard is a railway yard on the Great Western Railway near Paddington, London.
   It features a single slip switch installation that allows trains to access sidings without facing points.

2. Great Western Railway Infrastructure
   The yard is used for safety and reversing maneuvers, with specialized track configurations.

3. London Railway Yards Guide
   Comprehensive guide to railway infrastructure around Paddington station.
        """,
        "census 2020 miscount": """
Search Results:
1. 2020 U.S. Census Post-Enumeration Survey Results
   The Census Bureau's post-enumeration survey revealed significant miscounts in several states.
   Hawaii had the highest percentage miscount at 6.6%.

2. Census Bureau Official Report
   Detailed analysis of counting accuracy by state and demographic groups.
        """,
        "paul brainerd foundation": """
Search Results:
1. Brainerd Foundation - Official Website
   The Brainerd Foundation was established by Paul Brainerd after selling Aldus Corporation to Adobe in 1994.
   Focuses on environmental conservation and supporting nonprofit organizations in the Pacific Northwest.

2. Paul Brainerd Biography
   Co-founder of Aldus Corporation, creator of PageMaker software.
        """
    }

    # Find the best matching response
    keywords_lower = keywords.lower()
    for key, response in mock_responses.items():
        if any(word in keywords_lower for word in key.split()):
            return response

    # Default response
    return f"""
Search Results for: {keywords}

1. Example Result 1
   Relevant information about {keywords} can be found here.

2. Example Result 2
   Additional context and details related to your search query.

3. Example Result 3
   More comprehensive information on the topic.
    """


def mock_web_parse(link: str, query: str) -> str:
    """Mock web parse function for testing"""
    return f"""
Parsed content from {link}:

Title: Relevant Article Title
Content: This is the parsed content that would be extracted from the webpage.
The content is relevant to your query: {query}

Key information:
- Important fact 1
- Important fact 2
- Important fact 3

Additional details and context would be provided here.
    """


def mock_parse_img(link: str, query: str) -> str:
    """Mock image parse function for testing"""
    return f"""
Image analysis for {link}:

Query: {query}

Visual Description:
- The image shows relevant visual elements
- Key features are clearly visible
- Text content (if any): [extracted text]

Analysis Results:
- The image contains information relevant to your query
- Specific details that answer your question
- Additional context from visual elements
    """


def execute_code_via_http(code: str, config, increment_search_count: Callable) -> str:
    """
    Execute code via remote HTTP service (like the tool server).

    Args:
        code: Python code to execute
        config: SearchEnvConfig object
        increment_search_count: Callback to increment search count

    Returns:
        Execution result as string
    """
    increment_search_count()

    try:
        start_time = time.time()

        # Prepare code with tools import prefix
        full_code = f"from tools import *\n\n{code}"

        # Disable proxy and create session
        session = requests.Session()
        session.proxies = {'http': None, 'https': None}

        # Prepare payload
        payload = {"code": full_code}

        # Make request to remote service
        resp = session.post(
            f"{config.remote_service_url}/execute",
            json=payload,
            timeout=config.remote_service_timeout
        )

        # Check response status
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}: {resp.text}"

        # Check response content
        if not resp.text.strip():
            return "Empty response from remote service"

        try:
            response = resp.json()
        except json.JSONDecodeError as e:
            return f"JSON decode error: {e}, Response: '{resp.text[:100]}'"

        # Extract result
        if response.get('error'):
            return f"Execution error: {response['error']}"

        output = response.get('output', '')
        elapsed = time.time() - start_time

        # Return the output with timing info
        return f"{output}\n\n[Execution completed in {elapsed:.3f}s]"

    except requests.exceptions.Timeout:
        return "Request timeout - remote service took too long to respond"
    except requests.exceptions.ConnectionError:
        return "Connection error - could not reach remote service"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
