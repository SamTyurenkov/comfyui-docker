#from https://github.com/runpod-workers/worker-comfyui/blob/main/handler.py
import runpod
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import subprocess

# Import Firebase credits module for server-side credit deduction
try:
    from firebase_credits import (
        verify_user_token,
        get_user_credits,
        check_sufficient_credits,
        deduct_credits,
        _firebase_initialized as firebase_available
    )
    FIREBASE_CREDITS_AVAILABLE = True
    print("worker-comfyui - Firebase credits module loaded successfully")
except ImportError as e:
    print(f"worker-comfyui - Firebase credits module not available: {e}")
    FIREBASE_CREDITS_AVAILABLE = False

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 5000
# Websocket reconnection behaviour (can be overridden through environment variables)
# NOTE: more attempts and diagnostics improve debuggability whenever ComfyUI crashes mid-job.
#   • WEBSOCKET_RECONNECT_ATTEMPTS sets how many times we will try to reconnect.
#   • WEBSOCKET_RECONNECT_DELAY_S sets the sleep in seconds between attempts.
#
# If the respective env-vars are not supplied we fall back to sensible defaults ("5" and "3").
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    # This prints low-level frame information to stdout which is invaluable for diagnosing
    # protocol errors but can be noisy in production – therefore gated behind an env-var.
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# GPU pricing configuration (credits per second)
GPU_PRICING = {
    "RTX 3070": 0.03,
    "RTX 3080": 0.03,
    "RTX 3090": 0.03,
    "RTX 4070": 0.04,
    "RTX 4080": 0.04,
    "RTX 4090": 0.04,
    "RTX 5090": 0.06,
    "A4000": 0.04,
    "A5000": 0.05,
    "A6000": 0.06,
    "RTX A6000": 0.05,
    "RTX PRO 6000": 0.09,  # Professional cards
    "A40": 0.05,
    "A100": 0.06,
    "H100": 0.1,
    "default": 0.10,  # fallback pricing
}

# Base cost per job (to prevent abuse and cover overhead)
BASE_COST_PER_JOB = 1.0

# Cache for GPU type (to avoid calling nvidia-smi on every job)
_GPU_TYPE_CACHE = None

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


def detect_gpu_type():
    """
    Detect the GPU type using nvidia-smi. Results are cached to avoid repeated calls.
    
    Returns:
        str: GPU type (e.g., "RTX 4090", "A100") or "unknown" if detection fails
    """
    global _GPU_TYPE_CACHE
    
    # Return cached value if available
    if _GPU_TYPE_CACHE is not None:
        return _GPU_TYPE_CACHE
    
    try:
        # Run nvidia-smi to get GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout:
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU
            print(f"worker-comfyui - Detected GPU (raw): {gpu_name}")
            
            # Normalize GPU name to match our pricing keys
            # Remove common prefixes but preserve important parts
            normalized_name = gpu_name.replace("NVIDIA ", "")
            normalized_name = normalized_name.replace("GeForce ", "")
            normalized_name = normalized_name.strip()
            
            # Remove trailing descriptors like "Black", "Ada", etc. but keep important parts
            # Split and remove common suffixes
            parts = normalized_name.split()
            # Keep relevant parts (RTX, PRO, numbers, letters like A100, etc.)
            filtered_parts = []
            for part in parts:
                # Keep if it's a known keyword or starts with a number or is short
                if part in ["RTX", "PRO", "Ti", "SUPER"] or part[0].isdigit() or part in ["A100", "A40", "A6000", "A5000", "A4000", "H100"]:
                    filtered_parts.append(part)
                # Also keep if it starts with 'A' and has numbers (like A6000)
                elif part.startswith('A') and any(c.isdigit() for c in part):
                    filtered_parts.append(part)
            
            normalized_name = " ".join(filtered_parts) if filtered_parts else normalized_name
            
            print(f"worker-comfyui - Normalized GPU name: {normalized_name}")
            
            # Check if the GPU name matches any of our pricing keys
            # First try exact match
            if normalized_name in GPU_PRICING:
                print(f"worker-comfyui - Exact match found in pricing table")
                _GPU_TYPE_CACHE = normalized_name
                return normalized_name
            
            # Try partial matches for common GPUs (check if pricing key is in GPU name)
            for gpu_key in GPU_PRICING.keys():
                if gpu_key != "default" and gpu_key in normalized_name:
                    print(f"worker-comfyui - Matched GPU '{normalized_name}' to pricing key '{gpu_key}'")
                    _GPU_TYPE_CACHE = gpu_key
                    return gpu_key
            
            # If no match found, log the GPU name and use default
            print(f"worker-comfyui - GPU '{normalized_name}' not in pricing table, using default pricing")
            _GPU_TYPE_CACHE = normalized_name  # Cache the actual name even if not in pricing table
            return normalized_name
        else:
            print(f"worker-comfyui - nvidia-smi failed with return code {result.returncode}")
            _GPU_TYPE_CACHE = "unknown"
            return "unknown"
            
    except subprocess.TimeoutExpired:
        print("worker-comfyui - nvidia-smi timed out")
        _GPU_TYPE_CACHE = "unknown"
        return "unknown"
    except FileNotFoundError:
        print("worker-comfyui - nvidia-smi not found")
        _GPU_TYPE_CACHE = "unknown"
        return "unknown"
    except Exception as e:
        print(f"worker-comfyui - Error detecting GPU: {e}")
        _GPU_TYPE_CACHE = "unknown"
        return "unknown"


def calculate_job_cost(execution_time_ms, gpu_type=None):
    """
    Calculate the cost of a job based on GPU type and execution time.
    
    Args:
        execution_time_ms (int): Execution time in milliseconds
        gpu_type (str, optional): GPU type identifier. If None, auto-detects using nvidia-smi.
        
    Returns:
        dict: Dictionary containing cost breakdown with the following keys:
            - total_cost: Total credits to charge (float)
            - base_cost: Base cost per job (float)
            - execution_cost: Cost based on execution time (float)
            - execution_time_sec: Execution time in seconds (float)
            - gpu_type: GPU type used (str)
            - rate_per_second: Rate per second for this GPU (float)
    """
    execution_time_sec = execution_time_ms / 1000.0
    
    # Detect GPU type if not provided
    if gpu_type is None:
        gpu_type = detect_gpu_type()
    
    # Get pricing rate for this GPU type
    rate_per_second = GPU_PRICING.get(gpu_type, GPU_PRICING["default"])
    
    # Calculate execution cost
    execution_cost = execution_time_sec * rate_per_second
    
    # Total cost = base cost + execution cost
    total_cost = BASE_COST_PER_JOB + execution_cost
    
    # Round to 2 decimal places
    total_cost = round(total_cost, 2)
    execution_cost = round(execution_cost, 2)
    execution_time_sec = round(execution_time_sec, 2)
    
    return {
        "total_cost": total_cost,
        "base_cost": BASE_COST_PER_JOB,
        "execution_cost": execution_cost,
        "execution_time_sec": execution_time_sec,
        "execution_time_ms": execution_time_ms,
        "gpu_type": gpu_type,
        "rate_per_second": rate_per_second,
    }


def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url (str): The WebSocket URL (including client_id).
        max_attempts (int): Maximum number of reconnection attempts.
        delay_s (int): Delay in seconds between attempts.
        initial_error (Exception): The error that triggered the reconnect attempt.

    Returns:
        websocket.WebSocket: The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(
        f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect..."
    )
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        # Log current server status before each reconnect attempt so that we can
        # see whether ComfyUI is still alive (HTTP port 8188 responding) even if
        # the websocket dropped. This is extremely useful to differentiate
        # between a network glitch and an outright ComfyUI crash/OOM-kill.
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            # If ComfyUI itself is down there is no point in retrying the websocket –
            # bail out immediately so the caller gets a clear "ComfyUI crashed" error.
            print(
                f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: {srv_status.get('error', 'status '+str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )

        # Otherwise we proceed with reconnect attempts while server is up
        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            # Need to create a new socket object for reconnect
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)  # Use existing ws_url
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws  # Return the new connected socket
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as reconn_err:
            last_reconnect_error = reconn_err
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}"
            )
            if attempt < max_attempts - 1:
                print(
                    f"worker-comfyui - Waiting {delay_s} seconds before next attempt..."
                )
                time.sleep(delay_s)
            else:
                print(f"worker-comfyui - Max reconnection attempts reached.")

    # If loop completes without returning, raise an exception
    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list):
            return None, "'images' must be a list"
        
        # Validate each image object
        for i, image in enumerate(images):
            if not isinstance(image, dict):
                return None, f"Image at index {i} must be an object"
            
            # Check for required fields
            if "name" not in image:
                return None, f"Image at index {i} missing 'name' field"
            
            if "image" not in image:
                return None, f"Image at index {i} missing 'image' field"
            
            # Validate image data format
            image_data = image["image"]
            if not isinstance(image_data, str):
                return None, f"Image data at index {i} must be a string"
            
            # Check if it's a valid base64 or data URL (support both image and video)
            if "," in image_data:
                # Data URL format: data:image/png;base64,<base64_data> or data:video/mp4;base64,<base64_data>
                parts = image_data.split(",", 1)
                if len(parts) != 2 or not (parts[0].startswith("data:image/") or parts[0].startswith("data:video/")):
                    return None, f"Invalid data URL format at index {i}. Must be data:image/* or data:video/*"
                base64_data = parts[1]
            else:
                # Pure base64 format
                base64_data = image_data
            
            # Validate base64 data
            try:
                base64.b64decode(base64_data)
            except Exception:
                return None, f"Invalid base64 data at index {i}"

    # Optional: user authentication for credit system
    # The frontend can send a Firebase ID token for server-side credit deduction
    user_token = job_input.get("user_token")
    workflow_id = job_input.get("workflow_id")

    # Return validated data and no error
    return {
        "workflow": workflow,
        "images": images,
        "user_token": user_token,
        "workflow_id": workflow_id
    }, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException as e:
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images/videos to the ComfyUI server using the appropriate upload endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the media file and the 'image' as a base64 encoded string (can be image or video).

    Returns:
        dict: A dictionary indicating success or error with detailed information about uploaded media files.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": [], "uploaded_images": []}

    responses = []
    upload_errors = []
    uploaded_images = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for i, image in enumerate(images):
        try:
            name = image["name"]
            image_data_uri = image["image"]  # Get the full string (might have prefix)

            print(f"worker-comfyui - Processing image {i+1}/{len(images)}: {name}")

            # --- Strip Data URI prefix if present and detect media type ---
            media_type = "image/png"  # default
            if "," in image_data_uri:
                # Find the comma and take everything after it
                prefix, base64_data = image_data_uri.split(",", 1)
                # Extract media type from data URL prefix
                if prefix.startswith("data:video/"):
                    if "mp4" in prefix:
                        media_type = "video/mp4"
                    elif "webm" in prefix:
                        media_type = "video/webm"
                    else:
                        media_type = "video/mp4"  # default video type
                elif prefix.startswith("data:image/"):
                    if "jpeg" in prefix or "jpg" in prefix:
                        media_type = "image/jpeg"
                    elif "png" in prefix:
                        media_type = "image/png"
                    else:
                        media_type = "image/png"  # default image type
                print(f"worker-comfyui - Extracted base64 data from data URL for {name}, detected type: {media_type}")
            else:
                # Assume it's already pure base64
                base64_data = image_data_uri
                print(f"worker-comfyui - Using pure base64 data for {name}")
            # --- End strip ---

            blob = base64.b64decode(base64_data)  # Decode the cleaned data
            print(f"worker-comfyui - Decoded {len(blob)} bytes for {name}")

            # Use the same upload endpoint for both images and videos
            endpoint = f"http://{COMFY_HOST}/upload/image"
            
            # ComfyUI uses 'image' form field for all media types
            files = {
                "image": (name, BytesIO(blob), media_type),
                "overwrite": (None, "true"),
            }

            # POST request to upload the media file
            response = requests.post(endpoint, files=files, timeout=30)
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            uploaded_images.append({
                "name": name,
                "original_index": i,
                "uploaded": True,
                "size_bytes": len(blob)
            })
            print(f"worker-comfyui - Successfully uploaded {name} ({len(blob)} bytes)")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
            uploaded_images.append({
                "name": image.get('name', f'image_{i}'),
                "original_index": i,
                "uploaded": False,
                "error": error_msg
            })
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
            uploaded_images.append({
                "name": image.get('name', f'image_{i}'),
                "original_index": i,
                "uploaded": False,
                "error": error_msg
            })
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
            uploaded_images.append({
                "name": image.get('name', f'image_{i}'),
                "original_index": i,
                "uploaded": False,
                "error": error_msg
            })
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
            uploaded_images.append({
                "name": image.get('name', f'image_{i}'),
                "original_index": i,
                "uploaded": False,
                "error": error_msg
            })

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
            "uploaded_images": uploaded_images
        }

    print(f"worker-comfyui - image(s) upload complete - {len(uploaded_images)} images uploaded successfully")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
        "uploaded_images": uploaded_images
    }


def get_uploaded_images_info(upload_result):
    """
    Extract information about successfully uploaded images for workflow integration.
    
    Args:
        upload_result (dict): The result from upload_images function
        
    Returns:
        dict: Information about uploaded images organized by name and index
    """
    if not upload_result or upload_result.get("status") != "success":
        return {}
    
    uploaded_images = upload_result.get("uploaded_images", [])
    images_info = {
        "by_name": {},
        "by_index": {},
        "successful_count": 0,
        "total_count": len(uploaded_images)
    }
    
    for img_info in uploaded_images:
        if img_info.get("uploaded", False):
            name = img_info["name"]
            index = img_info["original_index"]
            
            images_info["by_name"][name] = {
                "index": index,
                "size_bytes": img_info.get("size_bytes", 0),
                "uploaded": True
            }
            images_info["by_index"][index] = {
                "name": name,
                "size_bytes": img_info.get("size_bytes", 0),
                "uploaded": True
            }
            images_info["successful_count"] += 1
    
    return images_info


def log_workflow_image_usage(workflow, images_info):
    """
    Log information about how uploaded images might be used in the workflow.
    
    Args:
        workflow (dict): The workflow object
        images_info (dict): Information about uploaded images
    """
    if not images_info or images_info["successful_count"] == 0:
        print("worker-comfyui - No uploaded images to analyze in workflow")
        return
    
    print(f"worker-comfyui - Analyzing workflow for {images_info['successful_count']} uploaded images...")
    
    # Look for nodes that might use uploaded images
    image_nodes = []
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Check for common image input nodes
        if node_class in ["LoadImage", "ImageLoader", "LoadImageFromUrl"]:
            image_nodes.append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        elif "image" in inputs or "filename" in inputs:
            # Generic check for any node with image-related inputs
            image_nodes.append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
    
    if image_nodes:
        print(f"worker-comfyui - Found {len(image_nodes)} potential image input nodes:")
        for node in image_nodes:
            print(f"worker-comfyui -   - Node {node['node_id']} ({node['class_type']})")
            for key, value in node['inputs'].items():
                if key in ["image", "filename", "url"]:
                    print(f"worker-comfyui -     {key}: {value}")
    else:
        print("worker-comfyui - No obvious image input nodes found in workflow")


def get_available_models():
    """
    Get list of available models from ComfyUI

    Returns:
        dict: Dictionary containing available models by type
    """
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        # Extract available checkpoints from CheckpointLoaderSimple
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow, client_id):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed
        client_id (str): The client ID for the websocket connection

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow

    Raises:
        ValueError: If the workflow validation fails with detailed error information
    """
    # Include client_id in the prompt payload
    payload = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")

    # Use requests for consistency and timeout
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30
    )

    # Handle validation errors with detailed information
    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
            print(f"worker-comfyui - Parsed error data: {error_data}")

            # Try to extract meaningful error information
            error_message = "Workflow validation failed"
            error_details = []

            # ComfyUI seems to return different error formats, let's handle them all
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                    if error_info.get("type") == "prompt_outputs_failed_validation":
                        error_message = "Workflow validation failed"
                else:
                    error_message = str(error_info)

            # Check for node validation errors in the response
            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(
                                f"Node {node_id} ({error_type}): {error_msg}"
                            )
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            # Check if the error data itself contains validation info
            if error_data.get("type") == "prompt_outputs_failed_validation":
                error_message = error_data.get("message", "Workflow validation failed")
                # For this type of error, we need to parse the validation details from logs
                # Since ComfyUI doesn't seem to include detailed validation errors in the response
                # Let's provide a more helpful generic message
                available_models = get_available_models()
                if available_models.get("checkpoints"):
                    error_message += f"\n\nThis usually means a required model or parameter is not available."
                    error_message += f"\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                else:
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += "\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(error_message)

            # If we have specific validation errors, format them nicely
            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(
                    f"• {detail}" for detail in error_details
                )

                # Try to provide helpful suggestions for common errors
                if any(
                    "not in list" in detail and "ckpt_name" in detail
                    for detail in error_details
                ):
                    available_models = get_available_models()
                    if available_models.get("checkpoints"):
                        detailed_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                    else:
                        detailed_message += "\n\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(detailed_message)
            else:
                # Fallback to the raw response if we can't parse specific errors
                raise ValueError(f"{error_message}. Raw response: {response.text}")

        except (json.JSONDecodeError, KeyError) as e:
            # If we can't parse the error response, fall back to the raw text
            raise ValueError(
                f"ComfyUI validation failed (could not parse error response): {response.text}"
            )

    # For other HTTP errors, raise them normally
    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    # Use requests for consistency and timeout
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_data(filename, subfolder, image_type):
    """
    Fetch image bytes from the ComfyUI /view endpoint.

    Args:
        filename (str): The filename of the image.
        subfolder (str): The subfolder where the image is stored.
        image_type (str): The type of the image (e.g., 'output').

    Returns:
        bytes: The raw image data, or None if an error occurs.
    """
    print(
        f"worker-comfyui - Fetching image data: type={image_type}, subfolder={subfolder}, filename={filename}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        # Use requests for consistency and timeout
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched image data for {filename}")
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(
            f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}"
        )
        return None


def extract_base64_images_from_workflow(workflow, unique_id=None):
    """
    Extract base64 image data from workflow nodes and prepare them for upload.
    
    Args:
        workflow (dict): The workflow object
        unique_id (str, optional): Unique identifier to prevent filename conflicts in concurrent requests
        
    Returns:
        tuple: (extracted_images, cleaned_workflow, error_message)
               extracted_images: list of image objects with name and base64 data
               cleaned_workflow: workflow with base64 data replaced with placeholder filenames
               error_message: None if successful, error string if failed
    """
    extracted_images = []
    cleaned_workflow = workflow.copy()
    
    print("worker-comfyui - Scanning workflow for base64 image data...")
    
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Look for LoadImage nodes, video nodes, or other media input nodes
        is_video_node = node_class in ["VHS_LoadVideoFFmpeg", "VHS_LoadVideo", "LoadVideo"] or "video" in inputs
        is_image_node = node_class in ["LoadImage", "ImageLoader"] or "image" in inputs
        
        if is_image_node or is_video_node:
            for input_key, input_value in inputs.items():
                if (input_key in ["image", "video"]) and isinstance(input_value, str):
                    # Check if this looks like base64 data or data URL
                    is_base64 = False
                    image_data = input_value
                    
                    # Handle data URL format: data:image/png;base64,<base64_data> or data:video/mp4;base64,<base64_data>
                    if input_value.startswith('data:image/') or input_value.startswith('data:video/'):
                        if ',' in input_value:
                            image_data = input_value.split(',', 1)[1]
                            is_base64 = True
                            media_type = "video" if input_value.startswith('data:video/') else "image"
                            print(f"worker-comfyui - Found {media_type} data URL in node {node_id}, input '{input_key}'")
                    # Handle pure base64 (long string, starts with common base64 chars)
                    elif len(input_value) > 100 and input_value.startswith(('iVBORw0KGgo', '/9j/', 'UklGR')):
                        is_base64 = True
                        print(f"worker-comfyui - Found base64 image data in node {node_id}, input '{input_key}'")
                    
                    if is_base64:
                        # Generate a unique filename for this media file
                        # Use unique_id if provided, otherwise generate a UUID to prevent conflicts
                        if unique_id:
                            unique_suffix = unique_id
                        else:
                            unique_suffix = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
                        
                        if input_value.startswith('data:video/'):
                            if 'video/webm' in input_value:
                                extension = 'webm'
                            elif 'video/mp4' in input_value:
                                extension = 'mp4'
                            else:
                                # Unsupported video format - extract the format from the data URL
                                format_part = input_value.split(';')[0].replace('data:video/', '')
                                error_msg = f"Unsupported video format '{format_part}' in node {node_id}, input '{input_key}'. Supported formats: mp4, webm"
                                print(f"worker-comfyui - ERROR: {error_msg}")
                                return [], workflow, error_msg
                            filename = f"uploaded_video_{node_id}_{input_key}_{unique_suffix}.{extension}"
                        else:
                            # Default to image
                            filename = f"uploaded_image_{node_id}_{input_key}_{unique_suffix}.png"
                        
                        # Add to extracted images list
                        extracted_images.append({
                            "name": filename,
                            "image": image_data,  # Pure base64 data (without data URL prefix)
                            "node_id": node_id,
                            "input_key": input_key
                        })
                        
                        # Replace base64 with filename in the cleaned workflow
                        cleaned_workflow[node_id]["inputs"][input_key] = filename
                        
                        print(f"worker-comfyui - Extracted base64 image from {node_id}.{input_key} -> {filename}")
    
    print(f"worker-comfyui - Extracted {len(extracted_images)} base64 images from workflow")
    return extracted_images, cleaned_workflow, None


# def update_workflow_with_images(workflow, images_info):
#     """
#     Update workflow to use uploaded images if needed.
#     This function can be used to automatically map uploaded images to workflow nodes.
    
#     Args:
#         workflow (dict): The workflow object
#         images_info (dict): Information about uploaded images
        
#     Returns:
#         dict: The updated workflow (or original if no changes needed)
#     """
#     if not images_info or images_info["successful_count"] == 0:
#         return workflow
    
#     updated_workflow = workflow.copy()
#     updated_nodes = []
    
#     print(f"worker-comfyui - Attempting to integrate {images_info['successful_count']} uploaded images into workflow...")
    
#     # Get list of uploaded image names
#     uploaded_names = list(images_info["by_name"].keys())
#     print(f"worker-comfyui - Uploaded image names: {uploaded_names}")
    
#     for node_id, node in workflow.items():
#         node_class = node.get("class_type", "")
#         inputs = node.get("inputs", {})
#         updated = False
        
#         # Handle LoadImage nodes
#         if node_class == "LoadImage":
#             print(f"worker-comfyui - Found LoadImage node {node_id} with inputs: {inputs}")
#             if "image" in inputs and uploaded_names:
#                 # Always update LoadImage nodes if we have uploaded images
#                 # This handles both empty fields and placeholder filenames
#                 image_name = uploaded_names[0]
#                 old_image = inputs["image"]
#                 updated_workflow[node_id]["inputs"]["image"] = image_name
#                 updated_nodes.append(f"{node_id} (LoadImage) -> {image_name} (was: {old_image})")
#                 updated = True
#                 print(f"worker-comfyui - Updated LoadImage node {node_id}: '{old_image}' -> '{image_name}'")
#                 # Remove used image from list
#                 uploaded_names.pop(0)
#             else:
#                 print(f"worker-comfyui - Skipping LoadImage node {node_id}: has_image_input={('image' in inputs)}, has_uploaded_names={bool(uploaded_names)}")
        
#         # Handle Video nodes
#         elif node_class in ["VHS_LoadVideoFFmpeg", "VHS_LoadVideo", "LoadVideo"]:
#             print(f"worker-comfyui - Found video node {node_id} ({node_class}) with inputs: {inputs}")
#             if "video" in inputs and uploaded_names:
#                 # Always update video nodes if we have uploaded media
#                 video_name = uploaded_names[0]
#                 old_video = inputs["video"]
#                 updated_workflow[node_id]["inputs"]["video"] = video_name
#                 updated_nodes.append(f"{node_id} ({node_class}) -> {video_name} (was: {old_video})")
#                 updated = True
#                 print(f"worker-comfyui - Updated video node {node_id}: '{old_video}' -> '{video_name}'")
#                 # Remove used video from list
#                 uploaded_names.pop(0)
#             else:
#                 print(f"worker-comfyui - Skipping video node {node_id}: has_video_input={('video' in inputs)}, has_uploaded_names={bool(uploaded_names)}")
        
#         # Handle ImageLoader nodes
#         elif node_class == "ImageLoader":
#             if "image" in inputs and uploaded_names:
#                 # Always update ImageLoader nodes if we have uploaded images
#                 image_name = uploaded_names[0]
#                 old_image = inputs["image"]
#                 updated_workflow[node_id]["inputs"]["image"] = image_name
#                 updated_nodes.append(f"{node_id} (ImageLoader) -> {image_name} (was: {old_image})")
#                 updated = True
#                 uploaded_names.pop(0)
        
#         # Handle other image input nodes
#         elif "image" in inputs and uploaded_names:
#             # Only update if the node class suggests it's an image input node
#             if any(keyword in node_class.lower() for keyword in ["image", "load", "input"]):
#                 image_name = uploaded_names[0]
#                 old_image = inputs["image"]
#                 updated_workflow[node_id]["inputs"]["image"] = image_name
#                 updated_nodes.append(f"{node_id} ({node_class}) -> {image_name} (was: {old_image})")
#                 updated = True
#                 uploaded_names.pop(0)
        
#         if updated:
#             print(f"worker-comfyui - Updated node {node_id} to use uploaded image")
    
#     if updated_nodes:
#         print(f"worker-comfyui - Updated {len(updated_nodes)} nodes with uploaded images:")
#         for update in updated_nodes:
#             print(f"worker-comfyui -   - {update}")
#     else:
#         print("worker-comfyui - No workflow nodes were updated with uploaded images")
    
#     return updated_workflow


def analyze_workflow_structure(workflow):
    """
    Analyze workflow structure to help with debugging and understanding image handling.
    
    Args:
        workflow (dict): The workflow object
        
    Returns:
        dict: Analysis of the workflow structure
    """
    analysis = {
        "total_nodes": len(workflow),
        "node_types": {},
        "image_input_nodes": [],
        "text_input_nodes": [],
        "model_nodes": [],
        "output_nodes": []
    }
    
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Count node types
        analysis["node_types"][node_class] = analysis["node_types"].get(node_class, 0) + 1
        
        # Identify image input nodes
        if node_class in ["LoadImage", "ImageLoader", "LoadImageFromUrl"]:
            analysis["image_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        elif "image" in inputs or "filename" in inputs:
            analysis["image_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        
        # Identify video input nodes
        if node_class in ["VHS_LoadVideoFFmpeg", "VHS_LoadVideo", "LoadVideo"]:
            if "video_input_nodes" not in analysis:
                analysis["video_input_nodes"] = []
            analysis["video_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        elif "video" in inputs:
            if "video_input_nodes" not in analysis:
                analysis["video_input_nodes"] = []
            analysis["video_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        
        # Identify text input nodes
        if node_class in ["CLIPTextEncode", "TextInput"]:
            analysis["text_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        elif "text" in inputs:
            analysis["text_input_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        
        # Identify model nodes
        if node_class in ["CheckpointLoaderSimple", "CheckpointLoader", "LoraLoader"]:
            analysis["model_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
        
        # Identify output nodes
        if node_class in ["SaveImage", "PreviewImage"]:
            analysis["output_nodes"].append({
                "node_id": node_id,
                "class_type": node_class,
                "inputs": inputs
            })
    
    return analysis


def log_workflow_analysis(workflow):
    """
    Log detailed analysis of workflow structure for debugging.
    
    Args:
        workflow (dict): The workflow object
    """
    analysis = analyze_workflow_structure(workflow)
    
    print(f"worker-comfyui - Workflow Analysis:")
    print(f"worker-comfyui -   Total nodes: {analysis['total_nodes']}")
    print(f"worker-comfyui -   Node types: {len(analysis['node_types'])}")
    
    # Log most common node types
    sorted_types = sorted(analysis["node_types"].items(), key=lambda x: x[1], reverse=True)
    print(f"worker-comfyui -   Top node types:")
    for node_type, count in sorted_types[:5]:
        print(f"worker-comfyui -     {node_type}: {count}")
    
    # Log image input nodes
    if analysis["image_input_nodes"]:
        print(f"worker-comfyui -   Image input nodes ({len(analysis['image_input_nodes'])}):")
        for node in analysis["image_input_nodes"]:
            print(f"worker-comfyui -     {node['node_id']} ({node['class_type']})")
            for key, value in node['inputs'].items():
                if key in ["image", "filename", "url"]:
                    print(f"worker-comfyui -       {key}: {value}")
    
    # Log text input nodes
    if analysis["text_input_nodes"]:
        print(f"worker-comfyui -   Text input nodes ({len(analysis['text_input_nodes'])}):")
        for node in analysis["text_input_nodes"]:
            print(f"worker-comfyui -     {node['node_id']} ({node['class_type']})")
    
    # Log model nodes
    if analysis["model_nodes"]:
        print(f"worker-comfyui -   Model nodes ({len(analysis['model_nodes'])}):")
        for node in analysis["model_nodes"]:
            print(f"worker-comfyui -     {node['node_id']} ({node['class_type']})")
            for key, value in node['inputs'].items():
                if key in ["ckpt_name", "model_name", "lora_name"]:
                    print(f"worker-comfyui -       {key}: {value}")


def handler(job):
    """
    Handles a job using ComfyUI via websockets for status and image retrieval.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    job_id = job["id"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")
    user_token = validated_data.get("user_token")
    workflow_id = validated_data.get("workflow_id")
    
    # Verify user and check credits if token is provided
    user_id = None
    user_email = None
    if user_token and FIREBASE_CREDITS_AVAILABLE:
        print("worker-comfyui - Verifying user authentication...")
        user_info = verify_user_token(user_token)
        if user_info:
            user_id = user_info["uid"]
            user_email = user_info.get("email", "unknown")
            print(f"worker-comfyui - User authenticated: {user_email} (ID: {user_id})")
            
            # Get current credits
            credits_result = get_user_credits(user_id)
            if credits_result["error"]:
                print(f"worker-comfyui - Warning: Failed to fetch user credits: {credits_result['error']}")
            else:
                current_credits = credits_result["credits"]
                print(f"worker-comfyui - User has {current_credits} credits")
                
                # Optional: Enforce minimum credit check (BASE_COST_PER_JOB)
                # Uncomment to require minimum credits before execution:
                # if current_credits < BASE_COST_PER_JOB:
                #     return {
                #         "error": f"Insufficient credits. You have {current_credits} credits, but minimum {BASE_COST_PER_JOB} required.",
                #         "current_credits": current_credits,
                #         "required_credits": BASE_COST_PER_JOB
                #     }
        else:
            print("worker-comfyui - Warning: Failed to verify user token. Proceeding without credit system.")
    elif user_token and not FIREBASE_CREDITS_AVAILABLE:
        print("worker-comfyui - User token provided but Firebase credits not available. Proceeding without credit system.")
    
    # Log workflow analysis for debugging (can be disabled with environment variable)
    if os.environ.get("LOG_WORKFLOW_ANALYSIS", "true").lower() == "true":
        log_workflow_analysis(workflow)

    # Extract base64 images from workflow nodes
    extracted_images, cleaned_workflow, extraction_error = extract_base64_images_from_workflow(workflow, job_id)
    if extraction_error:
        return {"error": extraction_error}
    
    # Combine extracted images with any additional input images
    all_images = extracted_images.copy()
    if input_images:
        all_images.extend(input_images)
    
    # Use the cleaned workflow (base64 replaced with filenames)
    workflow = cleaned_workflow

    # Make sure that the ComfyUI HTTP API is available before proceeding
    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {
            "error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."
        }

    # Upload all images (extracted from workflow + additional input images)
    images_info = {}
    if all_images:
        print(f"worker-comfyui - Processing {len(all_images)} image(s) ({len(extracted_images)} from workflow, {len(input_images) if input_images else 0} additional)...")
        upload_result = upload_images(all_images)
        
        # Extract information about uploaded images
        images_info = get_uploaded_images_info(upload_result)
        
        # Log workflow image usage analysis
        log_workflow_image_usage(workflow, images_info)
        
        if upload_result["status"] == "error":
            # Return upload errors
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
                "uploaded_images_info": images_info
            }
        
        print(f"worker-comfyui - Successfully uploaded {images_info['successful_count']}/{images_info['total_count']} images")
    else:
        print("worker-comfyui - No images found in workflow or provided as input")

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    errors = []
    
    # Track execution time for cost calculation
    execution_start_time = time.time()

    try:
        # Establish WebSocket connection
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=50)
        print(f"worker-comfyui - Websocket connected")

        # Queue the workflow
        try:
            queued_workflow = queue_workflow(workflow, client_id)
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(
                    f"Missing 'prompt_id' in queue response: {queued_workflow}"
                )
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            print(f"worker-comfyui - Error queuing workflow: {e}")
            raise ValueError(f"Error queuing workflow: {e}")
        except Exception as e:
            print(f"worker-comfyui - Unexpected error queuing workflow: {e}")
            # For ValueError exceptions from queue_workflow, pass through the original message
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Unexpected error queuing workflow: {e}")

        # Wait for execution completion via WebSocket
        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(
                            f"worker-comfyui - Status update: {status_data.get('exec_info', {}).get('queue_remaining', 'N/A')} items remaining in queue"
                        )
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            print(
                                f"worker-comfyui - Execution finished for prompt {prompt_id}"
                            )
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            print(
                                f"worker-comfyui - Execution error received: {error_details}"
                            )
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                print(f"worker-comfyui - Websocket receive timed out. Still waiting...")
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    # Attempt to reconnect
                    ws = _attempt_websocket_reconnect(
                        ws_url,
                        WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S,
                        closed_err,
                    )

                    print(
                        "worker-comfyui - Resuming message listening after successful reconnect."
                    )
                    continue
                except (
                    websocket.WebSocketConnectionClosedException
                ) as reconn_failed_err:
                    # If _attempt_websocket_reconnect fails, it raises this exception
                    # Let this exception propagate to the outer handler's except block
                    raise reconn_failed_err

            except json.JSONDecodeError:
                print(f"worker-comfyui - Received invalid JSON message via websocket.")

        if not execution_done and not errors:
            raise ValueError(
                "Workflow monitoring loop exited without confirmation of completion or error."
            )

        # Fetch history even if there were execution errors, some outputs might exist
        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                return {"error": error_msg}
            else:
                errors.append(error_msg)
                return {
                    "error": "Job processing failed, prompt ID not found in history.",
                    "details": errors,
                }

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        print(f"worker-comfyui - Processing {len(outputs)} output nodes...")
        text_outputs = {}
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                print(
                    f"worker-comfyui - Node {node_id} contains {len(node_output['images'])} image(s)"
                )
                for image_info in node_output["images"]:
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")

                    # skip temp images
                    if img_type == "temp":
                        print(
                            f"worker-comfyui - Skipping image {filename} because type is 'temp'"
                        )
                        continue

                    if not filename:
                        warn_msg = f"Skipping image in node {node_id} due to missing filename: {image_info}"
                        print(f"worker-comfyui - {warn_msg}")
                        errors.append(warn_msg)
                        continue

                    image_bytes = get_image_data(filename, subfolder, img_type)

                    if image_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".png"
                        
                        # Return as base64 string
                        try:
                            base64_image = base64.b64encode(image_bytes).decode(
                                "utf-8"
                            )
                            # Append dictionary with filename and base64 data
                            output_data.append(
                                {
                                    "filename": filename,
                                    "type": "base64",
                                    "data": base64_image,
                                }
                            )
                            print(f"worker-comfyui - Encoded {filename} as base64")
                        except Exception as e:
                            error_msg = f"Error encoding {filename} to base64: {e}"
                            print(f"worker-comfyui - {error_msg}")
                            errors.append(error_msg)
                    else:
                        error_msg = f"Failed to fetch image data for {filename} from /view endpoint."
                        errors.append(error_msg)

            # Handle video outputs
            if "videos" in node_output:
                print(
                    f"worker-comfyui - Node {node_id} contains {len(node_output['videos'])} video(s)"
                )
                for video_info in node_output["videos"]:
                    filename = video_info.get("filename")
                    subfolder = video_info.get("subfolder", "")
                    vid_type = video_info.get("type")

                    # skip temp videos
                    if vid_type == "temp":
                        print(
                            f"worker-comfyui - Skipping video {filename} because type is 'temp'"
                        )
                        continue

                    if not filename:
                        warn_msg = f"Skipping video in node {node_id} due to missing filename: {video_info}"
                        print(f"worker-comfyui - {warn_msg}")
                        errors.append(warn_msg)
                        continue

                    # Try to get video data using the same endpoint (ComfyUI might serve videos via /view)
                    video_bytes = get_image_data(filename, subfolder, vid_type)

                    if video_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".mp4"
                        
                        # Return as base64 string
                        try:
                            base64_video = base64.b64encode(video_bytes).decode(
                                "utf-8"
                            )
                            # Determine video format from extension
                            video_format = "video/webm" if file_extension.lower() == ".webm" else "video/mp4"
                            
                            # Append dictionary with filename and base64 data
                            output_data.append(
                                {
                                    "filename": filename,
                                    "type": "base64",
                                    "data": base64_video,
                                    "format": video_format,
                                    "media_type": "video"
                                }
                            )
                            print(f"worker-comfyui - Encoded video {filename} as base64")
                        except Exception as e:
                            error_msg = f"Error encoding video {filename} to base64: {e}"
                            print(f"worker-comfyui - {error_msg}")
                            errors.append(error_msg)
                    else:
                        error_msg = f"Failed to fetch video data for {filename} from /view endpoint."
                        errors.append(error_msg)

            # Handle gif outputs (which can contain video files)
            if "gifs" in node_output:
                print(
                    f"worker-comfyui - Node {node_id} contains {len(node_output['gifs'])} gif/video(s)"
                )
                for gif_info in node_output["gifs"]:
                    filename = gif_info.get("filename")
                    subfolder = gif_info.get("subfolder", "")
                    gif_type = gif_info.get("type")

                    # skip temp gifs/videos
                    if gif_type == "temp":
                        print(
                            f"worker-comfyui - Skipping gif/video {filename} because type is 'temp'"
                        )
                        continue

                    if not filename:
                        warn_msg = f"Skipping gif/video in node {node_id} due to missing filename: {gif_info}"
                        print(f"worker-comfyui - {warn_msg}")
                        errors.append(warn_msg)
                        continue

                    # Try to get gif/video data using the same endpoint
                    gif_bytes = get_image_data(filename, subfolder, gif_type)

                    if gif_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".mp4"
                        
                        # Return as base64 string
                        try:
                            base64_gif = base64.b64encode(gif_bytes).decode(
                                "utf-8"
                            )
                            # Determine format from extension or format field
                            gif_format = gif_info.get("format", "")
                            if not gif_format:
                                if file_extension.lower() == ".webm":
                                    gif_format = "video/webm"
                                elif file_extension.lower() == ".gif":
                                    gif_format = "image/gif"
                                else:
                                    gif_format = "video/mp4"
                            
                            # Determine media type based on format
                            media_type = "video" if gif_format.startswith("video/") else "image"
                            
                            # Append dictionary with filename and base64 data
                            output_data.append(
                                {
                                    "filename": filename,
                                    "type": "base64",
                                    "data": base64_gif,
                                    "format": gif_format,
                                    "media_type": media_type
                                }
                            )
                            print(f"worker-comfyui - Encoded gif/video {filename} as base64")
                        except Exception as e:
                            error_msg = f"Error encoding gif/video {filename} to base64: {e}"
                            print(f"worker-comfyui - {error_msg}")
                            errors.append(error_msg)
                    else:
                        error_msg = f"Failed to fetch gif/video data for {filename} from /view endpoint."
                        errors.append(error_msg)

            # Handle text outputs (tags, strings, etc.)
            text_output_types = ["tags", "text", "string", "strings"]
            for output_type in text_output_types:
                if output_type in node_output:
                    print(f"worker-comfyui - Node {node_id} contains {output_type}: {node_output[output_type]}")
                    text_outputs[node_id] = text_outputs.get(node_id, {})
                    text_outputs[node_id][output_type] = node_output[output_type]

            # Check for other output types
            handled_keys = ["images"] + text_output_types
            other_keys = [k for k in node_output.keys() if k not in handled_keys]
            if other_keys:
                print(f"worker-comfyui - Node {node_id} produced unhandled output keys: {other_keys}")
                # Store unhandled outputs as generic text
                for key in other_keys:
                    value = node_output[key]
                    if isinstance(value, (str, list)):
                        print(f"worker-comfyui - Storing {key} as text output: {value}")
                        text_outputs[node_id] = text_outputs.get(node_id, {})
                        text_outputs[node_id][key] = value

    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            print(f"worker-comfyui - Closing websocket connection.")
            ws.close()

    final_result = {}

    if output_data:
        final_result["images"] = output_data

    # Include text outputs if any were found
    if text_outputs:
        final_result["outputs"] = text_outputs
        print(f"worker-comfyui - Including text outputs from {len(text_outputs)} nodes")

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors/warnings: {errors}")

    # Include information about uploaded images in the response
    if images_info and images_info["total_count"] > 0:
        final_result["uploaded_images_info"] = {
            "successful_count": images_info["successful_count"],
            "total_count": images_info["total_count"],
            "uploaded_images": images_info["by_name"]
        }

    # Check if we have any outputs (images or text)
    has_outputs = output_data or text_outputs
    
    if not has_outputs and errors:
        print(f"worker-comfyui - Job failed with no outputs.")
        return {
            "error": "Job processing failed",
            "details": errors,
            "uploaded_images_info": images_info if images_info else None
        }
    elif not has_outputs and not errors:
        print(
            f"worker-comfyui - Job completed successfully, but the workflow produced no outputs."
        )
        final_result["status"] = "success_no_outputs"
        final_result["images"] = []

    # Calculate execution time and cost
    execution_end_time = time.time()
    execution_time_ms = int((execution_end_time - execution_start_time) * 1000)
    
    # Calculate cost based on GPU type and execution time
    cost_info = calculate_job_cost(execution_time_ms)
    final_result["cost_info"] = cost_info
    
    print(f"worker-comfyui - Execution time: {cost_info['execution_time_sec']}s")
    print(f"worker-comfyui - GPU type: {cost_info['gpu_type']}")
    print(f"worker-comfyui - Total cost: {cost_info['total_cost']} credits (base: {cost_info['base_cost']}, execution: {cost_info['execution_cost']})")
    
    # Deduct credits from user's account (server-side, independent of frontend)
    if user_id and FIREBASE_CREDITS_AVAILABLE:
        print(f"worker-comfyui - Deducting {cost_info['total_cost']} credits from user {user_email}...")
        deduction_result = deduct_credits(
            user_id=user_id,
            cost_info=cost_info,
            workflow_id=workflow_id,
            job_id=job_id
        )
        
        if deduction_result["success"]:
            print(f"worker-comfyui - Successfully charged {deduction_result['cost']} credits. New balance: {deduction_result['new_balance']}")
            final_result["credit_deduction"] = {
                "success": True,
                "charged": deduction_result["cost"],
                "new_balance": deduction_result["new_balance"],
                "user_id": user_id
            }
        else:
            print(f"worker-comfyui - Failed to deduct credits: {deduction_result['error']}")
            final_result["credit_deduction"] = {
                "success": False,
                "error": deduction_result["error"],
                "user_id": user_id
            }
    elif user_id and not FIREBASE_CREDITS_AVAILABLE:
        print("worker-comfyui - User authenticated but Firebase credits not available. No credits deducted.")
        final_result["credit_deduction"] = {
            "success": False,
            "error": "Firebase credits system not available"
        }
    else:
        print("worker-comfyui - No user authentication provided. No credits deducted.")
    
    output_summary = []
    if output_data:
        output_summary.append(f"{len(output_data)} image(s)")
    if text_outputs:
        output_summary.append(f"text outputs from {len(text_outputs)} node(s)")
    
    print(f"worker-comfyui - Job completed. Returning {', '.join(output_summary)}.")
    return final_result


if __name__ == "__main__":
    print("worker-comfyui - Starting handler...")
    runpod.serverless.start({"handler": handler})