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

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


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
            
            # Check if it's a valid base64 or data URL
            if "," in image_data:
                # Data URL format: data:image/png;base64,<base64_data>
                parts = image_data.split(",", 1)
                if len(parts) != 2 or not parts[0].startswith("data:image/"):
                    return None, f"Invalid data URL format at index {i}"
                base64_data = parts[1]
            else:
                # Pure base64 format
                base64_data = image_data
            
            # Validate base64 data
            try:
                base64.b64decode(base64_data)
            except Exception:
                return None, f"Invalid base64 data at index {i}"

    # Return validated data and no error
    return {"workflow": workflow, "images": images}, None


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
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.

    Returns:
        dict: A dictionary indicating success or error with detailed information about uploaded images.
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

            # --- Strip Data URI prefix if present ---
            if "," in image_data_uri:
                # Find the comma and take everything after it
                base64_data = image_data_uri.split(",", 1)[1]
                print(f"worker-comfyui - Extracted base64 data from data URL for {name}")
            else:
                # Assume it's already pure base64
                base64_data = image_data_uri
                print(f"worker-comfyui - Using pure base64 data for {name}")
            # --- End strip ---

            blob = base64.b64decode(base64_data)  # Decode the cleaned data
            print(f"worker-comfyui - Decoded {len(blob)} bytes for {name}")

            # Prepare the form data
            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            # POST request to upload the image
            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
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


def extract_base64_images_from_workflow(workflow):
    """
    Extract base64 image data from workflow nodes and prepare them for upload.
    
    Args:
        workflow (dict): The workflow object
        
    Returns:
        tuple: (extracted_images, cleaned_workflow)
               extracted_images: list of image objects with name and base64 data
               cleaned_workflow: workflow with base64 data replaced with placeholder filenames
    """
    extracted_images = []
    cleaned_workflow = workflow.copy()
    
    print("worker-comfyui - Scanning workflow for base64 image data...")
    
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Look for LoadImage nodes or other image input nodes
        if node_class in ["LoadImage", "ImageLoader"] or "image" in inputs:
            for input_key, input_value in inputs.items():
                if input_key == "image" and isinstance(input_value, str):
                    # Check if this looks like base64 data or data URL
                    is_base64 = False
                    image_data = input_value
                    
                    # Handle data URL format: data:image/png;base64,<base64_data>
                    if input_value.startswith('data:image/'):
                        if ',' in input_value:
                            image_data = input_value.split(',', 1)[1]
                            is_base64 = True
                            print(f"worker-comfyui - Found data URL in node {node_id}, input '{input_key}'")
                    # Handle pure base64 (long string, starts with common base64 chars)
                    elif len(input_value) > 100 and input_value.startswith(('iVBORw0KGgo', '/9j/', 'UklGR')):
                        is_base64 = True
                        print(f"worker-comfyui - Found base64 image data in node {node_id}, input '{input_key}'")
                    
                    if is_base64:
                        # Generate a filename for this image
                        filename = f"uploaded_image_{node_id}_{input_key}.png"
                        
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
    return extracted_images, cleaned_workflow


def update_workflow_with_images(workflow, images_info):
    """
    Update workflow to use uploaded images if needed.
    This function can be used to automatically map uploaded images to workflow nodes.
    
    Args:
        workflow (dict): The workflow object
        images_info (dict): Information about uploaded images
        
    Returns:
        dict: The updated workflow (or original if no changes needed)
    """
    if not images_info or images_info["successful_count"] == 0:
        return workflow
    
    updated_workflow = workflow.copy()
    updated_nodes = []
    
    print(f"worker-comfyui - Attempting to integrate {images_info['successful_count']} uploaded images into workflow...")
    
    # Get list of uploaded image names
    uploaded_names = list(images_info["by_name"].keys())
    print(f"worker-comfyui - Uploaded image names: {uploaded_names}")
    
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {})
        updated = False
        
        # Handle LoadImage nodes
        if node_class == "LoadImage":
            print(f"worker-comfyui - Found LoadImage node {node_id} with inputs: {inputs}")
            if "image" in inputs and uploaded_names:
                # Always update LoadImage nodes if we have uploaded images
                # This handles both empty fields and placeholder filenames
                image_name = uploaded_names[0]
                old_image = inputs["image"]
                updated_workflow[node_id]["inputs"]["image"] = image_name
                updated_nodes.append(f"{node_id} (LoadImage) -> {image_name} (was: {old_image})")
                updated = True
                print(f"worker-comfyui - Updated LoadImage node {node_id}: '{old_image}' -> '{image_name}'")
                # Remove used image from list
                uploaded_names.pop(0)
            else:
                print(f"worker-comfyui - Skipping LoadImage node {node_id}: has_image_input={('image' in inputs)}, has_uploaded_names={bool(uploaded_names)}")
        
        # Handle ImageLoader nodes
        elif node_class == "ImageLoader":
            if "image" in inputs and uploaded_names:
                # Always update ImageLoader nodes if we have uploaded images
                image_name = uploaded_names[0]
                old_image = inputs["image"]
                updated_workflow[node_id]["inputs"]["image"] = image_name
                updated_nodes.append(f"{node_id} (ImageLoader) -> {image_name} (was: {old_image})")
                updated = True
                uploaded_names.pop(0)
        
        # Handle other image input nodes
        elif "image" in inputs and uploaded_names:
            # Only update if the node class suggests it's an image input node
            if any(keyword in node_class.lower() for keyword in ["image", "load", "input"]):
                image_name = uploaded_names[0]
                old_image = inputs["image"]
                updated_workflow[node_id]["inputs"]["image"] = image_name
                updated_nodes.append(f"{node_id} ({node_class}) -> {image_name} (was: {old_image})")
                updated = True
                uploaded_names.pop(0)
        
        if updated:
            print(f"worker-comfyui - Updated node {node_id} to use uploaded image")
    
    if updated_nodes:
        print(f"worker-comfyui - Updated {len(updated_nodes)} nodes with uploaded images:")
        for update in updated_nodes:
            print(f"worker-comfyui -   - {update}")
    else:
        print("worker-comfyui - No workflow nodes were updated with uploaded images")
    
    return updated_workflow


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
    
    # Log workflow analysis for debugging (can be disabled with environment variable)
    if os.environ.get("LOG_WORKFLOW_ANALYSIS", "true").lower() == "true":
        log_workflow_analysis(workflow)

    # Extract base64 images from workflow nodes
    extracted_images, cleaned_workflow = extract_base64_images_from_workflow(workflow)
    
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