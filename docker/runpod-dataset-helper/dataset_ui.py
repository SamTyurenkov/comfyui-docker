#!/usr/bin/env python3
import os
import subprocess
import threading
import time
import json
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
import queue
import uuid
from io import BytesIO

from .wd14_tagger.tagger import WD14Tagger
from .wd14_tagger.model_manager import download_model
from .wd14_tagger.utils import iter_images
from .wd14_tagger.worker import autotag_worker   # ✅ REQUIRED

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Store running processes
running_processes = {}
process_outputs = {}

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.outputs = {}
    
    def start_process(self, process_id, command):
        """Start a new process and capture its output"""
        
        # Initialize the outputs dictionary for this process_id before starting the thread
        self.outputs[process_id] = []
        
        def run_process():
            try:
                # Handle config file path - use the full path to OneTrainerConfigs
                config_path = command
                if not config_path.startswith('/home/comfyuser/OneTrainerConfigs/config/'):
                    config_path = f"/home/comfyuser/OneTrainerConfigs/config/{config_path}"
                
                # Check if training script exists
                train_script_path = "/home/comfyuser/OneTrainer/scripts/train.py"
                if not os.path.exists(train_script_path):
                    error_msg = f"Training script not found: {train_script_path}"
                    self.outputs[process_id].append(error_msg)
                    print(f"[{process_id}] {error_msg}")
                    return
                
                # Create the full command
                full_command = f"/workspace/venv_onetrainer/bin/python scripts/train.py --config-path={config_path}"
                
                # Log the command being executed
                print(f"[{process_id}] Executing: {full_command}")
                self.outputs[process_id].append(f"Executing: {full_command}")
                time.sleep(0.1)  # Small delay to ensure output is captured
                
                env = os.environ.copy()
                env['PATH'] = f"/workspace/venv_onetrainer/bin:{env.get('PATH', '')}"
                
                # Debug: Log working directory and environment
                self.outputs[process_id].append(f"Working directory: /home/comfyuser/OneTrainer")
                self.outputs[process_id].append(f"Python path: {env.get('PATH', '')}")
                print(f"[{process_id}] Working directory: /home/comfyuser/OneTrainer")
                print(f"[{process_id}] Python path: {env.get('PATH', '')}")

                # Start the process with separate stdout and stderr pipes
                process = subprocess.Popen(
                    full_command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd="/home/comfyuser/OneTrainer",
                    env=env
                )
                
                self.processes[process_id] = process
                
                # Capture output in real-time with better error handling
                import datetime
                import select
                
                # Use select to avoid blocking on readline for both stdout and stderr
                while True:
                    # Check if process is still running
                    if process.poll() is not None:
                        break
                    
                    # Try to read output with timeout from both stdout and stderr
                    try:
                        readable, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)
                        
                        for stream in readable:
                            line = stream.readline()
                            if line:
                                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                stream_type = "[STDERR]" if stream == process.stderr else "[STDOUT]"
                                output_line = f"[{timestamp}] {stream_type} {line.strip()}"
                                self.outputs[process_id].append(output_line)
                                print(f"[{process_id}] {output_line}")
                                
                    except (OSError, IOError) as e:
                        print(f"[{process_id}] Error reading process output: {e}")
                        break
                
                # Read any remaining output from both streams
                remaining_stdout, remaining_stderr = process.communicate()
                
                # Process remaining stdout
                if remaining_stdout:
                    for line in remaining_stdout.splitlines():
                        if line.strip():
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            output_line = f"[{timestamp}] [STDOUT] {line.strip()}"
                            self.outputs[process_id].append(output_line)
                            print(f"[{process_id}] {output_line}")
                
                # Process remaining stderr
                if remaining_stderr:
                    for line in remaining_stderr.splitlines():
                        if line.strip():
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            output_line = f"[{timestamp}] [STDERR] {line.strip()}"
                            self.outputs[process_id].append(output_line)
                            print(f"[{process_id}] {output_line}")
                
                # Wait for process to complete
                return_code = process.wait()
                self.outputs[process_id].append(f"\nProcess completed with return code: {return_code}")
                
                # Force flush the output to ensure it's available for streaming
                import sys
                sys.stdout.flush()
                
            except Exception as e:
                if process_id in self.outputs:
                    self.outputs[process_id].append(f"Error: {str(e)}")
                print(f"[{process_id}] Error: {str(e)}")
            finally:
                if process_id in self.processes:
                    del self.processes[process_id]
        
        # Start the process in a separate thread
        thread = threading.Thread(target=run_process)
        thread.daemon = True
        thread.start()
        
        return process_id
    
    def stop_process(self, process_id):
        """Stop a running process"""
        if process_id in self.processes:
            process = self.processes[process_id]
            process.terminate()
            return True
        return False
    
    def get_output(self, process_id):
        """Get the output of a process"""
        return self.outputs.get(process_id, [])
    
    def get_running_processes(self):
        """Get list of running process IDs"""
        return list(self.processes.keys())

process_manager = ProcessManager()

@app.route('/')
def index():
    return render_template('caption_editor.html')

@app.route('/api/list_images', methods=['POST'])
def list_images():
    """List all images and their corresponding caption files from a directory"""
    data = request.get_json()
    directory = data.get('directory', '')
    
    if not directory:
        return jsonify({'error': 'Directory is required'}), 400
    
    if not os.path.exists(directory):
        return jsonify({'error': 'Directory does not exist'}), 404
    
    if not os.path.isdir(directory):
        return jsonify({'error': 'Path is not a directory'}), 400
    
    try:
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        images = []
        
        # Recursively walk through directory and subdirectories
        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.isfile(filepath):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in image_extensions:
                        # Look for corresponding caption file in the same directory
                        caption_path = os.path.join(root, name + '.txt')
                        caption_text = ''
                        if os.path.exists(caption_path):
                            try:
                                with open(caption_path, 'r', encoding='utf-8') as f:
                                    caption_text = f.read()
                            except Exception as e:
                                caption_text = f'Error reading caption: {str(e)}'
                        
                        # Get relative path from the root directory for display
                        rel_path = os.path.relpath(root, directory)
                        if rel_path == '.':
                            display_path = filename
                        else:
                            display_path = os.path.join(rel_path, filename)
                        
                        images.append({
                            'filename': display_path,
                            'caption_filename': os.path.join(rel_path, name + '.txt') if rel_path != '.' else (name + '.txt'),
                            'caption': caption_text,
                            'image_path': filepath
                        })
        
        # Sort by filename
        images.sort(key=lambda x: x['filename'])
        
        return jsonify({
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_caption', methods=['POST'])
def save_caption():
    """Save a caption file"""
    data = request.get_json()
    directory = data.get('directory', '')
    caption_filename = data.get('caption_filename', '')
    caption_text = data.get('caption', '')
    
    if not directory or not caption_filename:
        return jsonify({'error': 'Directory and caption filename are required'}), 400
    
    try:
        caption_path = os.path.join(directory, caption_filename)
        
        # Ensure root directory exists
        if not os.path.exists(directory):
            return jsonify({'error': 'Directory does not exist'}), 404
        
        # Ensure the subdirectory for the caption file exists
        caption_dir = os.path.dirname(caption_path)
        if caption_dir and not os.path.exists(caption_dir):
            os.makedirs(caption_dir, exist_ok=True)
        
        # Write caption file
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        
        return jsonify({
            'success': True,
            'message': 'Caption saved successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_image')
def get_image():
    """Serve image files"""
    image_path = request.args.get('path', '')
    thumbnail = request.args.get('thumbnail', 'false').lower() == 'true'
    size = int(request.args.get('size', 300))
    
    if not image_path:
        return jsonify({'error': 'Image path is required'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    if not os.path.isfile(image_path):
        return jsonify({'error': 'Path is not a file'}), 400
    
    try:
        if thumbnail:
            if not PIL_AVAILABLE:
                # Fallback: serve original image if PIL not available
                return send_file(image_path)
            
            # Generate thumbnail on-the-fly
            img = Image.open(image_path)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for formats like PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            img_io = BytesIO()
            img.save(img_io, 'JPEG', quality=85, optimize=True)
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/jpeg')
        else:
            return send_file(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================
# AUTOTAG ROUTES (FIXED)
# =============================

@app.post("/api/autotag/start")
def start_autotag():
    data = request.json
    job_id = str(uuid.uuid4())

    autotag_jobs[job_id] = {
        "status": "running",
        "done": 0,
        "total": 0,
        "results": {}
    }

    thread = threading.Thread(
        target=autotag_worker,
        args=(
            job_id,
            data["path"],
            data.get("mode", "all"),
            data.get("models", {})
        ),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.get("/api/autotag/status/<job_id>")
def autotag_status(job_id):
    return jsonify(autotag_jobs.get(job_id, {}))
    data = request.json
    job_id = str(uuid.uuid4())

    autotag_jobs[job_id] = {
        "status": "running",
        "done": 0,
        "total": 0,
        "results": {}
    }

    thread = threading.Thread(
        target=autotag_worker,
        args=(
            job_id,
            data["path"],
            data.get("mode", "all"),
            data.get("models", {})
        ),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})