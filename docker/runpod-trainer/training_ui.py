#!/usr/bin/env python3
import os
import subprocess
import threading
import time
import json
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import queue
import uuid

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
        def run_process():
            try:
                # Handle config file path - if it doesn't start with 'config/', add it
                config_path = command
                if not config_path.startswith('config/'):
                    config_path = f"config/{config_path}"
                
                # Create the full command
                full_command = f"/workspace/venv_onetrainer/bin/python scripts/train.py --config-path={config_path}"
                
                # Log the command being executed
                print(f"[{process_id}] Executing: {full_command}")
                self.outputs[process_id].append(f"Executing: {full_command}")
                
                # Start the process
                process = subprocess.Popen(
                    full_command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd="/home/comfyuser/OneTrainer"
                )
                
                self.processes[process_id] = process
                self.outputs[process_id] = []
                
                # Capture output in real-time
                import datetime
                for line in iter(process.stdout.readline, ''):
                    if line:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] {line.strip()}"
                        self.outputs[process_id].append(output_line)
                        print(f"[{process_id}] {output_line}")
                
                # Wait for process to complete
                return_code = process.wait()
                self.outputs[process_id].append(f"\nProcess completed with return code: {return_code}")
                
            except Exception as e:
                self.outputs[process_id].append(f"Error: {str(e)}")
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
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    data = request.get_json()
    config_file = data.get('config_file', '')
    
    if not config_file:
        return jsonify({'error': 'Config file is required'}), 400
    
    # Generate unique process ID
    process_id = str(uuid.uuid4())
    
    # Start the training process
    process_manager.start_process(process_id, config_file)
    
    return jsonify({
        'process_id': process_id,
        'message': 'Training started successfully'
    })

@app.route('/api/stop_training/<process_id>', methods=['POST'])
def stop_training(process_id):
    success = process_manager.stop_process(process_id)
    return jsonify({
        'success': success,
        'message': 'Training stopped' if success else 'Process not found'
    })

@app.route('/api/processes')
def get_processes():
    running = process_manager.get_running_processes()
    return jsonify({
        'running_processes': running
    })

@app.route('/api/output/<process_id>')
def get_output(process_id):
    output = process_manager.get_output(process_id)
    return jsonify({
        'output': output
    })

@app.route('/api/configs')
def get_configs():
    """Get list of available config files"""
    try:
        from find_configs import find_config_files
        configs = find_config_files()
        return jsonify({
            'configs': configs
        })
    except Exception as e:
        return jsonify({
            'configs': [],
            'error': str(e)
        })

@app.route('/api/stream_output/<process_id>')
def stream_output(process_id):
    def generate():
        last_length = 0
        while True:
            output = process_manager.get_output(process_id)
            if len(output) > last_length:
                new_lines = output[last_length:]
                for line in new_lines:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                last_length = len(output)
            
            # Check if process is still running
            if process_id not in process_manager.processes:
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                break
            
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 