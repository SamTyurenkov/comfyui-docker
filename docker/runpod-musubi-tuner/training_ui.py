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
    
    def cache_latents(self, process_id, config_path):
        """Cache latents for the dataset to speed up training"""
        try:
            # Handle config file path - use the full path to OneTrainerConfigs
            if not config_path.startswith('/home/comfyuser/MusubiConfigs/config/'):
                full_config_path = f"/home/comfyuser/MusubiConfigs/config/{config_path}"
            else:
                full_config_path = config_path
            
            # Create the latent caching command
            cache_command = (
                "/workspace/venv_musubi/bin/python src/musubi_tuner/cache_latents.py "
                f"--dataset_config {full_config_path} "
                "--dit /workspace/models/diffusion_models/wan2.1_t2v_14B_bf16.safetensors "
                "--task t2v-14B --mixed_precision bf16 " #--fp8_base 
                "--xformers --max_data_loader_n_workers 2 --persistent_data_loader_workers"
            )
            
            # Log the caching command
            print(f"[{process_id}] Caching latents: {cache_command}")
            self.outputs[process_id].append(f"Caching latents: {cache_command}")
            
            # Start the caching process
            env = os.environ.copy()
            env['PATH'] = f"/workspace/venv_musubi/bin:{env.get('PATH', '')}"
            
            cache_process = subprocess.Popen(
                cache_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd="/home/comfyuser/musubi-tuner",
                env=env
            )
            
            # Capture caching output in real-time
            import datetime
            for line in iter(cache_process.stdout.readline, ''):
                if line:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    output_line = f"[{timestamp}] [CACHE] {line.strip()}"
                    self.outputs[process_id].append(output_line)
                    print(f"[{process_id}] [CACHE] {output_line}")
            
            # Wait for caching to complete
            cache_return_code = cache_process.wait()
            if cache_return_code == 0:
                self.outputs[process_id].append(f"Latent caching completed successfully")
                print(f"[{process_id}] Latent caching completed successfully")
                return True
            else:
                self.outputs[process_id].append(f"Latent caching failed with return code: {cache_return_code}")
                print(f"[{process_id}] Latent caching failed with return code: {cache_return_code}")
                return False
                
        except Exception as e:
            self.outputs[process_id].append(f"Error during latent caching: {str(e)}")
            print(f"[{process_id}] Error during latent caching: {str(e)}")
            return False
    
    def start_process(self, process_id, config_path, lora_name, learning_rate=2e-4, max_train_epochs=16, enable_latent_caching=True):
        """Start a new process and capture its output"""
        
        # Initialize the outputs dictionary for this process_id before starting the thread
        self.outputs[process_id] = []
        
        def run_process():
            try:
                # Handle config file path - use the full path to OneTrainerConfigs
                if not config_path.startswith('/home/comfyuser/MusubiConfigs/config/'):
                    full_config_path = f"/home/comfyuser/MusubiConfigs/config/{config_path}"
                else:
                    full_config_path = config_path
                
                # Cache latents first to speed up training (if enabled)
                if enable_latent_caching:
                    self.outputs[process_id].append("Starting latent caching...")
                    print(f"[{process_id}] Starting latent caching...")
                    
                    cache_success = self.cache_latents(process_id, full_config_path)
                    if not cache_success:
                        self.outputs[process_id].append("Warning: Latent caching failed, continuing with training...")
                        print(f"[{process_id}] Warning: Latent caching failed, continuing with training...")
                else:
                    self.outputs[process_id].append("Latent caching disabled, skipping...")
                    print(f"[{process_id}] Latent caching disabled, skipping...")
                
                self.outputs[process_id].append("Starting training process...")
                print(f"[{process_id}] Starting training process...")
                
                # Create the full command with virtual environment
                full_command = (
                    "/workspace/venv_musubi/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 "
                    "src/musubi_tuner/wan_train_network.py --compile "
                    "--task t2v-14B "
                    "--dit /workspace/models/diffusion_models/wan2.1_t2v_14B_bf16.safetensors "
                    f"--dataset_config {full_config_path} --xformers --mixed_precision bf16 " #--fp8_base 
                    f"--optimizer_type adamw8bit --learning_rate {learning_rate} --gradient_checkpointing "
                    "--max_data_loader_n_workers 2 --persistent_data_loader_workers "
                    "--network_module networks.lora_wan --network_dim 32 "
                    "--timestep_sampling shift --discrete_flow_shift 3.0 "
                    f"--max_train_epochs {max_train_epochs} --save_every_n_epochs 1 --seed 42 "
                    f"--output_dir /workspace/models/loras/training/wan --output_name {lora_name}"
                )

                # Log the command being executed
                print(f"[{process_id}] Executing: {full_command}")
                self.outputs[process_id].append(f"Executing: {full_command}")
                
                # Start the process with virtual environment
                env = os.environ.copy()
                env['PATH'] = f"/workspace/venv_musubi/bin:{env.get('PATH', '')}"
                
                process = subprocess.Popen(
                    full_command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd="/home/comfyuser/musubi-tuner",
                    env=env
                )
                
                self.processes[process_id] = process
                
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
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    data = request.get_json()
    config_file = data.get('config_file', '')
    lora_name = data.get('lora_name', '')
    learning_rate = data.get('learning_rate', 2e-4)
    max_train_epochs = data.get('max_train_epochs', 16)
    enable_latent_caching = data.get('enable_latent_caching', True)
    
    if not config_file:
        return jsonify({'error': 'Config file is required'}), 400
    
    if not lora_name:
        return jsonify({'error': 'Lora name is required'}), 400
    
    # Convert string values to appropriate types
    try:
        learning_rate = float(learning_rate)
        max_train_epochs = int(max_train_epochs)
        enable_latent_caching = bool(enable_latent_caching)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid learning_rate, max_train_epochs, or enable_latent_caching values'}), 400
    
    # Generate unique process ID
    process_id = str(uuid.uuid4())
    
    # Start the training process
    process_manager.start_process(process_id, config_file, lora_name, learning_rate, max_train_epochs, enable_latent_caching)
    
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