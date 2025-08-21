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
                "/workspace/venv_musubi/bin/python src/musubi_tuner/wan_cache_latents.py "
                f"--dataset_config {full_config_path} "
                "--vae /workspace/models/vae/wan_2.1_vae.safetensors"
            )
            
            # Check if cache_latents.py exists
            cache_script_path = "/home/comfyuser/musubi-tuner/src/musubi_tuner/wan_cache_latents.py"
            if not os.path.exists(cache_script_path):
                error_msg = f"Cache script not found: {cache_script_path}"
                self.outputs[process_id].append(error_msg)
                print(f"[{process_id}] {error_msg}")
                return False
            
            # Log the caching command
            print(f"[{process_id}] Caching latents: {cache_command}")
            self.outputs[process_id].append(f"Caching latents: {cache_command}")
            
            # Start the caching process
            env = os.environ.copy()
            env['PATH'] = f"/workspace/venv_musubi/bin:{env.get('PATH', '')}"
            
            # Debug: Log working directory and environment
            self.outputs[process_id].append(f"Working directory: /home/comfyuser/musubi-tuner")
            self.outputs[process_id].append(f"Python path: {env.get('PATH', '')}")
            print(f"[{process_id}] Working directory: /home/comfyuser/musubi-tuner")
            print(f"[{process_id}] Python path: {env.get('PATH', '')}")
            
            cache_process = subprocess.Popen(
                cache_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd="/home/comfyuser/musubi-tuner",
                env=env
            )
            
            # Capture caching output in real-time with better error handling
            import datetime
            import select
            
            # Use select to avoid blocking on readline for both stdout and stderr
            while True:
                # Check if process is still running
                if cache_process.poll() is not None:
                    break
                
                # Try to read output with timeout from both stdout and stderr
                try:
                    readable, _, _ = select.select([cache_process.stdout, cache_process.stderr], [], [], 1.0)
                    
                    for stream in readable:
                        line = stream.readline()
                        if line:
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            stream_type = "[STDERR]" if stream == cache_process.stderr else "[STDOUT]"
                            output_line = f"[{timestamp}] [CACHE] {stream_type} {line.strip()}"
                            self.outputs[process_id].append(output_line)
                            print(f"[{process_id}] [CACHE] {output_line}")
                except (OSError, IOError) as e:
                    print(f"[{process_id}] Error reading cache output: {e}")
                    break
            
            # Read any remaining output from both streams
            remaining_stdout, remaining_stderr = cache_process.communicate()
            
            # Process remaining stdout
            if remaining_stdout:
                for line in remaining_stdout.splitlines():
                    if line.strip():
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] [CACHE] [STDOUT] {line.strip()}"
                        self.outputs[process_id].append(output_line)
                        print(f"[{process_id}] [CACHE] {output_line}")
            
            # Process remaining stderr
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    if line.strip():
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] [CACHE] [STDERR] {line.strip()}"
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

    def cache_te(self, process_id, config_path):
        """Cache latents for the dataset to speed up training"""
        try:
            # Handle config file path - use the full path to OneTrainerConfigs
            if not config_path.startswith('/home/comfyuser/MusubiConfigs/config/'):
                full_config_path = f"/home/comfyuser/MusubiConfigs/config/{config_path}"
            else:
                full_config_path = config_path
            
            # Create the latent caching command
            cache_command = (
                "/workspace/venv_musubi/bin/python src/musubi_tuner/wan_cache_text_encoder_outputs.py "
                f"--dataset_config {full_config_path} "
                "--t5 /workspace/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
            )
    
            # Check if cache_latents.py exists
            cache_script_path = "/home/comfyuser/musubi-tuner/src/musubi_tuner/wan_cache_text_encoder_outputs.py"
            if not os.path.exists(cache_script_path):
                error_msg = f"Cache script not found: {cache_script_path}"
                self.outputs[process_id].append(error_msg)
                print(f"[{process_id}] {error_msg}")
                return False
            
            # Log the caching command
            print(f"[{process_id}] Caching text encoder: {cache_command}")
            self.outputs[process_id].append(f"Caching text encoder: {cache_command}")
            
            # Start the caching process
            env = os.environ.copy()
            env['PATH'] = f"/workspace/venv_musubi/bin:{env.get('PATH', '')}"
            
            # Debug: Log working directory and environment
            self.outputs[process_id].append(f"Working directory: /home/comfyuser/musubi-tuner")
            self.outputs[process_id].append(f"Python path: {env.get('PATH', '')}")
            print(f"[{process_id}] Working directory: /home/comfyuser/musubi-tuner")
            print(f"[{process_id}] Python path: {env.get('PATH', '')}")
            
            cache_process = subprocess.Popen(
                cache_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd="/home/comfyuser/musubi-tuner",
                env=env
            )
            
            # Capture caching output in real-time with better error handling
            import datetime
            import select
            
            # Use select to avoid blocking on readline for both stdout and stderr
            while True:
                # Check if process is still running
                if cache_process.poll() is not None:
                    break
                
                # Try to read output with timeout from both stdout and stderr
                try:
                    readable, _, _ = select.select([cache_process.stdout, cache_process.stderr], [], [], 1.0)
                    
                    for stream in readable:
                        line = stream.readline()
                        if line:
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            stream_type = "[STDERR]" if stream == cache_process.stderr else "[STDOUT]"
                            output_line = f"[{timestamp}] [CACHE] {stream_type} {line.strip()}"
                            self.outputs[process_id].append(output_line)
                            print(f"[{process_id}] [CACHE] {output_line}")
                except (OSError, IOError) as e:
                    print(f"[{process_id}] Error reading cache output: {e}")
                    break
            
            # Read any remaining output from both streams
            remaining_stdout, remaining_stderr = cache_process.communicate()
            
            # Process remaining stdout
            if remaining_stdout:
                for line in remaining_stdout.splitlines():
                    if line.strip():
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] [CACHE] [STDOUT] {line.strip()}"
                        self.outputs[process_id].append(output_line)
                        print(f"[{process_id}] [CACHE] {output_line}")
            
            # Process remaining stderr
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    if line.strip():
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] [CACHE] [STDERR] {line.strip()}"
                        self.outputs[process_id].append(output_line)
                        print(f"[{process_id}] [CACHE] {output_line}")
            
            # Wait for caching to complete
            cache_return_code = cache_process.wait()
            if cache_return_code == 0:
                self.outputs[process_id].append(f"TE caching completed successfully")
                print(f"[{process_id}] TE caching completed successfully")
                return True
            else:
                self.outputs[process_id].append(f"TE caching failed with return code: {cache_return_code}")
                print(f"[{process_id}] TE caching failed with return code: {cache_return_code}")
                return False
                
        except Exception as e:
            self.outputs[process_id].append(f"Error during TE caching: {str(e)}")
            print(f"[{process_id}] Error during TE caching: {str(e)}")
            return False

    def start_process(self, process_id, config_path, lora_name, learning_rate=2e-4, max_train_epochs=16, enable_latent_caching=True, block_swap=0, task='t2v-14B', attention='sdpa', timestep_sampling='shift', flow_shift=2.0, lr_scheduler='constant', lr_warmup_steps=100, network_dim=32, network_alpha=16, network_dropout=0.05, optimizer='AdamW8bit'):
        """Start a new process and capture its output"""
        
        # Initialize the outputs dictionary for this process_id IMMEDIATELY
        self.outputs[process_id] = []
        
        # Add initial status message immediately (before thread starts)
        self.outputs[process_id].append("=== MUSUBI TRAINING INITIALIZED ===")
        self.outputs[process_id].append(f"Process ID: {process_id}")
        self.outputs[process_id].append(f"Config: {config_path}")
        self.outputs[process_id].append(f"LoRA Name: {lora_name}")
        self.outputs[process_id].append(f"Learning Rate: {learning_rate}")
        self.outputs[process_id].append(f"Max Epochs: {max_train_epochs}")
        self.outputs[process_id].append(f"Latent Caching: {'Enabled' if enable_latent_caching else 'Disabled'}")
        self.outputs[process_id].append("Starting training thread...")
        
        def run_process():
            try:
                # Create a placeholder process entry to mark this process as running
                # This prevents the stream from thinking the process is complete during caching
                self.processes[process_id] = "INITIALIZING"
                
                # Handle config file path - use the full path to OneTrainerConfigs
                if not config_path.startswith('/home/comfyuser/MusubiConfigs/config/'):
                    full_config_path = f"/home/comfyuser/MusubiConfigs/config/{config_path}"
                else:
                    full_config_path = config_path
                
                # Cache latents first to speed up training (if enabled)
                if enable_latent_caching:
                    self.outputs[process_id].append("=== PHASE 1: LATENT CACHING ===")
                    self.outputs[process_id].append("Starting latent caching...")
                    print(f"[{process_id}] Starting latent caching...")
                    time.sleep(0.1)  # Small delay to ensure output is captured
                    
                    cache_success = self.cache_latents(process_id, full_config_path)
                    if not cache_success:
                        self.outputs[process_id].append("Warning: Latent caching failed, continuing with training...")
                        print(f"[{process_id}] Warning: Latent caching failed, continuing with training...")
                        time.sleep(0.1)
                    else:
                        self.outputs[process_id].append("Latent caching completed successfully!")
                        print(f"[{process_id}] Latent caching completed successfully!")
                        time.sleep(0.1)
                else:
                    self.outputs[process_id].append("Latent caching disabled, skipping...")
                    print(f"[{process_id}] Latent caching disabled, skipping...")
                    time.sleep(0.1)

                if enable_latent_caching:
                    self.outputs[process_id].append("=== PHASE 2: TE CACHING ===")
                    self.outputs[process_id].append("Starting TE caching...")
                    print(f"[{process_id}] Starting TE caching...")
                    time.sleep(0.1)  # Small delay to ensure output is captured
                    
                    cache_success = self.cache_te(process_id, full_config_path)
                    if not cache_success:
                        self.outputs[process_id].append("Warning: TE caching failed, continuing with training...")
                        print(f"[{process_id}] Warning: TE caching failed, continuing with training...")
                        time.sleep(0.1)
                    else:
                        self.outputs[process_id].append("TE caching completed successfully!")
                        print(f"[{process_id}] TE caching completed successfully!")
                        time.sleep(0.1)
                else:
                    self.outputs[process_id].append("TE caching disabled, skipping...")
                    print(f"[{process_id}] TE caching disabled, skipping...")
                    time.sleep(0.1)
                
                self.outputs[process_id].append("=== PHASE 3: MAIN TRAINING ===")
                self.outputs[process_id].append("Starting training process...")
                print(f"[{process_id}] Starting training process...")
                time.sleep(0.1)

                
                # Create the full command with virtual environment
                full_command = (
                    "/workspace/venv_musubi/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 "
                    "src/musubi_tuner/wan_train_network.py "
                    "--mixed_precision bf16 "
                    f"--task {task} " #t2v-14B
                    # "--dynamo_backend INDUCTOR "
                    # "--dynamo_mode default "
                    "--dit /workspace/models/diffusion_models/wan2.1_t2v_14B_bf16.safetensors "
                    "--vae /workspace/models/vae/wan_2.1_vae.safetensors "
                    f"--dataset_config {full_config_path} "
                    f"--{attention} " #sdpa, xformers
                    f"--optimizer_type {optimizer} " #AdamW, AdamW8bit, AdaFactor
                    f"--learning_rate {learning_rate} " #2e-4
                    f"--lr_scheduler {lr_scheduler} " #constant
                    "--gradient_checkpointing "
                    "--max_data_loader_n_workers 2 "
                    "--persistent_data_loader_workers "
                    "--network_module networks.lora_wan "
                    f"--network_dim {network_dim} " #int
                    f"--network_alpha {network_alpha} " #int
                    f"--network_dropout {network_dropout} " #float
                    "--sigmoid_scale 5.0 "
                    f"--timestep_sampling {timestep_sampling} " #shift
                    f"--discrete_flow_shift {flow_shift} " #2.0
                    f"--max_train_epochs {max_train_epochs} " #16
                    "--weighting_scheme logit_normal "     # logit_normal,mode,cosmap,sigma_sqrt,none
                    "--logit_mean 0.05 "                   # Biased toward earlier timesteps
                    "--logit_std 0.03 "
                    "--num_timestep_buckets 8 "            # ✅ Stratified sampling → stable timesteps
                    "--preserve_distribution_shape "       # Keeps distribution shape when clamping
                    "--min_timestep 100 "                  # Skip very high noise (0-100)
                    "--max_timestep 900 "                  # Skip very clean steps (900-1000)
                    "--save_every_n_epochs 1 "
                    "--seed 42 "
                    "--scale_weight_norms 1.0 "
                    "--output_dir /workspace/models/loras/training/wan "
                    f"--output_name {lora_name} "
                    "--save_last_n_epochs 10 "
                    "--max_grad_norm 1.0 "
                )
                
                if lr_scheduler == 'constant_with_warmup' or lr_scheduler == 'cosine_with_restarts':
                    full_command += f"--lr_warmup_steps {lr_warmup_steps} " #0.1

                # Add blocks_to_swap parameter if block_swap > 0
                if block_swap > 0:
                    full_command += f"--blocks_to_swap {block_swap} "

                # Check if training script exists
                train_script_path = "/home/comfyuser/musubi-tuner/src/musubi_tuner/wan_train_network.py"
                if not os.path.exists(train_script_path):
                    error_msg = f"Training script not found: {train_script_path}"
                    self.outputs[process_id].append(error_msg)
                    print(f"[{process_id}] {error_msg}")
                    return
                
                # Log the command being executed
                print(f"[{process_id}] Executing: {full_command}")
                self.outputs[process_id].append(f"Executing: {full_command}")
                
                # Start the process with virtual environment
                env = os.environ.copy()
                env['PATH'] = f"/workspace/venv_musubi/bin:{env.get('PATH', '')}"
                
                process = subprocess.Popen(
                    full_command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd="/home/comfyuser/musubi-tuner",
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
            # Only terminate if it's a real subprocess object
            if hasattr(process, 'terminate'):
                process.terminate()
            # Remove from processes dict regardless
            del self.processes[process_id]
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
    block_swap = data.get('block_swap', 0)
    task = data.get('task', 't2v-14B')
    attention = data.get('attention', 'sdpa')
    timestep_sampling = data.get('timestep_sampling', 'shift')
    flow_shift = data.get('flow_shift', 2.0)
    lr_scheduler = data.get('lr_scheduler', 'constant')
    lr_warmup_steps = data.get('lr_warmup_steps', 100)
    network_dim = data.get('network_dim', 32)
    network_alpha = data.get('network_alpha', 16)
    network_dropout = data.get('network_dropout', 0.05)
    optimizer = data.get('optimizer', 'AdamW8bit')


    if not config_file:
        return jsonify({'error': 'Config file is required'}), 400
    
    if not lora_name:
        return jsonify({'error': 'Lora name is required'}), 400
    
    # Convert string values to appropriate types
    try:
        learning_rate = float(learning_rate)
        max_train_epochs = int(max_train_epochs)
        block_swap = int(block_swap)
        enable_latent_caching = bool(enable_latent_caching)
        flow_shift = float(flow_shift)
        lr_warmup_steps = int(lr_warmup_steps)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid learning_rate, max_train_epochs, or enable_latent_caching values'}), 400
    
    # Generate unique process ID
    process_id = str(uuid.uuid4())
    
    # Start the training process
    process_manager.start_process(process_id, config_file, lora_name, learning_rate, max_train_epochs, enable_latent_caching, block_swap, task, attention, timestep_sampling, flow_shift, lr_scheduler, lr_warmup_steps, network_dim, network_alpha, network_dropout, optimizer)
    
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
                # Send any remaining output before marking as completed
                final_output = process_manager.get_output(process_id)
                if len(final_output) > last_length:
                    new_lines = final_output[last_length:]
                    for line in new_lines:
                        yield f"data: {json.dumps({'line': line})}\n\n"
                
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                break
            
            # Check if the process is a real subprocess object (not just a placeholder string)
            current_process = process_manager.processes.get(process_id)
            if isinstance(current_process, str):
                # Process is still initializing (caching phase), continue waiting
                # Send a keepalive to prevent connection timeout
                yield f"data: {json.dumps({'keepalive': True})}\n\n"
                time.sleep(1.0)  # Longer sleep during initialization, but with keepalive
                continue
            
            time.sleep(0.1)  # More frequent polling
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 