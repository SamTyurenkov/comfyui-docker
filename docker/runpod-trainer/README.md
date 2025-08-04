# OneTrainer Training UI

A web-based interface for running OneTrainer LoRA training commands.

## Features

- **Web-based UI**: Access the training interface through your browser
- **Real-time output**: See training progress in real-time
- **Process management**: Start, stop, and monitor training processes
- **Config file discovery**: Automatically find available config files
- **Multiple processes**: Run multiple training jobs simultaneously

## Access

The training UI is available at:
- **URL**: `http://your-domain:80`
- **Port**: 80 (via nginx proxy)

## Usage

1. **Access the UI**: Navigate to the training UI URL
2. **Load Configs**: Click "Load Available Configs" to see available config files
3. **Select Config**: Choose a config file from the dropdown or enter the path manually
4. **Start Training**: Click "Start Training" to begin the training process
5. **Monitor Progress**: Watch the real-time output in the terminal-like display
6. **Stop Training**: Use the "Stop Training" button to halt the process

## API Endpoints

- `GET /api/configs` - Get list of available config files
- `POST /api/start_training` - Start a new training process
- `POST /api/stop_training/<process_id>` - Stop a running process
- `GET /api/processes` - Get list of running processes
- `GET /api/output/<process_id>` - Get output for a specific process
- `GET /api/stream_output/<process_id>` - Stream real-time output (SSE)

## Training Command

The UI runs the following command for each training job:

```bash
/workspace/venv_onetrainer/bin/python scripts/train.py --config-path="config/$CONFIG_FILE"
```

Config files are typically stored in `/workspace/OneTrainer/config/` directory.

## Configuration

The training UI is configured to:
- Run on port 5000 internally
- Be proxied through nginx on port 81 at `/training`
- Work with OneTrainer installed at `/workspace/OneTrainer`
- Use the Python virtual environment at `/workspace/venv_onetrainer/bin/python`

## Troubleshooting

1. **No config files found**: Make sure you have config files in the OneTrainer directory
2. **Training fails to start**: Check that the config file path is correct
3. **No output**: Verify that the training process is actually running
4. **Connection issues**: Ensure the container is running and ports are properly exposed

## Development

To modify the training UI:

1. Edit `training_ui.py` for backend changes
2. Edit `templates/index.html` for frontend changes
3. Rebuild the Docker container to apply changes

## Dependencies

- Flask 2.3.3
- Flask-CORS 4.0.0
- Werkzeug 2.3.7 