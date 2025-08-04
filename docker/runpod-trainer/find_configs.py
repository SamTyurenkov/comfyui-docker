#!/usr/bin/env python3
import os
import json
from pathlib import Path

def find_config_files():
    """Find all config files in the OneTrainer config directory"""
    config_path = Path("/workspace/OneTrainerConfigs/config")
    config_files = []
    
    if not config_path.exists():
        print(f"Config directory not found: {config_path}")
        return config_files
    
    # Look for common config file patterns in the config directory
    patterns = ["*.json", "*.yaml", "*.yml", "config*.json", "config*.yaml", "config*.yml"]
    
    for pattern in patterns:
        for config_file in config_path.rglob(pattern):
            if config_file.is_file():
                # Skip hidden files and common directories
                if any(part.startswith('.') or part in ['node_modules', '__pycache__', '.git'] for part in config_file.parts):
                    continue
                
                relative_path = config_file.relative_to(config_path)
                config_files.append(str(relative_path))
    
    return sorted(config_files)

if __name__ == "__main__":
    configs = find_config_files()
    print("Available config files:")
    for config in configs:
        print(f"  - {config}") 