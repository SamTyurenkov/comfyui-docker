# Enhanced Image Handling in ComfyUI Handler

## Overview

The handler has been enhanced to better handle multiple image inputs in JSON format for ComfyUI workflows. This allows workflows to process multiple images and provides better debugging and integration capabilities.

## Key Features

### 1. Enhanced Image Validation
- Validates image format (base64 or data URL)
- Checks for required fields (`name` and `image`)
- Provides detailed error messages for each image
- Supports both pure base64 and data URL formats

### 2. Improved Image Upload
- Better logging of upload progress
- Detailed information about uploaded images
- Error handling for each image individually
- Returns comprehensive upload status

### 3. Workflow Analysis
- Analyzes workflow structure for debugging
- Identifies image input nodes
- Logs workflow node types and connections
- Helps understand how images are used

### 4. Automatic Image Integration
- Automatically maps uploaded images to workflow nodes
- Updates LoadImage and ImageLoader nodes
- Essential for workflows to work properly with input images

## Input Format

Images should be provided in the following format:

```json
{
  "input": {
    "workflow": { ... },
    "images": [
      {
        "name": "image1.png",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
      },
      {
        "name": "image2.png", 
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
      }
    ]
  }
}
```

## Environment Variables

- `LOG_WORKFLOW_ANALYSIS`: Set to "false" to disable workflow analysis logging

## Response Format

The handler now returns additional information about uploaded images:

```json
{
  "images": [...],
  "uploaded_images_info": {
    "successful_count": 2,
    "total_count": 2,
    "uploaded_images": {
      "image1.png": {
        "index": 0,
        "size_bytes": 1024,
        "uploaded": true
      },
      "image2.png": {
        "index": 1,
        "size_bytes": 2048,
        "uploaded": true
      }
    }
  }
}
```

## Debugging

The handler provides extensive logging for debugging:

1. **Workflow Analysis**: Logs workflow structure and node types
2. **Image Upload**: Detailed progress and status for each image
3. **Image Integration**: Shows which nodes were updated with images
4. **Error Handling**: Comprehensive error messages with context

## Testing

Use `test_input_with_images.json` to test the enhanced image handling functionality.

## Benefits

1. **Multiple Image Support**: Handle workflows with multiple image inputs
2. **Better Error Handling**: Detailed error messages for debugging
3. **Workflow Integration**: Automatic mapping of images to workflow nodes
4. **Comprehensive Logging**: Better visibility into image processing
5. **Flexible Format**: Support for both base64 and data URL formats
