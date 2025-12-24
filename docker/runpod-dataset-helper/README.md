# Dataset Management UI - Caption Editor

A web-based tool for managing image datasets with manual and automated caption/tagging capabilities. Built with Flask backend and vanilla JavaScript frontend.

## Features

### Core Functionality

- **Manual Caption Editing**: Edit captions directly in the UI with real-time auto-save
- **Single Image Auto-Captioning**: Generate captions for individual images using WD14 tagger models
- **Batch Auto-Captioning**: Process entire directories with two modes:
  - **Re-caption All**: Process all images in the directory
  - **Caption Empty**: Only process images without existing caption files
- **Job Management**: Track batch operations with real-time progress updates
- **Tag Statistics**: View tag frequency counts that automatically refresh after operations
- **Lazy Image Loading**: Efficient thumbnail loading with progressive full-image loading
- **Recursive Directory Support**: Handles nested subdirectories automatically

### Technical Features

- **WD14 Tagger Integration**: Uses multiple WD14 models for accurate tagging
  - `wd-v1-4-moat-tagger-v2`
  - `wd-convnext-tagger-v3`
- **Thumbnail Generation**: On-the-fly thumbnail generation for fast browsing
- **Real-time Updates**: Job status polling and automatic tag count refresh
- **Error Handling**: Retry logic for failed image loads with configurable max retries

## Architecture

### Backend (`dataset_ui.py`)

Flask-based REST API with the following components:

- **Process Manager**: Handles long-running training processes (legacy)
- **WD14 Tagger Integration**: Automated caption generation using WD14 models
- **Job System**: Thread-based job processing with status tracking
- **Tag Counter**: Analyzes caption files to generate tag frequency statistics

### Frontend (`caption_editor.html`)

Single-page application with:

- **Image Grid**: Responsive grid layout with lazy loading
- **Caption Editor**: Inline textarea editors with auto-save
- **Job Status Bar**: Real-time job progress tracking
- **Tag Count Panel**: Collapsible tag frequency display
- **Intersection Observer**: Efficient viewport-based image loading

## API Endpoints

### Image Management

#### `POST /api/list_images`
List all images and their captions from a directory.

**Request:**
```json
{
  "directory": "/path/to/images"
}
```

**Response:**
```json
{
  "images": [
    {
      "filename": "image.jpg",
      "caption_filename": "image.txt",
      "caption": "tag1, tag2, tag3",
      "image_path": "/full/path/to/image.jpg"
    }
  ],
  "count": 1
}
```

#### `POST /api/save_caption`
Save a caption file.

**Request:**
```json
{
  "directory": "/path/to/images",
  "caption_filename": "image.txt",
  "caption": "tag1, tag2, tag3"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Caption saved successfully"
}
```

#### `GET /api/get_image`
Serve image files with optional thumbnail generation.

**Query Parameters:**
- `path` (required): Full path to image file
- `thumbnail` (optional): `true`/`false` - generate thumbnail
- `size` (optional): Thumbnail max dimension in pixels (default: 300)

### Auto-Captioning

#### `POST /api/caption_single`
Generate caption for a single image.

**Request:**
```json
{
  "image_path": "/full/path/to/image.jpg"
}
```

**Response:**
```json
{
  "caption": "tag1, tag2, tag3"
}
```

#### `POST /api/autotag/start`
Start a batch captioning job.

**Request:**
```json
{
  "path": "/path/to/directory",
  "mode": "all"  // or "missing" for empty captions only
}
```

**Response:**
```json
{
  "job_id": "uuid-string"
}
```

#### `GET /api/autotag/status/<job_id>`
Get status of a batch captioning job.

**Response:**
```json
{
  "status": "running",  // or "completed", "error"
  "done": 10,
  "total": 100,
  "results": {
    "/path/to/image1.jpg": "tag1, tag2",
    "/path/to/image2.jpg": "tag3, tag4"
  }
}
```

### Tag Statistics

#### `GET /api/tags_count`
Get tag frequency counts for a directory.

**Query Parameters:**
- `directory` (required): Path to directory

**Response:**
```json
{
  "tag1": 45,
  "tag2": 32,
  "tag3": 28
}
```

## Installation

### Docker (Recommended)

The project includes a Dockerfile for containerized deployment:

```bash
docker build -t dataset-ui ./docker/runpod-dataset-helper
docker run -p 5000:5000 dataset-ui
```

### Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Required System Packages:**
- Python 3.12+
- PIL/Pillow (for image processing)
- ONNX Runtime (for WD14 models)

3. **Run the Application:**
```bash
python dataset_ui.py
```

The application will be available at `http://localhost:5000`

## Usage

### Basic Workflow

1. **Load Images:**
   - Enter the directory path containing your images
   - Click "Load Images"
   - Images will be displayed in a grid with their current captions

2. **Manual Editing:**
   - Click on any caption textarea to edit
   - Changes are automatically saved after 1 second of inactivity
   - A "Saved" indicator confirms successful saves

3. **Single Image Captioning:**
   - Click the "Caption" button below any image
   - Confirm the action
   - The caption will be generated and saved automatically

4. **Batch Operations:**
   - **Re-caption All**: Click "📝 Re-caption All" to process all images
   - **Caption Empty**: Click "📝 Caption Empty" to only process images without captions
   - Monitor progress in the job status bar

5. **View Tag Statistics:**
   - Tag counts are displayed in the collapsible panel at the top
   - Counts automatically refresh after save/autotag operations
   - Tags are sorted by frequency (highest first)

### Job Status Tracking

- Jobs appear in the sticky status bar at the top
- Progress shows: `done/total (percentage%)`
- Status badges: Running (yellow), Completed (green), Error (red)
- Completed/error jobs auto-remove after 5 seconds

### Image Loading

- Thumbnails (300px) load first for fast browsing
- Full images load after 2 seconds if still in viewport
- Failed loads retry up to 3 times with 10-second delays
- Images load in batches of 5 with 300ms delays between batches

## Configuration

### WD14 Tagger Models

Models are configured in `wd14_tagger/worker.py`:

```python
FIXED_MODELS = {
    "wd-v1-4-moat-tagger-v2": {
        "threshold": 0.35,
        "character_threshold": 0.85
    },
    "wd-convnext-tagger-v3": {
        "threshold": 0.35,
        "character_threshold": 0.85
    }
}
```

### Image Loading Settings

Configured in `caption_editor.html`:

```javascript
const BATCH_SIZE = 5;           // Images per batch
const BATCH_DELAY = 300;         // ms between batches
const RETRY_DELAY = 10000;       // ms before retry
const MAX_RETRIES = 3;           // Max retry attempts
```

### Job Polling

Jobs are polled every 2 seconds when active.

## File Structure

```
docker/runpod-dataset-helper/
├── dataset_ui.py              # Flask backend application
├── tag_counter.py            # Tag frequency analysis
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container build file
├── templates/
│   └── caption_editor.html  # Frontend UI
└── wd14_tagger/
    ├── tagger.py            # WD14 tagger implementation
    ├── worker.py            # Batch processing worker
    ├── jobs.py              # Job state management
    ├── model_manager.py     # Model download/management
    └── utils.py             # Utility functions
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)
- TIFF (.tiff, .tif)

## Caption File Format

- Caption files use `.txt` extension
- Same name as image file (e.g., `image.jpg` → `image.txt`)
- Comma-separated tags: `tag1, tag2, tag3`
- UTF-8 encoding

## Error Handling

- **Image Load Failures**: Automatic retry with exponential backoff
- **Save Failures**: Error indicator shown in UI
- **Job Failures**: Error status with message displayed
- **API Errors**: User-friendly error messages

## Performance Considerations

- **Lazy Loading**: Images load only when visible in viewport
- **Thumbnail First**: Small thumbnails load before full images
- **Batch Processing**: Images processed in controlled batches
- **Tag Counting**: Efficient single-pass directory scanning

## Browser Compatibility

- Modern browsers with ES6+ support
- Intersection Observer API required for lazy loading
- Fetch API for HTTP requests

## Development

### Running in Debug Mode

```python
# In dataset_ui.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
```

### Testing Endpoints

```bash
# List images
curl -X POST http://localhost:5000/api/list_images \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/images"}'

# Start batch job
curl -X POST http://localhost:5000/api/autotag/start \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/images", "mode": "all"}'
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

