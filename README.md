# Multi-Modal Data Processing API

A FastAPI-based web application that provides preprocessing and augmentation capabilities for multiple data types:
- Text
- Images
- Audio
- 3D Meshes

## Demo
![Demo](demo.gif)

## Features

### Text Processing
- Preprocessing: Lowercase conversion, punctuation removal, stopword removal, lemmatization
- Augmentation: Synonym replacement, random word swap, random deletion

### Image Processing
- Preprocessing: Grayscale conversion, histogram equalization, noise reduction
- Augmentation: Random rotation, brightness/contrast adjustment, horizontal flip, noise addition

### Audio Processing
- Preprocessing: Mono conversion, resampling, noise reduction, normalization
- Augmentation: Time stretching, pitch shifting, background noise addition
- MFCC Spectrogram visualization

### 3D Mesh Processing
- Preprocessing: Scale normalization, duplicate vertex removal, mesh orientation fixing
- Augmentation: Random rotation, scaling, vertex noise addition
- 3D visualization

## Installation

1. Create a Python virtual environment (Python 3.12 recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## Usage

1. Start the server:

```bash
uvicorn app.main:app --reload
```

2. Open your browser and navigate to:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

3. Use the web interface to:
- Upload data files
- Apply preprocessing
- Apply augmentation
- View results and visualizations

## Supported File Formats

- Text: .txt files
- Images: Common image formats (PNG, JPG, etc.)
- Audio: MP3, WAV files
- 3D Meshes: .off files

## Project Structure

project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── image_processing.py
│   ├── image_augmentation.py
│   ├── audio_processing.py
│   ├── audio_augmentation.py
│   ├── mesh_processing.py
│   ├── mesh_augmentation.py
│   ├── utils.py
│   └── templates/
│       └── index.html
├── data/
│   └── sample.txt
├── demo.gif
├── requirements.txt
└── README.md

## License

MIT License
