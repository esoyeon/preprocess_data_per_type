from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import io
from .preprocessing import preprocess_text
from .augmentation import augment_text
from .image_processing import preprocess_image
from .image_augmentation import augment_image
from .audio_processing import preprocess_audio
from .audio_augmentation import augment_audio
from .mesh_processing import preprocess_mesh
from .mesh_augmentation import augment_mesh
import base64

app = FastAPI(title="Text and Image Processing API")

# HTML 템플릿 설정만 유지
templates = Jinja2Templates(directory="app/templates")

# StaticFiles 마운트 부분 제거
# app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Store the current state of the data
class DataState:
    # Text states
    original_text: str = ""
    preprocessed_text: str = ""
    augmented_text: str = ""

    # Image states
    original_image: bytes = b""
    preprocessed_image: bytes = b""
    augmented_image: bytes = b""

    # Audio states
    original_audio: bytes = b""
    preprocessed_audio: bytes = b""
    augmented_audio: bytes = b""

    # 3D mesh states
    original_mesh: bytes = b""
    preprocessed_mesh: bytes = b""
    augmented_mesh: bytes = b""


data_state = DataState()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Text processing endpoints
@app.post("/upload/text")
async def upload_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        data_state.original_text = text
        return {"message": "Text file uploaded successfully", "sample": text[:500]}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Error processing text file: {str(e)}"},
        )


@app.get("/show-sample")
async def show_sample():
    if not data_state.original_text:
        return {"message": "No text loaded yet"}
    return {"sample": data_state.original_text[:500]}


@app.post("/preprocess/text")
async def preprocess_text_endpoint():
    if not data_state.original_text:
        return {"message": "No text loaded yet"}

    data_state.preprocessed_text = preprocess_text(data_state.original_text)
    return {"preprocessed_sample": data_state.preprocessed_text[:500]}


@app.post("/augment/text")
async def augment_text_endpoint():
    if not data_state.preprocessed_text:
        return {"message": "Please preprocess the text first"}

    data_state.augmented_text = augment_text(data_state.preprocessed_text)
    return {"augmented_sample": data_state.augmented_text[:500]}


# Image processing endpoints
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data_state.original_image = content
        return {
            "message": "Image uploaded successfully",
            "image": base64.b64encode(content).decode(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Error processing image file: {str(e)}"},
        )


@app.post("/preprocess/image")
async def preprocess_image_endpoint():
    if not data_state.original_image:
        return JSONResponse(status_code=400, content={"message": "No image loaded yet"})

    data_state.preprocessed_image = preprocess_image(data_state.original_image)
    return {
        "message": "Image preprocessed successfully",
        "image": base64.b64encode(data_state.preprocessed_image).decode(),
    }


@app.post("/augment/image")
async def augment_image_endpoint():
    if not data_state.preprocessed_image:
        return JSONResponse(
            status_code=400, content={"message": "Please preprocess the image first"}
        )

    data_state.augmented_image = augment_image(data_state.preprocessed_image)
    return {
        "message": "Image augmented successfully",
        "image": base64.b64encode(data_state.augmented_image).decode(),
    }


# Audio processing endpoints
@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # MP3와 WAV 모두 허용
        if not (
            file.filename.lower().endswith(".mp3")
            or file.filename.lower().endswith(".wav")
        ):
            return JSONResponse(
                status_code=400, content={"message": "Please upload an MP3 or WAV file"}
            )

        data_state.original_audio = content
        return {
            "message": "Audio uploaded successfully",
            "audio": base64.b64encode(content).decode(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Error processing audio file: {str(e)}"},
        )


@app.post("/preprocess/audio")
async def preprocess_audio_endpoint():
    if not data_state.original_audio:
        return JSONResponse(status_code=400, content={"message": "No audio loaded yet"})
    
    processed_audio, spectrogram = preprocess_audio(data_state.original_audio)
    data_state.preprocessed_audio = processed_audio
    
    return {
        "message": "Audio preprocessed successfully",
        "audio": base64.b64encode(processed_audio).decode(),
        "spectrogram": base64.b64encode(spectrogram).decode() if spectrogram else None
    }


@app.post("/augment/audio")
async def augment_audio_endpoint():
    if not data_state.preprocessed_audio:
        return JSONResponse(
            status_code=400, content={"message": "Please preprocess the audio first"}
        )

    augmented_audio, spectrogram = augment_audio(data_state.preprocessed_audio)
    data_state.augmented_audio = augmented_audio
    
    return {
        "message": "Audio augmented successfully",
        "audio": base64.b64encode(augmented_audio).decode(),
        "spectrogram": base64.b64encode(spectrogram).decode() if spectrogram else None
    }


# 3D mesh processing endpoints
@app.post("/upload/mesh")
async def upload_mesh(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not file.filename.lower().endswith('.off'):
            return JSONResponse(
                status_code=400,
                content={"message": "Please upload an OFF file"}
            )
            
        data_state.original_mesh = content
        processed_mesh, visualization = preprocess_mesh(content)
        return {
            "message": "Mesh uploaded successfully",
            "mesh": base64.b64encode(content).decode(),
            "visualization": base64.b64encode(visualization).decode() if visualization else None
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Error processing mesh file: {str(e)}"}
        )

@app.post("/preprocess/mesh")
async def preprocess_mesh_endpoint():
    if not data_state.original_mesh:
        return JSONResponse(status_code=400, content={"message": "No mesh loaded yet"})
    
    processed_mesh, visualization = preprocess_mesh(data_state.original_mesh)
    data_state.preprocessed_mesh = processed_mesh
    
    return {
        "message": "Mesh preprocessed successfully",
        "mesh": base64.b64encode(processed_mesh).decode(),
        "visualization": base64.b64encode(visualization).decode() if visualization else None
    }

@app.post("/augment/mesh")
async def augment_mesh_endpoint():
    if not data_state.preprocessed_mesh:
        return JSONResponse(
            status_code=400, content={"message": "Please preprocess the mesh first"}
        )
    
    augmented_mesh, visualization = augment_mesh(data_state.preprocessed_mesh)
    data_state.augmented_mesh = augmented_mesh
    
    return {
        "message": "Mesh augmented successfully",
        "mesh": base64.b64encode(augmented_mesh).decode(),
        "visualization": base64.b64encode(visualization).decode() if visualization else None
    }
