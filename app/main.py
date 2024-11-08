from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import io
from .preprocessing import preprocess_text
from .augmentation import augment_text
from .image_processing import preprocess_image
from .image_augmentation import augment_image
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
