from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
from .preprocessing import preprocess_text
from .augmentation import augment_text

app = FastAPI(title="Text Processing API")

# HTML 템플릿을 위한 Jinja2 설정
templates = Jinja2Templates(directory="app/templates")


# Store the current state of the text
class TextState:
    original_text: str = ""
    preprocessed_text: str = ""
    augmented_text: str = ""


text_state = TextState()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        text_state.original_text = text

        return {"message": "File uploaded successfully", "sample": text[:500]}
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"message": f"Error processing file: {str(e)}"}
        )


@app.get("/show-sample")
async def show_sample():
    if not text_state.original_text:
        return {"message": "No text loaded yet"}
    return {"sample": text_state.original_text[:500]}


@app.post("/preprocess")
async def preprocess():
    if not text_state.original_text:
        return {"message": "No text loaded yet"}

    text_state.preprocessed_text = preprocess_text(text_state.original_text)
    return {"preprocessed_sample": text_state.preprocessed_text[:500]}


@app.post("/augment")
async def augment():
    if not text_state.preprocessed_text:
        return {"message": "Please preprocess the text first"}

    text_state.augmented_text = augment_text(text_state.preprocessed_text)
    return {"augmented_sample": text_state.augmented_text[:500]}
