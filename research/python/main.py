from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
async def test():
    return {"message": "Hello World"}


@app.post("/uploadfile/")
async def upload(file: UploadFile = File(...)):
    return {"filename": file.filename}