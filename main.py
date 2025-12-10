from fastapi import FastAPI

from api.router import router as detect_router

app = FastAPI()
app.include_router(detect_router)
