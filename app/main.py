from fastapi import FastAPI

from app.api.routes.crisis import router as crisis_router
from app.api.routes.emotion import router as emotion_router
from app.api.routes.response import router as response_router

app = FastAPI(title="MindGuard API", version="0.1.0")

app.include_router(emotion_router, prefix="/predict", tags=["emotion"])
app.include_router(crisis_router, prefix="/predict", tags=["crisis"])
app.include_router(response_router, prefix="/generate", tags=["response"])
