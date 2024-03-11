import argparse

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware


def api(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(
        title="My Title",
        description="My Description",
        version="0.1.0",
    )

    app.add_middleware(
