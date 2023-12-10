# import argparse

# import uvicorn
# from fastapi import APIRouter, FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from .routers import models, predict


# def api(args: argparse.Namespace) -> FastAPI:
#     app = FastAPI(
#         title="PubMed Research Trends",
#         description="PubMed Research Trends is a web application that allows users to explore trends in biomedical research. Users can search for topics and visualize the trends in the number of publications over time. The application also provides a topic clustering feature that allows users to explore the topics that are related to a given topic.",
#         version="0.1.0",
#     )

#     for (
import os

print(os.path.abspath(os.path.dirname(__file__)))
import json
import socket

json.loads(
