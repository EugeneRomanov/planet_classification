import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from src.containers.conainers import AppContainer
from src.routes.routers import router
from src.services.planet_classifier import PlanetClassifier


@router.get('/planets')
@inject
def planets_list(service: PlanetClassifier = Depends(Provide[AppContainer.planet_classifier])):
    return {
        'planets': service.planets,
    }


@router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    service: PlanetClassifier = Depends(Provide[AppContainer.planet_classifier]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    planets = service.predict(img)

    return {'planets': planets}


@router.post('/predict_proba')
@inject
def predict_proba(
    image: bytes = File(),
    service: PlanetClassifier = Depends(Provide[AppContainer.planet_classifier]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict_proba(img)
