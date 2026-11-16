from typing import Literal
from datetime import date as CalendarDate
from fastapi import APIRouter,HTTPException,Query,Request
from services.ml_catalogue import CatalogueUnavailable

router=APIRouter(prefix='/ml',tags=['experimental MLP'])

def catalogue(request):
    cache=request.app.state.ml_catalogue
    try:cache.ready()
    except CatalogueUnavailable as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
    return cache

@router.get('/model')
def model(request:Request):return catalogue(request).model()

@router.get('/scenes')
def scenes(request:Request,date:CalendarDate|None=Query(default=None)):
    cache=catalogue(request)
    items=cache.list_scenes(date.isoformat() if date else None)
    return {'model_version':cache.model()['version'],'scenes':items,'count':len(items)}

@router.get('/scenes/{scene_id}')
def scene(scene_id:str,request:Request):
    try:return catalogue(request).scene(scene_id)
    except KeyError as exc:raise HTTPException(status_code=404,detail='Scene is unavailable; choose a processed scene.') from exc

@router.get('/scenes/{scene_id}/predictions')
def predictions(scene_id:str,request:Request,layer:Literal['all','mlp','modis','viirs','difference']='all'):
    try:return catalogue(request).predictions(scene_id,layer)
    except KeyError as exc:raise HTTPException(status_code=404,detail='Scene is unavailable; choose a processed scene.') from exc

@router.get('/scenes/{scene_id}/coverage')
def coverage(scene_id:str,request:Request):
    try:return catalogue(request).coverage(scene_id)
    except KeyError as exc:raise HTTPException(status_code=404,detail='Coverage is unavailable for this scene.') from exc
