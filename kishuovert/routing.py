from django.urls import re_path 
from . import backend

websocket_urlpatterns = [
    re_path(r'ws/socket-server/', backend.BackendSocket.as_asgi())
]