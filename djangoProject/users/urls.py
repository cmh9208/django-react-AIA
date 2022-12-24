from django.urls import re_path as url
from users import views

urlpatterns = [
    url(r'users', views.users),
]
