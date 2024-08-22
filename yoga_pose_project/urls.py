from django.contrib import admin
from django.urls import path
from yoga_pose_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
]