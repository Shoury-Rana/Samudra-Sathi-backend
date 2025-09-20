from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.gis.db import models as gis_models
from django.contrib.gis.forms import OSMWidget
from .models import User

class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'is_staff')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('email', 'first_name', 'last_name',)
    ordering = ('email',)

    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'location')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'location', 'password1', 'password2'),
        }),
    )

    formfield_overrides = {
        gis_models.PointField: {"widget": OSMWidget(attrs={
            'default_lon': 80,
            'default_lat': 22,
            'default_zoom': 4,
        })}
    }

admin.site.register(User, CustomUserAdmin)
