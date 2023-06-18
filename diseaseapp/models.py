from django.db import models

from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser



# customer model
# class Patient(models.Model):
    
#     user = models.OneToOneField(User, null = True, blank = True, on_delete = models.CASCADE)
#     name = models.CharField(max_length=200, null=True)
#     email = models.CharField(max_length=200)

#     def __str__(self):
#         return self.name


class Doctor(models.Model):
    name = models.CharField(max_length=200, null=True)
    email = models.CharField(max_length=200 , null = False)
    image = models.ImageField(upload_to='Doctor' , null=True)
    hospital_name = models.CharField(max_length=200, null=True)
    specialist = models.CharField(max_length=200, null=True)

    def __str__(self):
        return str(self.name)
