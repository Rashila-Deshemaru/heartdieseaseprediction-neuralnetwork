
import math
import pandas as pd
import numpy as np
from django.shortcuts import redirect, render



from .decorators import unauthenticated_user, allowed_users
from django.contrib.auth.decorators import login_required

from django.conf import settings







from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

#classification
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers

def home(request):
    return render(request, "index.html")


@login_required(login_url='login')

def predict_form(request):
    return render(request, 'predict_form.html')

def predict(request):
    if request.method == 'POST':
        temp = {}
        temp['ages'] = request.POST.get('ages')
        temp['sex'] = request.POST.get('sex')
        temp['cp'] = request.POST.get('cp')
        temp['trestbps'] = request.POST.get('trestbps')
        temp['chol'] = request.POST.get('chol')
        temp['fbs'] = request.POST.get('fbs')
        temp['restecg'] = request.POST.get('restecg')
        temp['thalach'] = request.POST.get('thalach')
        temp['exang'] = request.POST.get('exang')
        temp['oldpeak'] = request.POST.get('oldpeak')
        temp['slope'] = request.POST.get('slope')
        temp['ca'] = request.POST.get('ca')
        temp['thal'] = request.POST.get('thal')


        data = pd.read_csv('./model/heart.csv')

        
        # label_encoder = preprocessing.LabelEncoder()
        # data['target'] = label_encoder.fit_transform(
        #                                 data['target'])

      

        features = ['age', 'sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        target_col = ['target']
        df = data.fillna(method='ffill')
        x = df[features]
        y = df[target_col]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x)
        X = scaler.transform(x)

        Y = tf.keras.utils.to_categorical(y,2)
        X_train,X_test,Y_train,Y_test = train_test_split( X, Y, test_size=0.3)
        
        NB_CLASSES=2

        model = tf.keras.models.Sequential()

        model.add(keras.layers.Dense(128,         #Number of nodes
                                input_shape=(13,), #Number of input variables
                                name='Hidden-Layer-1', #Logical name
                                activation='relu'))    #activation function

        model.add(keras.layers.Dense(128,
                                    name='Hidden-Layer-2',
                                    activation='relu'))

        model.add(keras.layers.Dense(NB_CLASSES,
                                    name='Output-Layer',
                                    activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])     

        VERBOSE=1
        BATCH_SIZE=16
        EPOCHS=10
        VALIDATION_SPLIT=0.2          

        history=model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

        label_encoder = preprocessing.LabelEncoder() 


        prediction_input = [[
            temp['ages'], temp['sex'], temp['cp'], temp['trestbps'], temp['chol'],
                  temp['fbs'],temp['restecg'],temp['thalach'],
                  temp['exang'],float(temp['oldpeak']), temp['slope'],
                  temp['ca'],temp['thal']
        ]]    


        print(prediction_input)

        scaled_input = scaler.transform(prediction_input)

        raw_prediction = model.predict(scaled_input)                
        prediction = np.argmax(raw_prediction)

        # print(prediction)

        z = label_encoder.fit_transform([prediction])

        # print(z)

        output = label_encoder.inverse_transform(z)
        # print("OUTPUT  ",output)

        if output == 0:
            conclusion = "Heart Disease not Found"
        elif output == 1:
            conclusion = "Heart Disease Found"
        else: 
            conclusion = "Wrong Data Provided"

    return render(request, 'result.html', {'Disease': conclusion})


# Registering Users
from .forms import NewUserForm
from django.shortcuts import  render, redirect ,HttpResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)

            login(request, user)

            return redirect('index')
        else:
            return render(request, 'authentication/register.html', {'form': form})
    
    else:
        form = UserCreationForm()
        return render(request, "authentication/register.html", {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'authentication/login.html', {'form': form})
    
    else:
        form = AuthenticationForm()
        return render(request, 'authentication/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('index')

# appointment
from django.core.mail import send_mail

@login_required(login_url='login')
def appointment(request):
    return render(request, 'appointment.html')

@login_required(login_url='login')
def appointment_result(request):
    your_name = request.POST['your_name']
    your_phone = request.POST['your_phone']
    your_address = request.POST['your_address']
    your_email = request.POST['your_email']
    appointment_day = request.POST['appointment_day']
    appointment_time = request.POST['appointment_time']
    your_doctor = request.POST['your_doctor']
    your_message = request.POST['your_message']
 
    email_message = "Patient name: " + your_name + " Patient Phone number: " + your_phone + " Appointment day: " + appointment_day + " Appointment Time: " + appointment_time + " Patient's Remarks: " + your_message
    
    send_mail(
        your_name,
        email_message,
        your_email,
        ['ghosthunter5470@gmail.com', 'rashirashila2000@gmail.com']
    )
    return render(request, 'appointment_result.html', {
        'your_name': your_name,
        'your_phone': your_phone, 
        'your_address': your_address,
        'your_email' : your_email,
        'appointment_day' : appointment_day,
        'appointment_time' : appointment_time,
        'your_doctor' : your_doctor,
        'your_message' : your_message
        })



# display doctors

from . models import Doctor


def get_data(request):
    data = Doctor.objects.all()
    return render(request, 'doctor.html', {'data': data})
