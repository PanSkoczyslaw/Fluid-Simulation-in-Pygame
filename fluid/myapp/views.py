from django.shortcuts import render
from django.utils.timezone import now

def home(request):
    return render(request, 'myapp/home.html', {
        'timestamp': int(now().timestamp())
    })
