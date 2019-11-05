from django.shortcuts import render
from django.views import View
# Create your views here.
import pickle
import numpy as np
from django.conf import settings

class IndexView(View):

    def get(self, request):
        show = False
        flag = True
        dt = {'da': '','fag': flag,  'show': show}
        return render(request, 'index.html', dt)

    def post(self, request):
        filemodel = settings.BASE_DIR + '/core/mod.plk'
        print(filemodel)

        air_pressure_9am = float(request.POST.get('air_pressure_9am', 0))
        air_temp_9am = float(request.POST.get('air_temp_9am', 0))
        avg_wind_direction_9am= float(request.POST.get('avg_wind_direction_9am', 0))
        avg_wind_speed_9am= float(request.POST.get('avg_wind_speed_9am', 0))
        max_wind_direction_9am= float(request.POST.get('max_wind_direction_9am', 0))
        max_wind_speed_9am= float(request.POST.get('max_wind_speed_9am', 0))
        rain_accumulation_9am= float(request.POST.get('rain_accumulation_9am', 0))
        rain_duration_9am= float(request.POST.get('rain_duration_9am', 0))
        relative_humidity_9am= float(request.POST.get('air_temp_9am', 0))

        with open(filemodel, 'rb') as f:
            model_classify = pickle.load(f)
            ar = np.array([[air_pressure_9am, air_temp_9am, avg_wind_direction_9am,
                            avg_wind_speed_9am, max_wind_direction_9am, max_wind_speed_9am,
                            rain_accumulation_9am, rain_duration_9am]])
            # dat = ar.reshape(-1, 1)
            result = model_classify.predict(ar)
            print(result)
            show = True
            if result[0] == 0:
                kq = 'Không mưa'
                flag = False
            else:
                flag = True
                kq = 'Có mưa'

        dt = {'da': kq, 'fag': flag, 'show': show}
        return render(request, 'index.html', dt)


