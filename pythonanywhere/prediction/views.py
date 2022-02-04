from django.shortcuts import render, redirect
import joblib


# model = joblib.load('model_rf')
# unique_vals = joblib.load('unique_vals')
# encoders = joblib.load('encoders_')

model = joblib.load('/home/DevilsHorn999/MyProject1_deployable/model_rf')
unique_vals = joblib.load('/home/DevilsHorn999/MyProject1_deployable/unique_vals')
encoders = joblib.load('/home/DevilsHorn999/MyProject1_deployable/encoders_')
print(model)
print(unique_vals)
print(encoders)

def test(request):
    if request.method == 'POST':
        print(request.POST.keys())
        city1 = request.POST.get('city')
        city = int(encoders['city'].transform([city1]))
        year1 = float(request.POST.get('year'))
        year = int(encoders['year'].transform([year1]))
        manu = request.POST.get('manufacturer')
        manufaturer = int(encoders['manufacturer'].transform([manu]))
        make1 = request.POST.get('make')
        make = int(encoders['make'].transform([make1]))
        condition1 = request.POST.get('condition')
        condition = int(encoders['condition'].transform([condition1]))
        cylin = request.POST.get('cylinders')
        cylinders = int(encoders['cylinders'].transform([cylin]))
        fuel1 = request.POST.get('fuel')
        fuel = int(encoders['fuel'].transform([fuel1]))
        odometer = float(request.POST.get('odometer'))
        title1 = request.POST.get('title')
        title = int(encoders['title_status'].transform([title1]))
        trans = request.POST.get('transmission')
        transmission = int(encoders['transmission'].transform([trans]))
        drive1 = request.POST.get('drive')
        drive = int(encoders['drive'].transform([drive1]))
        typeof1 = request.POST.get('type')
        typeof = int(encoders['type'].transform([typeof1]))
        painr1 = request.POST.get('color')
        paint = int(encoders['paint_color'].transform([painr1]))
        name = request.POST.get('make') + '-' + manu + " of type " + typeof1 + " is"
        vals_ = [painr1, year1, manu, make1, cylin, drive1, typeof1, trans, city1, condition1
            , fuel1, odometer, title1]
        vals = [city, year, manufaturer, make, condition, cylinders, fuel, odometer, title, transmission, drive, typeof,
                paint, name]
        return price(request, vals, vals_)
    else:
        locations = unique_vals.city
        years = unique_vals.year
        manu = unique_vals.manufacturer
        mods = unique_vals.make
        conditions = unique_vals.condition
        cylinders = unique_vals.cylinders
        fuels = unique_vals.fuel
        titles = unique_vals.title_status
        transmissions = unique_vals.transmission
        drives = unique_vals.drives
        types = unique_vals.type
        colors = unique_vals.paint_color
        return render(request, 'index.html', {
            'locations': locations,
            'years': years,
            'manu': manu,
            'mods': mods,
            'conditions': conditions,
            'cylinders': cylinders,
            'fuels': fuels,
            'titles': titles,
            'transmissions': transmissions,
            'drives': drives,
            'types': types,
            'colors': colors
        })


def price(request, vals, vals_):
    name = vals.pop(-1)
    return render(request, 'price.html',
                  {'price': round(float(model.predict([vals])), 2), 'name': name.title(), 'vals': [vals_]})
