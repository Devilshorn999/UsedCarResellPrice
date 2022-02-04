@REM install python using cmd

@REM cd C:\Downloads\Programs\
@REM python-3.9.2-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
@REM python -m ensurepip

@REM    THis is to create a venv named site_venv
python -m venv site_venv
.\site_venv\scripts\activate

@REM  This is to install folowing lib into venv #python
REM cd C:\Project_ML\site_venv\Scripts


pip3 install --upgrade pip3
pip3 install django
pip3 install scikit-learn
pip3 install pandas
pip3 install xgboost
pip3 install gunicorn
pip3 install django-heroku

REM cd C:\Project_ML
pip3 freeze > requirements.txt
