FROM python:3.9

WORKDIR /diango_web

COPY . .
COPY requirement.txt requirements.txt

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
