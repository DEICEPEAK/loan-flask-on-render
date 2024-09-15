FROM python:3.10
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
COPY requirements.txt /usr/app/
RUN pip install --default-timeout=1000 -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_loan:app"]