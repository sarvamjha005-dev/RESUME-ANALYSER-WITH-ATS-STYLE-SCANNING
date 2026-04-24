FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

#  Add this
RUN python -m spacy download en_core_web_sm

EXPOSE 6996

CMD ["uvicorn", "km:app", "--host", "0.0.0.0", "--port", "6996"]

