FROM python:3.9.18

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pdf_query.py .
COPY model_context.py .

CMD [ "python", "./pdf_query_gradio.py" ]