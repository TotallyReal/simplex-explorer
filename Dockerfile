FROM python:3.13-slim

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        texlive-latex-recommended \
        ghostscript \
        imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8050
EXPOSE 8050

CMD ["python", "app.py"]
