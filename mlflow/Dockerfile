FROM python:3.11-slim

# Pinned to the same version uv.lock resolves for train.py and the inference image,
# so the tracking server and every client speak the same REST/artifact API.
RUN pip install --no-cache-dir mlflow==3.16.1

# Mount point for the named volume; created here too so the image also runs standalone.
RUN mkdir -p /mlflow-data

EXPOSE 5000

# --artifacts-destination (not --default-artifact-root): artifacts land on the volume but are
# proxied over HTTP (mlflow-artifacts:/ URIs), so the inference container downloads them from
# the server instead of trying to open a file:// path that only exists inside this container
# (the same "No such artifact" problem Lab 3 hit with a host-local artifact root).
# --allowed-hosts: mlflow 3's host-header middleware rejects unknown hosts, "mlflow:5000" included.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]
