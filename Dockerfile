# Use a Miniconda base image for a lean Conda environment
FROM --platform=linux/amd64 conda/miniconda3:latest

WORKDIR /app

# Copy the environment file and create the Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Set the shell to run commands within our Conda environment
SHELL ["conda", "run", "-n", "mlops-env", "/bin/bash", "-c"]

# Copy the pre-trained model and the prediction script into the image
# These will be provided by our CI/CD pipeline later
COPY sentiment_model.pkl .
COPY predict.py .

# Define the command to run when the container starts
ENTRYPOINT ["conda", "run", "-n", "mlops-env", "python", "predict.py"]

# FROM --platform=$BUILDPLATFORM conda/miniconda3:latest

# WORKDIR /app

# Copy environment file and create environment
# COPY environment.yml .
# RUN conda env create -f environment.yml

# Copy application files
# COPY sentiment_model.pkl .
# COPY predict.py .

# Define the entrypoint for the container
# This calls the python interpreter directly from the created environment
# ENTRYPOINT ["/opt/conda/envs/mlops-env/bin/python", "predict.py"]

# The user's input on "docker run" will be passed as arguments to the entrypoint
