# Set base image (host OS)
FROM python:3.9-slim-buster

# Set author field of the generated images
LABEL maintainer="Hoang (Justinianus) Le Ngoc (lengochoang681@gmail.com)"

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy all files and directories from the local src directory to the working directory
COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install any dependencies
RUN pip install -r requirements.txt

# Specify the command to run on container start
CMD [ "python", "./app.py" ]