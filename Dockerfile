FROM ubuntu:latest

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg

# Copy the entry script to the container
COPY entry_script.sh /usr/local/bin/entry_script.sh
RUN chmod +x /usr/local/bin/entry_script.sh

# Copy the folder to the container
COPY . /app/transcribe-app

# Set the working directory
WORKDIR /app

# Set the entry script as the entrypoint
ENTRYPOINT ["/usr/local/bin/entry_script.sh"]