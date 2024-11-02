# # set python version
# FROM python:3.10

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any dependencies specified in requirements.txt
# RUN pip install --upgrade pip
# RUN pip install chainlit langchain langchain_huggingface langchain_ollama chromadb torch sentence-transformers

# # Set environment variables
# ENV LANCHAIN_TRACING=False

# # expose port 8000
# EXPOSE 8000

# # Run the app
# CMD python chainlit run app.py -p 8000


# Set Python version
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install chainlit langchain langchain_huggingface langchain_ollama chromadb torch sentence-transformers

# Install dependencies for Ollama
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/download.sh | bash

# Pull the llama3.2 model using Ollama
RUN ollama pull llama3.2

# Set environment variables
ENV LANCHAIN_TRACING=False

# Expose port 8000
EXPOSE 8000

# Run the app
CMD ["python", "-m", "chainlit", "run", "app.py", "-p", "8000"]
