# Project description
This project aims at performing the task of image classification with the ResNet-50 deep neural network, through hosting the model in a Docker container and interacting with it via API calls through FastAPI.

## Requirements
- Python 3.11
- Docker
- Torch
- Torchvision
- FastAPI
- Uvicorn
- Pydantic

## Installation

Clone this repository and navigate to the project directory via the GitHub CLI command shown below:

```bash
gh repo clone https://github.com/Loganpd/Portfolio.git 'Dockerized Image Classification'
```

Build the Docker image:

```bash
docker build -t fastapi-image-classification .
```

Run the Docker container:

```bash
docker run -p 8000:8000 fastapi-image-classification
```

## Usage

The application exposes a REST API that accepts JSON requests and returns JSON responses. The request should contain a singular field:
- url: a URL to the image that is intended to be classified by the ResNet50 model.

The response will contain the **top 3** most probable classes detected by the model.
