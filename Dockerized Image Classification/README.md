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

Example request:

```bash
curl -X 'POST' \
  'http://localhost:8000/classify/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "https://raw.githubusercontent.com/Loganpd/Portfolio/refs/heads/main/Dockerized%20Image%20Classification/images/img1.jpg"
}'
```

Resulting response:
{
  "prediction": [
    [
      [
        "hotdog, hot dog, red hot",
        "85.1%"
      ],
      [
        "strawberry",
        "0.1%"
      ],
      [
        "corkscrew, bottle screw",
        "0.1%"
      ]
    ]
  ]
}

You can use other images provided in this repository, or any other image link on the web for the automatic classification procedure.
