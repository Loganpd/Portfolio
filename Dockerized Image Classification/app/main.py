from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
import torch


# defining custom data models
class ImageUrlRequest(BaseModel):
    url: HttpUrl


# Initialize the FastAPI app
app = FastAPI()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load the trained PyTorch model
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
resnet50.eval().to(device)

@app.post("/classify/")
async def predict(request: ImageUrlRequest):
    """Receives an image url, processes it, and returns the model's prediction."""
    try:
        response = requests.get(request.url)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Apply transformations and make a prediction
    with torch.no_grad():
        output = torch.nn.functional.softmax(resnet50(utils.prepare_input_from_uri(request.url.__str__()).to(device)), dim=1)
        results = utils.pick_n_best(predictions=output, n=3)
    return {"prediction": results}

@app.get("/")
def read_root():
    message = ('Welcome to the PyTorch model API! '
               'You can use the curl command with a post method to provide an image URL to the container. '
               'Then the container carries out image classification and returns the top 3 possible classes for the image!')
    return {"message": f"{message}"}