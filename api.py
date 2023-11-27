import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import model
from annoy import AnnoyIndex
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
dim = 576
k = 5

df = pd.read_csv("./movie_poster_paths.csv")
df = df.apply(lambda x: "./" + x)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Get embeddings
    with torch.no_grad():
        query_vector = model(tensor)[0]

    annoy_index = AnnoyIndex(dim, 'angular')
    annoy_index.load('./rec_imdb.ann')

    indices = annoy_index.get_nns_by_vector(query_vector, k)

    recommended_paths = list(df.iloc[indices]["path"])

    # recommendations_for_current_image = []
    # for recommended_path in recommended_paths:
    #     recommended_img_pil = Image.open(recommended_path)
    #     recommended_img_data = io.BytesIO()
    #     recommended_img_pil.save(recommended_img_data, format='JPEG')
    #     recommended_img_data.seek(0)
    #     recommendations_for_current_image.append(recommended_img_data.read())

    return jsonify({"recommendations": recommended_paths})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)