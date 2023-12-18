import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from posters_model import model as poster_model
from bard import model as text_model
from annoy import AnnoyIndex
import pandas as pd
from transformers import DistilBertTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
dim = 576
k = 5

posters_df = pd.read_csv("./movie_poster_paths.csv")
posters_df = posters_df.apply(lambda x: "./" + x)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

text_df = pd.read_csv("./movies_metadata.csv")

@app.route('/poster_predict', methods=['POST'])
def poster_predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Get embeddings
    with torch.no_grad():
        query_vector = poster_model(tensor)[0]

    annoy_index = AnnoyIndex(dim, 'angular')
    annoy_index.load('./rec_imdb.ann')

    indices = annoy_index.get_nns_by_vector(query_vector, k)

    recommended_paths = list(posters_df.iloc[indices]["path"])

    return jsonify({"recommendations": recommended_paths})


@app.route('/description_predict', methods=['POST'])
def description_predict():
    description = request.data["description"]
    radio = request.data["radio"]
    
    tensor = transform(description).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    if radio == "Bag of words":
        with torch.no_grad():
            query_vector = bow(tensor)[0]

        annoy_index = AnnoyIndex(dim, 'angular')
        annoy_index.load('./description_embeddings_bow.ann')
    
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        tensor = tokenizer(tensor)
        with torch.no_grad():
            query_vector = text_model(tensor)[0]

        annoy_index = AnnoyIndex(dim, 'angular')
        annoy_index.load('./description_embeddings_bard.ann')

    indices = annoy_index.get_nns_by_vector(query_vector, k)

    recommended_titles = list(text_df.iloc[indices]["title"])

    return jsonify({"recommendations": recommended_titles})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)