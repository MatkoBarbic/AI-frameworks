import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

def recommend_movie(image):
    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Send request to the API
    response = requests.post("http://annoy-db:5000/predict", data=img_binary.getvalue())
    recommended_titles = response.json()["recommendations"]

    recommended_movies = []
    for path in recommended_titles:
        recommended_movies.append(np.array(Image.open(path)))
    
    return recommended_movies

if __name__=='__main__':

    gr.Interface(fn=recommend_movie, 
                inputs="image", 
                outputs=['image' for i in range(5)],
                live=True,
                description="Upload the movie poster.",
                ).launch(server_name="0.0.0.0", debug=True, share=True, server_port=7860)
    