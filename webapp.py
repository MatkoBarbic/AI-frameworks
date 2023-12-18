import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

def recommend_movie(image, radio, description):
    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Send request to the API
    response = requests.post("http://annoy-db:5000/poster_predict", data=img_binary.getvalue())
    recommended_titles = response.json()["recommendations"]

    recommended_posters = []
    for path in recommended_titles:
        recommended_posters.append(np.array(Image.open(path)))
    

    ## Description recommendations
    response = requests.post("http://annoy-db:5000/description_predict", data={"radio": radio, "description": description})
    recommended_descriptions = list(response.json()["recommendations"])
    
    return recommended_posters + recommended_descriptions

if __name__=='__main__':

    radio_input = gr.inputs.Radio(["Bag of words", "BARD"], label="Choose embedding technique:")
    text_input = gr.inputs.Textbox(max_lines=100, placeholder="Enter Description Here", label="Description of movie")

    gr.Interface(fn=recommend_movie, 
                inputs=["image", radio_input, text_input], 
                outputs=[['image' for _ in range(5)] + [gr.outputs.Textbox() for _ in range (5)]],
                live=True,
                description="Upload the movie poster and description.",
                ).launch(server_name="0.0.0.0", debug=True, share=True, server_port=7860)
    