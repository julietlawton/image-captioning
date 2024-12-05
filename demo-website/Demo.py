import streamlit as st
from PIL import Image
import os
import model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Image Captioning with Visual Attention Demo")

option_map = {
    0: "Greedy",
    1: "Beam Search",
}

search_strategy_selection = st.pills(
    "Prediction Strategy:",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
    default=0,
    help="Determines search strategy for predicting a caption. Greedy search predicts the most probable word at each step, while beam search considers multiple candidates."
)

image_file = st.file_uploader(
    "Choose an image:", 
    type=['jpg'], 
    accept_multiple_files=False, 
    help="Upload the image you want to test.", 
    on_change=None,
    label_visibility="visible"
)

if image_file is not None:
    temp_dir = os.path.join(ROOT_DIR, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_file_path = os.path.join(temp_dir, image_file.name)

    with open(temp_file_path, "wb") as f: 
        f.write(image_file.getbuffer())         
    
    uploaded_img = Image.open(temp_file_path)
    st.image(uploaded_img, width=800, channels="BGR", use_container_width="always")
    st.success("Success")

    st.markdown("**Predicted caption:**")
    with st.spinner('Predicting caption...'):
        if search_strategy_selection == 0:  
            prediction, attention_maps = model.greedy_predict(uploaded_img)
            st.text(prediction)
        else:
            prediction, attention_maps = model.beamsearch_predict(uploaded_img, 3)
            st.text(prediction)

    if st.checkbox('Show attention maps'):
        with st.spinner('Getting attention maps...'):
            fig = model.show_attention_maps(attention_maps, uploaded_img, prediction)
            st.pyplot(fig)

else:
    file = os.path.join(ROOT_DIR, "images", "test_dog.jpg")
    img = Image.open(file)
    st.image(img, width=800, channels="BGR", use_container_width="always")
    st.markdown("### Predicted caption:")
    st.text("a dog runs through a grassy field")