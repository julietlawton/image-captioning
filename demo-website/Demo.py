import streamlit as st
from PIL import Image
import os
import random
import model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Image Captioning with Visual Attention Demo")

search_option_map = {
    0: "Greedy",
    1: "Beam Search",
}

search_strategy_selection = st.pills(
    "Prediction strategy:",
    options=search_option_map.keys(),
    format_func=lambda option: search_option_map[option],
    selection_mode="single",
    default=0,
    help="Determines search strategy for predicting a caption. Greedy search predicts the most probable word at each step, while beam search considers multiple candidates."
)

if search_strategy_selection == 1:
    num_beams_option_map = {
    0: "2",
    1: "3",
    2: "4"
    }

    num_beams = st.pills(
        "Number of beams:",
        options=num_beams_option_map.keys(),
        format_func=lambda option: num_beams_option_map[option],
        selection_mode="single",
        default=0,
        help="Number of beams for search."
    )

if "image_index" not in st.session_state:
    random_index = random.choice(range(1, 20))
    st.session_state["image_index"] = random_index

def increment_image_index():
    if st.session_state["image_index"] + 1 > 20:
        st.session_state["image_index"] = 1
    else:
        st.session_state["image_index"] += 1

def select_random_image():
    index = st.session_state["image_index"]
    demo_image_path = os.path.join(ROOT_DIR, "demo-images", f"demo_img{index}.jpg")
    st.session_state["image_file"] = demo_image_path
    st.session_state["source"] = "random"
    increment_image_index()

def clear_uploaded_image():
    if "source" in st.session_state and st.session_state["source"] == "upload":
        del st.session_state["image_file"]
        del st.session_state["source"]

uploaded_file = st.file_uploader(
    "Choose an image:", 
    type=['jpg'], 
    accept_multiple_files=False, 
    help="Upload the image you want to test.", 
    on_change=clear_uploaded_image,
    label_visibility="visible",
)

if uploaded_file is not None:
    temp_dir = os.path.join(ROOT_DIR, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f: 
        f.write(uploaded_file.getbuffer())

    st.session_state["image_file"] = temp_file_path
    st.session_state["source"] = "upload"

st.markdown("**Or**")
if st.button("Pick a random image", help="Selects a random image from the validation set the model was evaluated on."):
    select_random_image()

if "image_file" not in st.session_state or st.session_state["image_file"] is None:
    placeholder_path = os.path.join(ROOT_DIR, "placeholder.jpg")
    img = Image.open(placeholder_path)
    st.image(img, width=800, channels="BGR", use_container_width="always")
    st.markdown("### Predicted caption:")
    st.text("a dog runs through a grassy field")
else:
    current_image_path = st.session_state["image_file"]
    img = Image.open(current_image_path)
    st.image(img, width=800, channels="BGR", use_container_width="always")

    st.markdown("### Predicted caption:")
    with st.spinner('Predicting caption...'):
        if st.session_state["source"] == "upload":
            if search_strategy_selection == 0:
                prediction, attention_maps = model.greedy_predict(img)
            else:
                k = int(num_beams_option_map[num_beams])
                prediction, attention_maps = model.beamsearch_predict(img, k)
        elif st.session_state["source"] == "random":
            if search_strategy_selection == 0:
                prediction, attention_maps = model.greedy_predict(img)
            else:
                k = int(num_beams_option_map[num_beams])
                prediction, attention_maps = model.beamsearch_predict(img, k)

        st.text(prediction)


    if st.checkbox('Show attention maps', help="Shows where the model \"looked\" at an image when predicting a word in the caption."):
        with st.spinner('Getting attention maps...'):
            fig = model.show_attention_maps(attention_maps, img, prediction)
            st.pyplot(fig)
