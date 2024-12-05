import tensorflow as tf
import numpy as np
import os
import einops
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
import model_definition

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(ROOT_DIR, "model", "visual_attention_model_mobilenet_v1.keras")

# Load model
captioning_model = tf.keras.models.load_model(model_path, compile=False)

densenet121 = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
)

# Load tokenizer
tokenizer = captioning_model.tokenizer
index_to_word = {i: w for i, w in enumerate(tokenizer.get_vocabulary())}

def predict(image):
    image = preprocess_input_densenet(
        tf.expand_dims(
            tf.image.resize(image, [224, 224]), 
            axis=0
        )
    )

    image_features = densenet121(image)

    start_token = tokenizer(['startseq']).numpy()[0][0]
    end_token = tokenizer(['endseq']).numpy()[0][0]

    max_caption_length = 30
    tokens = tf.constant([[start_token] + [0] * (max_caption_length - 1)])

    for n in range(max_caption_length - 1):
        # Predict the next token based on the sequence so far
        preds = captioning_model((image_features, tokens)).numpy()
        preds = preds[:, n, :]

        # Take the token with the highest logits as the next token
        next_token = tf.argmax(preds, axis=-1).numpy()[0]

        # If the token predicted was the end sequence token, end here
        if next_token == end_token:
            break

        # Add the token to the sequence
        tokens = tokens.numpy()
        tokens[0, n + 1] = next_token
        tokens = tf.constant(tokens)

    # Decode the tokens in the caption
    words = [index_to_word[token] for token in tokens.numpy()[0] if token not in [0, start_token, end_token]]
    caption = ' '.join(words)

    return caption, [layer.last_attention_scores for layer in captioning_model.decoder_block]


def show_attention_maps(attention_maps, image, caption):
    # Prepare the image
    img_tensor = tf.convert_to_tensor(image)
    resized_image = tf.image.resize(img_tensor, [224, 224])/255

    # Concatenate all of the attention maps
    attn_maps = tf.concat(attention_maps, axis=0)

    # Peform mean reduction to get a single attention map that averages
    # all attention maps for the sequence
    attn_maps = einops.reduce(
        attn_maps,
        'batch heads sequence (height width) -> sequence height width',
        height=7, width=7,
        reduction='mean'
    )

    # Split the caption into tokens
    tokens = caption.split()

    # fig = plt.figure(figsize=(25, 10))
    # grid_size = max(int(np.ceil(len(tokens)/4)), 4)
    # num_tokens = len(tokens)
    # rows = 4
    # cols = int(np.ceil(num_tokens / rows)) + 1

    num_tokens = len(tokens)
    cols = 3
    rows = int(np.ceil(len(tokens) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), constrained_layout=True)

    axes = axes.ravel()

    # titles = []

    for i in range(len(tokens)):
        # Get the averaged attention map for this token
        map = attn_maps[i]

        # Plot the image
        img = axes[i].imshow(resized_image)

        # Plot the attention map as a transparent heatmap over the image
        axes[i].imshow(map, cmap='gray', alpha=0.7, extent=img.get_extent(), clim=[0.0, np.max(map)])

        # Set the title and remove axes
        axes[i].set_title(tokens[i], fontsize=30)
        axes[i].set_axis_off()

    #     # Label this subplot with its token
    #     ax = fig.add_subplot(rows, cols, i+1)
    #     titles.append(ax.set_title(tokens[i]))

    #     # Plot the image
    #     img = ax.imshow(resized_image)

    #     # Plot the attention map as a transparent heatmap over the image
    #     ax.imshow(map,  cmap='gray', alpha=0.7, extent=img.get_extent(), clim=[0.0, np.max(map)])
    #     ax.set_axis_off()
    # fig.tight_layout()

    for i in range(len(tokens), len(axes)):
        axes[i].axis("off")
    
    return fig
