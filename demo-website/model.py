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

def greedy_predict(image):
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

def beamsearch_predict(image, num_beams):

    image = preprocess_input_densenet(
        tf.expand_dims(
            tf.image.resize(image, [224, 224]), 
            axis=0
        )
    )

    image_features = densenet121(image)

    # Set the start and end sequence tokens and initialize candidate sequences
    start_token = tokenizer(['startseq']).numpy()[0][0]
    end_token =  tokenizer(['endseq']).numpy()[0][0]
    sequences = [[start_token]]
    scores = [0.0]

    max_caption_length = 40
    # Run until sequences maximum length or all have an end token
    for _ in range(max_caption_length - 1):
      all_candidates = []
      for seq, score in zip(sequences, scores):

        # If this sequence is already complete, move to the next one
        if seq[-1] == end_token:
          all_candidates.append((seq, score))
          continue

        # Generate predictions for the next token in the sequence
        tokens = tf.constant([seq + [0] * (max_caption_length - len(seq))])
        preds = captioning_model((image_features, tokens)).numpy()

        preds = preds[:, len(seq) - 1, :]

        # Convert the raw logits to probabilities
        probs = tf.nn.softmax(preds[0])
        log_probs = tf.math.log(probs)

        # Take the k tokens with the highest probabilities
        top_k = tf.math.top_k(log_probs, k=num_beams)
        top_k_tokens = top_k.indices.numpy()
        top_k_probs = top_k.values.numpy()

        # For each of the top k tokens, build out a candidate sequence using
        # that token and add its probability to the overall score
        for token, prob in zip(top_k_tokens, top_k_probs):
          candidate_seq = seq + [token]
          candidate_score = score + prob
          all_candidates.append((candidate_seq, candidate_score))

        # Sort the candidate sequences by their joint probability scores
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)

        # Prune the sequences to keep only the top k beam
        sequences, scores = zip(*ordered[:num_beams])

        # If all of the sequences are complete, break out of the loop
        if all(seq[-1] == end_token for seq in sequences):
          break

    # Take the sequence with the highest score
    best_sequence = sequences[0]

    # Decode the tokens
    words = [index_to_word[token] for token in best_sequence if token not in [0, start_token, end_token]]
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

    cols = 3
    rows = int(np.ceil(len(tokens) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), constrained_layout=True)

    axes = axes.ravel()

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

    for i in range(len(tokens), len(axes)):
        axes[i].axis("off")
    
    return fig
