import tensorflow as tf
import keras

@tf.keras.utils.register_keras_serializable()
# Define a transformer decoder layer for the visual attention model
class TransformerDecoder(tf.keras.Layer):
  def __init__(self, units, num_heads=1, dropout=0.0):
    super().__init__()
    # Self attention layer
    self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)

    # Cross attention layer
    self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)

    # Feed forward network
    self.feed_forward = tf.keras.Sequential([
      tf.keras.layers.Dense(units=2*units, activation='relu'),
      tf.keras.layers.Dropout(rate=dropout),
      tf.keras.layers.LayerNormalization(),
      tf.keras.layers.Dense(units=units),
    ])

    # Normalization layers
    self.self_attention_norm = tf.keras.layers.LayerNormalization()
    self.cross_attention_norm = tf.keras.layers.LayerNormalization()
    self.feed_forward_norm = tf.keras.layers.LayerNormalization()

    # Residual connection layers
    self.self_attention_res = tf.keras.layers.Add()
    self.cross_attention_res = tf.keras.layers.Add()

  def build(self, input_shape):
    super().build(input_shape)

  def call(self, inputs, training=False):
    image, seq_emb = inputs

    # Perform self-attention on the caption (causal mask prevents future tokens
    # from being attended to by past tokens)
    self_attn = self.self_attention(query=seq_emb, value=seq_emb, use_causal_mask=True)
    seq_emb = self.self_attention_res([seq_emb, self_attn])
    seq_emb = self.self_attention_norm(seq_emb)

    # Peform cross-attention using the captions as the query and the image as the key
    cross_attn, attn_scores = self.cross_attention(query=seq_emb, value=image, return_attention_scores=True)
    seq_emb = tf.keras.layers.Add()([seq_emb, cross_attn])
    seq_emb = self.cross_attention_norm(seq_emb)

    # Save the last attention scores from the cross attention layer
    self.last_attention_scores = attn_scores

    # Run the updated sequential embedding through the feed forward network and
    # normalize
    seq_emb = seq_emb + self.feed_forward(seq_emb)
    return self.feed_forward_norm(seq_emb)

  def get_config(self):
    config = super().get_config()
    config.update({
      "units": self.units,
      "num_heads": self.num_heads,
      "dropout": self.dropout,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)

@tf.keras.utils.register_keras_serializable()
class VisualAttentionModel(tf.keras.Model):
  def __init__(self, tokenizer, **kwargs):
    super().__init__(**kwargs)
    # Model utilities
    self.tokenizer = tokenizer
  
    # Input processor layers
    self.image_projection = tf.keras.layers.Dense(units=256, activation='relu')
    self.positional_embedding = tf.keras.layers.Embedding(input_dim=80, output_dim=256)
    self.token_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=256, mask_zero=True)
    self.add = tf.keras.layers.Add()

    # Decoder layers
    self.decoder_block = [
        TransformerDecoder(256, num_heads=2, dropout=0.4),
        TransformerDecoder(256, num_heads=2, dropout=0.4),
    ]

    self.output_layer = tf.keras.layers.Dense(units=10000)

  def call(self, inputs):
    image, caption = inputs

    # Flatten the image features and project them
    image_shape = tf.shape(image)
    batch_size, height, width, channels = image_shape[0], image_shape[1], image_shape[2], image_shape[3]
    image = tf.reshape(image, (batch_size, height * width, channels))
    image = self.image_projection(image)

    # Create an embedding for the tokens that captures semantic information
    token_emb = self.token_embedding(caption)

    # Create an embedding for the tokens that captures positinal information
    positions = tf.range(tf.shape(token_emb)[1])
    positions = positions[tf.newaxis, :]
    pos_emb = self.positional_embedding(positions)

    # Combine the token and positional embedding into a single embedding
    seq_emb = self.add([token_emb, pos_emb])

    # Run the images and sequential embeddings through the decoder
    for layer in self.decoder_block:
      seq_emb = layer(inputs=(image, seq_emb))

    # Predict tokens (as logits)
    prediction = self.output_layer(seq_emb)
    return prediction

  def get_config(self):
    config = super().get_config()
    config.update({
      "tokenizer": keras.saving.serialize_keras_object(self.tokenizer),
    })
    return config

  @classmethod
  def from_config(cls, config):
    tokenizer_config = config.pop("tokenizer")
    tokenizer = keras.saving.deserialize_keras_object(tokenizer_config)

    return cls(tokenizer=tokenizer, **config)