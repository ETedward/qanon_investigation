from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

tf.disable_eager_execution()

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
embeddings = embed([
    "Why are children immune to COVID-19.",
    "Why did Mueller meet POTUS 1-day prior to FBI",
    "Repub distortion of facts to remove Mueller"
    ])

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(session.run(embeddings))

#message_embeddings = session.run(embed(messages))
  #
  # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  #   print("Message: {}".format(messages[i]))
  #   print("Embedding size: {}".format(len(message_embedding)))
  #   message_embedding_snippet = ", ".join(
  #       (str(x) for x in message_embedding[:3]))
  #   print("Embedding: [{}, ...]\n".format(message_embedding_snippet))