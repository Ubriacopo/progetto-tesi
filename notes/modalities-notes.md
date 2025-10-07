# Text

- Even if the text for a segment is missing (so not speech recognized) we pass the embeddings. This is important
  because not giving anything (empty tensor or whatever) is different from "" embedding.

# Audio

# Video

- Video embeddings are too big thus I decided to opt for pyramid pooling to reduce the count of tokens from 1336 to 64

# EEG

# ECG