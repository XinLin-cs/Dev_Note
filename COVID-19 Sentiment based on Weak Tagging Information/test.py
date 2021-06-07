from keras.datasets import imdb
k=imdb.load_data(num_words=10000)
print(imdb.get_word_index())