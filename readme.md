## Setup Instructions

1. Create a data directory in the root of the project
2. Download the data from here: https://rutgers.box.com/s/fftnr3cesmrboafaouilo3v4k0api9kz then move all the folders into your data folder. The structure of the folder should look something like this:
```
./ > data > Abraham Lincoln > Address to a Joint Session of Congress on Voting Legislation.txt
./ > data > Agatha Christie > Address to a Joint Session of Congress on Voting Legislation.txt
```
3. Then simply run the `bow_tfidf.ipynb` file in the root directory. This creates a pytorch dataloader which does automatic batching and is extremely useful for any pytorch model you build down the road. The IDF vectors are saved in the class and are 405 dimensions long.

## TODO
- [ ] Use the difference of the tfidf vector and the original text vector to learn the mappings rather than just the generated tfidf vector for the text
- [ ] Use word2vec to generate the vectors (embeddings) and see if that performs better than the tfidf vectors
- [ ] Create a basic Multi Layer Perceptron into which the vectors are inputs and maps to 1 of 100 labels
- [ ] Experiment with LSTMs
- [ ] Generate More Data
- [ ] Fix Confusion Matrix
- [ ] Fix Label Encodings (possibly broken unsure)
- [ ] Improve Model Architecture
- [ ] Find Features (stat. analysis of current output)
- [ ] Slides + Paper
