# Paraphrasepy - Paraphrase any English Sentence
 
 Takes a sentence as input and generates paraphrases for the given sentence. Using Huggingface and PyTorch under the hood. Pretrained model from Huggingface's model hub.
 
 Installation:</br>
 Clone / Download the latest version from the master branch.</br>
CD into the downloaded directory.</br>
Run: pip install . 


Example Usage:
```python
import paraphrasepy
sentence = "What should we eat?"
num_sentences = 5
paraphrasepy.generate(sentence,num_sentences)
```
Returns a dictionary with the input as key and the output as values:
{'What should we eat?': ['What is the best food to eat?','What should I eat daily?','What are the healthy food to eat?','What can we eat?']}


