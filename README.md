# steganography-with-masked-lm

This repository contains implementations of the steganography algorithms with masked language models.
Please refer to our paper [Frustratingly Easy Edit-based Linguistic Steganography with a Masked Language Model](https://arxiv.org/abs/2104.09833)
for more details.

## Requirements

- Python >= 3.6.5

## Installation

```
$ git clone https://github.com/ku-nlp/steganography-with-masked-lm.git
$ cd steganography-with-masked-lm
$ pip install -r requirements.txt
```

Then, download the NLTK stopwords corpus data (see also https://www.nltk.org/data.html#interactive-installer ).

```
$ python
>> import nltk
>> nltk.download()
Downloader> d
  Identifier> stopwords
```


## Example usage

```
$ python main.py "The quick brown fox jumps over the lazy dog." -m 010101
{'stego_text': 'The quick red fox jumps over the poor dog.', 'encoded_message': '010101'}

$ python main.py "The quick red fox jumps over the poor dog." --decode
{'decoded_message': '010101'}
```

Run `python main.py -h` for more help.
