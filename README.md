# Comparison of word vector models for german language

This project makes a comparison of different word vector models for german
language. The intention for is that many word vector models are only
trained for the english language, for the german language there are only
a few pre-trained models.

The following models are compared in this project:

* Context tensors from [spaCy](https://github.com/explosion/spaCy)
* [fastText](https://github.com/facebookresearch/fastText) vectors
    * normal model
    * quantized model
* ELMo vectors from [bilm](https://github.com/allenai/bilm-tf)
* [Word2Bits](https://github.com/agnusmaximus/Word2Bits) vectors

The context tensors from spaCy are described as vectors for words which
are not as good as other word vector models. Well trained models are
[announced for version 2.1.0](https://github.com/explosion/spaCy/issues/2523).
The fastText vectors are available for many different languages including
german. For the comparison with other vectors, the pre-trained vectors are
used. For fastText, there is also a method to quantize pre-trained models.
Since there are no pre-trained quantized fastText vectors for german
language are provided, a quantized model was trained for this project. For
the quantization of the fastText model, it has to be retrained on the texts
which it was trained. The fastText models were trained on german wikipedia
articles, but the train data which was used was not published.

So, for this project, it was generated new train data from german wikipedia
articles. The difficulty of that is to clean the texts in the right way,
because they are only available as xml. One has to clean out the xml tags
and other wikipedia specific formation tags.

When this is done, the fastText vectors can be quantized. Also the two
other models, ELMo and Word2Bits, are trained on these data because there
are also no vectors provided for the german language.

## Dataset

For evaluating the models, the German Sentiment Corpus [SB-10k](https://www.spinningbytes.com/resources/germansentiment/)
was used. It contains 9738 German tweets from Twitter which are annotated with
“positive”, “negative”, “neutral”. Since Twitter doesn't allow to publish
tweets in data sets, the corpus only contains sentiment information and
the ids of the tweets. To get the texts, the data must be downloaded with
the twitter api. To do that, the tool [twitter_download](https://github.com/aritter/twitter_download).
If you want to reproduce the results, please download this data and mount it
into the docker image (See section Usage).

## Results

### Sentiment Analysis

In the following table, the accuracy values for the different sentiment
classes are shown.

| Vector-Type | Negative | Neutral | Positive |
| ------------------ | ------------------ |------------------ | ------------------ |
| spacy-context | 0.818 | 0.734 | 0.760 |
| fasttext | 0.856 | 0.766 | 0.766 |
| fasttext quantized | 0.852 | 0.768 | 0.756 |
| ELMo | 0.852 | 0.766 | 0.818 |
| Word2Bits | upcoming ...

## Usage
To run the evaluation on your own, there is the opportunity to build a
docker image. To build the image, a dockerfile is provided. The file
load all dependencies except of the twitter dataset, which has to be mounted.
Since a gpu was used for the evaluation, nvidia-docker has to be used
to run the docker image. A tutorial how to set up nvidia-docker can be found
[here](https://github.com/NVIDIA/nvidia-docker). When the image was built,
it can be run with
```
docker run --runtime=nvidia --mount type=bind,source=path/to/your/datafile,target=/data/ -it image-name
```
where path/to/your/datafile has to be replaced with the location where your
dataset file has been saved and image-name with the name of the image or the
container id.

After running the image, the following commands are possible:
* python evaluation.py --model spacy_context
* python evaluation.py --model fasttext
* python evaluation.py --model fasttext_quantized
* python evaluation.py --model elmo

## Downloads

Some word vector models for the german language, like fasttext and spacy,
are available online. Additional to these, the following word vector
model can be downloaded here:
* [Fasttext with quantized weights (german)](https://s3.eu-central-1.amazonaws.com/nlp-machine-learning-data/cc.de.300_quantized.ftz.gz)
* [ELMo model (german)](https://s3.eu-central-1.amazonaws.com/nlp-machine-learning-data/elmo_weights.zip)
