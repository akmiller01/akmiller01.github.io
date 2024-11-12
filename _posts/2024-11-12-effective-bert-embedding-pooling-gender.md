---
layout: post
title:  "What is the most effective method to pool BERT embeddings for text classification? A case study in gender-responsive aid"
author: alex_miller
date:   2024-11-12
categories: blog
---

Text embeddings are typically aggregated before use in classification. Does the method in which the embeddings are pooled affect the ease of classifying the text?

As always, all code used for this blog is available open-source on Github here: [https://github.com/akmiller01/pooling-method-gender-blog](https://github.com/akmiller01/pooling-method-gender-blog){:target="_blank"}

## What is pooling and why is it needed?

First, if you’re unfamiliar with the concept of word embeddings, I recommend reading Ben Schmidt’s excellent blog post on the topic: [Word Embeddings for the Digital Humanities](https://benschmidt.org/posts/2015-10-25-Word-Embeddings/){:target="_blank"}. In a nutshell, word embeddings are long numerical vectors that mathematically represent the contextual meaning of words. An important concept to take away from the blog is that each and every word (or token) within a sentence is given its own numerical vector.

When training a classifier network on word embeddings, we need to define the shape of our inputs and outputs in advance. Outputs are easy (just the number of categories we’re trying to classify), but the input dimensions can vary depending on the length of the text. The model that we will be working with is capable of calculating the embedding vectors of 768 dimensions for up to 512 word sequences. This means that if we use a sentence that is exactly 512 tokens long, we will get 393,216 embedding values, but a 3 word sentence would only yield 2,304 values. Pooling is the method we will use to ensure that the shape of the embeddings are constant regardless of the length of the text. Aside from the main goal of standardizing the dimensions, pooling also needs to minimize the amount of meaningful information destroyed in the dimensional reduction.

## Classifying gender-responsive aid

The data we will use throughout this blog is derived from the International Aid Transparency Initiative (IATI). The IATI data standard contains a “policy-marker” element that allows donors to mark aid as responsive to gender equality at two different levels, significant or principal. Principal means that gender equality is the main objective of the project and that the project would not have been undertaken without the gender objective. Significant means that gender was an important and deliberate objective, but not the main objective. While this marker is also available in the OECD DAC CRS, IATI was chosen for training data as it allows donors to publish a greater quantity of text per activity. The entire, curated dataset is available here: [https://huggingface.co/datasets/alex-miller/curated-iati-gender-equality](https://huggingface.co/datasets/alex-miller/curated-iati-gender-equality){:target="_blank"} This is the same data I have used to train models to screen aid for gender relevance when the donors have not performed that screening themselves.

For all models discussed below, the classifier must identify whether a project has no gender objective, a significant gender objective, or a principal gender objective. Principle is considered as an ordinal improvement over significant, meaning all principal projects are also significant, but not all significant projects are principal. The base model (BERT), learning rates, and batch sizes are held constant between all of the classifiers.

## Unpooled baseline (padding)

The most basic method we can apply to regularize the dimensions of word embeddings isn’t pooling at all. Despite the fact that the length of text can vary, we know that BERT cannot process sequences longer than 512 tokens, so we can fix the dimensional size of the output embeddings by padding all sequences to that maximum. This is accomplished in practice by setting the `padding` parameter on our tokenizer:

{% highlight python %}  
tokenizer(example['text'], truncation=True, padding='max_length')  
{% endhighlight %}

Passing all of the unpooled, padded embedding values to our classifier model also requires us to define a new model class. This new class will be largely based on the [transformer default BertForSequenceClassification](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1623){:target="_blank"} class, but instead of the classifier network input shape being BERT’s default hidden size value of 768, we need to set it to 393,216 (one 768 long embedding vector for each of the 512 tokens/padded spaces). Our model will be defined and initialized like so:

{% highlight python %}  
class BertForSequenceClassificationUnpooled(BertPreTrainedModel):  
    def __init__(self, config):  
        super().__init__(config)  
        self.num_labels = config.num_labels  
        self.config = config

        self.bert = BertModel(config)  
        classifier_dropout = (  
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob  
        )  
        self.dropout = nn.Dropout(classifier_dropout)  
        self.classifier_input_size = config.hidden_size * config.max_position_embeddings
        self.classifier = nn.Linear(self.classifier_input_size, config.num_labels)

        # Initialize weights and apply final processing  
        self.post_init()  
{% endhighlight %}

The forward function also needs a slight modification. The BERT last hidden state matrix is of the shape (batch size, sequence length, hidden size) by default (in our case (24, 512, 768)), and we need to reshape it into a matrix of the shape (batch size, sequence length * hidden size) or (24, 393216). This is accomplished with a little reshaping:

{% highlight python %}  
batch_size = outputs.last_hidden_state.shape[0]  
unpooled_output = outputs.last_hidden_state.view(1, -1).reshape(batch_size, self.classifier_input_size)  
{% endhighlight %}

Over 5 epochs of training (and 1 hour and 13 minutes on a T4 GPU), the unpooled classification model achieves a minimum validation loss of 0.3196 and a corresponding accuracy of 92.69%:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 | Precision | Recall |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1.0 | 1.0053 | 0.5759 | 0.8613 | 0.8326 | 0.7649 | 0.9135 |
| 2.0 | 0.4624 | 0.3887 | 0.9083 | 0.8858 | 0.8365 | 0.9413 |
| 3.0 | 0.3385 | 0.3461 | 0.9274 | 0.9070 | 0.8775 | 0.9387 |
| 4.0 | 0.2909 | 0.3244 | 0.9259 | 0.9057 | 0.8716 | 0.9425 |
| 5.0 | 0.2699 | 0.3196 | 0.9269 | 0.9071 | 0.8715 | 0.9458 |

## Mean pooling

Instead of passing very long, sparse matrices to our classifier network, we can aggregate the embeddings along the hidden state dimension. One simple way to perform this aggregation is by taking the mean. As stated above, the BERT embedding outputs have the shape (batch size, sequence length, hidden size), and because they are a pytorch vector, the mean to reduce them to (batch size, hidden size) can be found like so:

{% highlight python %}  
last_hidden_state = outputs.last_hidden_state  
mean_pooled_output = last_hidden_state.mean(dim=1)  
{% endhighlight %}

Otherwise, training and evaluation proceeds as normal. Over 5 epochs of training (and 55 minutes on a T4 GPU), the mean pooled classification model achieves a minimum validation loss of 0.2068 and a corresponding accuracy of 95.08%:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 | Precision | Recall |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1.0 | 0.4392 | 0.2876 | 0.9078 | 0.8878 | 0.8219 | 0.9651 |
| 2.0 | 0.2471 | 0.2366 | 0.9361 | 0.9198 | 0.8748 | 0.9697 |
| 3.0 | 0.2048 | 0.2143 | 0.9539 | 0.9407 | 0.9156 | 0.9671 |
| 4.0 | 0.1847 | 0.2080 | 0.9498 | 0.9357 | 0.9053 | 0.9684 |
| 5.0 | 0.1706 | 0.2068 | 0.9508 | 0.9370 | 0.9065 | 0.9697 |

## [CLS] pooling (the default)

Ben Schmidt’s embeddings were made with an older model (Word2vec) that is only capable of assigning one fixed vector per unique word in a sentence. This means that the embeddings fail to differentiate cases of polysemy (one word having multiple definitions depending on the context). For example, Word2vec would assign the same embedding vector for the word “bank” in sentences like “We went fishing by the river bank.” and “I went to the bank to deposit a check.” BERT solves this problem by considering the position of each word within the context of the larger sentence. In considering words ahead of the current word, as well as words prior to it, BERT is bidirectional. Because of this bidirectionality, BERT is also capable of pooling the entire contextual meaning of a sentence within a single token, [CLS].

[CLS] is short for “classification” as it was intended to be used for downstream classification tasks. When text is split into tokens before being fed into BERT, the [CLS] token is prepended to the sentence automatically. Although it becomes the very first token in the sentence, BERT essentially reads a sentence front to back, and then back to front, and ends up summarizing the entire sentence in the embedding vector of the [CLS] token.

[CLS] pooling is the method that Huggingface chose to code into their transformer library’s BertForSequenceClassification class. If you take a look at the forward function definition within the class, it simply runs tokenized text through the pre-trained BERT model and then [fetches the embedding vector of the first token](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1680){:target="_blank"} before passing it to the classifier network. Over 5 epochs (and 54 minutes of training on a T4 GPU) our [CLS] pooled model achieves a minimum validation loss of 0.2146 and a corresponding accuracy of 95.22%:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 | Precision | Recall |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1.0 | 0.5237 | 0.3170 | 0.9047 | 0.8842 | 0.8167 | 0.9638 |
| 2.0 | 0.2785 | 0.2489 | 0.9388 | 0.9226 | 0.8831 | 0.9658 |
| 3.0 | 0.2301 | 0.2242 | 0.9537 | 0.9402 | 0.9171 | 0.9645 |
| 4.0 | 0.2071 | 0.2169 | 0.9517 | 0.9381 | 0.9091 | 0.9690 |
| 5.0 | 0.1941 | 0.2146 | 0.9522 | 0.9388 | 0.9093 | 0.9703 |

## Conclusions

After all of that exploration, we come to the conclusion that the default implementation of [CLS] token pooling is not only the fastest method, but also yields the most accurate result. It makes sense that performing back propagation over a neural network that connects 393,216 inputs to 2 outputs takes significantly more training time than one that connects 768 to 2, but why did the classifier with more information fail to produce better results?

At least for the classification of development project text, this tells us that the presence of certain concepts within the text is more important to determining the correct class than the exact position of the concepts within the text. When we reshaped the full, unpooled embedding vector into the matrix used for the classifier network, we preserved all of the ordering information: the values from (1, 1) to (1, 768) correspond to the embeddings for the first token from the first batch, (1, 768) to (1, 1536) correspond to the second token from the first batch, etc. The classifier weights connected to the embedding vector for the first token will only lead to an activation when the concept of gender equality occurs at that exact position, rather than anywhere within the text.

Why does [CLS] pooling slightly out-perform mean pooling? It’s because the [CLS] token itself is the result of the self-attention mechanism of BERT’s transformer. Where some tokens that might yield extreme results in some of the 768 embedding dimensions that could throw off a mean result, BERT’s transformer has been trained to pay attention to which parts of the sentence are the most meaningful for classifying the text.