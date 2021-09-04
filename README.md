# Amazon-Comprehend --- Natural-Language-Processing-NLP.

### Building Custom Classifier using Amazon Comprehend.
A quick rundown of how to perform NLP tasks with AWS s3, Boto3, and Amazon Comprehend in Python.


![analyze-twitter-comprehend-sagemaker-1](https://user-images.githubusercontent.com/58945964/132104618-a502117c-0f5c-43a0-b9c9-51f8e7151e7b.gif)


### NLP - Natural Language Processing.

Natural Language Processing or NLP is a field of Artificial Intelligence that gives machines the ability to read, understand and derive meaning from human languages.



### Amazon Comprehend.

Amazon Comprehend is a natural-language processing (NLP) service that uses machine learning to uncover information in unstructured data. 
The service can identify critical elements in data, including references to language, people, and places,and the text files can be categorized by relevant topics. 

Comprehend not only locates any content that contains personally identifiable information, it also redacts and masks those content.


### Boto3

Boto3 is AWS SDK for python. Boto3 makes it easy to integrate our Python application, library, or script with AWS services. The SDK provides an object-oriented API as well as low-level access to AWS services.


### Topic modeling
A topic model is a form of statistical model used in machine learning and natural language processing to find abstract "topics" that appear in a collection of documents.
Topic Modeling is an unsupervised learning method for clustering documents and identifying topics based on their contents. It works in the same way as the K-Means algorithm and Expectation-Maximization. We will have to evaluate individual words in each document to uncover topics and assign values to each depending on the distribution of these terms because we are clustering texts

![download](https://user-images.githubusercontent.com/58945964/132104713-661db5cc-1334-4d0a-95e0-535d893e1ba9.png)



### Task: Given the abstract and title for a set of Document, Comprehend has to predict the topics for each Document included in the test set.

   Example:
    Let's say I have 5 documents:
    Document 1: I like mongo and apple.
    Document 2: Crab and fish live in water.
    Document 3: Puppies and kittens are fluffy.
    Document 4: I had spinach and berry smoothie.
    Document 5: My pup loves mango.
    
So if I take this corpus and apply LDA to it. As an example, the model might output something like this:
      Document 1: 100% Topic A.
      Document 2: 100% Topic B.
      Document 3: 100% Topic B.
      Document 4: 100% Topic A.
      Document 5: 60% Topic A, 40% Topic B.
      
Now if a take a look at the topic in detail, we can say that
      Topic A: 40% apple, 20% mango, 10% breakfast....
      Topic B: 60% pup, 40% kitten, 30% dog, 15% cute.....


Now that we know what our topic is about we can know that,
      Document 1 is talking about Food.
      Document 2 is talking about Animals.
      Document 3 is talking about Animals.
      Document 4 is talking about Food.
      Document 5 is talking about Food + Animals.
































