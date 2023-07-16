# Fake News Detection: Machine Learning Model

## Backround/Motivation
Fake news, spanning all fields and subjects, is a pervasive issue in today's information age. Often associated with politics, fake news isn't limited to this realm alone. It is disseminated by individuals and organizations alike, either as a persuasion tactic or to overshadow undesirable truths. A pertinent example is the varied news circulating around the Covid-19 vaccine development and availability. The inherent danger of fake news lies in its potential to alter people's viewpoints, attitudes, and behaviors, leading to harmful consequences.

Recognizing the importance of discerning trustworthy sources, one might consider manually fact-checking by analyzing the source, scrutinizing the publication date, and researching the author, among other strategies. However, this process can be time-consuming and impractical for every news article. Fortunately, with the rise of big data, pattern detection can be automated to differentiate real news from fake. This was the driving idea behind the Fake News Detection Machine Learning Model project, which uses advanced natural language processing techniques to classify news websites as either real or fake. This innovative application of technology and data analysis provides a feasible solution to the widespread challenge of identifying and combatting fake news.

## Project Description
This machine learning model utilizes binary classification to identify whether a news site is fake or real, in which an output of ‘1’ indicates that the website is most likely fake and ‘0’ indicates that the site is indeed trustworthy. It will take in a list of website URLs and corresponding raw HTML as input data and will train a logistic regression model to output a label of either 0 or 1 depending on
whether the website is real or fake. I chose to use logistic regression as my machine learning algorithm as this efficiently outputs the probability of a news site being fake based on several factors, and if this probability is greater than the threshold value of 0.5, then we can assume that the website is most likely fake.

The core of this model comes in the form of the various natural language processing techniques deployed to transform the input data, previously in the form of words, into numbers that the machine can understand and learn from. I have transformed this data by creating and importing several functions generally referred to as featurizers. As I briefly mentioned before, the purpose of these featurizers is to extract key features of the URL and HTML that may help predict the trustworthiness of the site and transform the data into numerical values to input into the logistic regression model.


## Dataset
To obtain the data necessary for my model, I scraped the web for news websites and compiled a set of *2557 sites, consisting of roughly 50% fake and 50% real. I utilized a python library known as BeautifulSoup to clean up and parse the information I pulled from the web. My dataset was in the form of a list of 2557 elements (AKA websites), each element being a tuple consisting of the URL, HTML, and label (0 or 1). I then split my data into a training set, consisting of 2002 samples; a cross-validation set, consisting of 309 samples; and a test set, with 246 samples. This three-portion split enabled me to train my model using the training set, perform an initial round of testing on my cross-validation set to obtain score reports and tweak my model accordingly, and to finally test my polished model on the unseen test set and obtain a final set of score metrics.

## Initial Exploration
Before creating the featurizers and building my model, I performed some exploratory analysis of my data. I developed some intuitive hypotheses as to what I thought might distinguish a fake website from a real website given the URL and HTML, and quantified my predictions using numerical data. As an example, I hypothesized that websites with .com extensions are likely to be real news sites. To test this out using my dataset, I decided to see what percentage of websites using the .com domain are real and what percent are fake and calculate the ratio of fraction of fake percentage to real percentage. If this ratio is less than 1, then I have reason to believe that websites with a .com domain extension are most likely real. If the ratio is greater than 1, then I have reason to believe that websites with a .com domain extension are fake. If this ratio is 1, or extremely close to 1, then both fake and real news websites can be associated with the .com extension, proving that the .com domain of a website is not the best indicator of the website’s trustworthiness.

After building a function to perform these calculations and obtaining an output of 0.94 for the real fraction, 0.75 for the fake fraction, and 0.80 for the ratio, I observed that the .com domain extension usually corresponded with real news websites, meaning that the
domain of a website might be a reasonable predictor.

## Machine Learning Models and Evaluation Metrics

### Domain Featurizer
I first created a domain featurizer that extracts basic features from the domain name extension of each website. This domain featurizer takes in a URL and an HTML and returns a dictionary mapping the feature descriptions (eg: ‘.com domain’) to numerical features (eg: 0 or 1 depending on whether the URL ends with .com or not). The accuracy of this model was only 55%, which was not surprising as the domain extension, while might provide some clues, cannot be the deterministic predictor of a website’s trustworthiness. I also obtained a confusion matrix as well as a precision, recall, and f-score report on this baseline model to analyze the errors further.

<img width="275" alt="Screen Shot 2023-07-16 at 3 14 52 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/62ccb296-dc07-4414-9aea-3d2096e929c8">

### Keyword Featurizer
The key problem with this model is that there is simply not enough information. To combat this issue, I decided my next step would be to make use of specific (and potentially predictive) keywords of the HTML in addition to the domain extension to feed into the logistic regression model. This new keyword featurizer would get the normalized count of each chosen keyword and add that name-count mapping to the features dictionary. 

<img width="266" alt="Screen Shot 2023-07-16 at 3 15 55 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/b3247319-d230-4a90-bdcb-301c6e0981a0">

### Bag-of-Words NLP Model
Then, as an improvement to my last keyword featurizer, in which I had to manually select various keywords that I thought might be good predictors, I used the Bag-of-Words NLP model to automatically collect the 300 most important keywords in the websites and add their counts for each website description to a feature vector. I used the CountVectorizer class from scikit-learn to find these 300 words and transform the descriptions into a feature vector with the number of rows being the number of descriptions and each of the 300 columns filled with the count of each keyword for each description. This model proved to perform better.

<img width="260" alt="Screen Shot 2023-07-16 at 3 17 11 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/934eded2-674e-4493-a85e-fc08b910e331">

### GloVe Model
Now a shortcoming of the bag-of-words model was that it only looked at the counts of words in the description for each website. I utilized a model called GloVe, which has a plethora of word vectors associated with words, to create a function that returns a word vector of length 300 given a particular word. I observed that words with a higher cosine similarity indicate they are similar and those with a lower cosine similarity score are often dissimilar. 

<img width="258" alt="Screen Shot 2023-07-16 at 3 18 28 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/e0653819-a055-4d3e-b909-f55ddb6febae">

### Combined Approach
Given that I tried out several different featurizers and observed the score reports for each, I was curious to find out if I would obtain improved results when I combine all of the featurization approaches. And it turns out, combining the approaches is quite simple given that each of the feature vectors has the same number of rows. Thus, I concatenated the feature vectors for each website produced using each of the three approaches (domain + keyword featurizer, BOW featurizer, and GloVe featurizer). Since the keyword and domain-based approach had 15 features and the BOW and GloVe had 300 features, the combined approach had a total of 615 features for each website. I then passed this concatenated vector into my logistic regression model and obtained an accuracy of 80%, which was the highest yet.

<img width="265" alt="Screen Shot 2023-07-16 at 3 19 07 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/06eee06d-1353-4093-9e07-aaec620e2d46">

## Conclusion
My fake news detection model helped with predicting whether a website is fake or real with 87% accuracy. 

<img width="268" alt="Screen Shot 2023-07-16 at 3 19 22 AM" src="https://github.com/aashikavishwanath/fake-news-detection/assets/63748375/33f0fcca-7804-4bdf-a410-84c0e9e20888">

As with any machine learning model, there are places for improving the score metrics even further, such as obtaining a larger dataset, developing more featurization approaches, etc. That said, my model seems to be a reasonable predictor for the trustworthiness of a website, and certainly
makes the task of vetting sources much easier and faster than manual labor. 

## Dependencies
- Python 3.9
- Libraries: beautiful soup, pickle, bag-of-words, gloVe

