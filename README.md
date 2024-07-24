# Demo Model for Climate Change Twitter Analysis
- This code takes a Tweet message and Date and predicts its sentiment. It's trained over the [Kaggle Climate Change Twitter Dataset](https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset).
- It generates features based on the tweet, identifies topics with LDA, and then runs a Random Forest to predict Sentiment. The model is trained locally and pickle files are saved and then loaded for inference.
~~- The model is deployed in Heroku. Requests can be made to it on the [API](https://climate-change-model-12a504c55e5e.herokuapp.com/api)~~
~~- It is better for a user to see it using the [Web App](https://climate-change-app-server-6c4cbb1cf04f.herokuapp.com/) also available on Heroku.~~

- You can find the source code on GitHub:
    - [Model](https://github.com/pablo-marin-vicuna/tweet_model)
    - [Web App](https://github.com/pablo-marin-vicuna/tweet_app)
- Authors: 
    - [Pablo Marin Vicuna](pmarin@andrew.cmu.edu)
    - [Marvin Espinoza-Leiva](mespinoz@andrew.cmu.edu)
