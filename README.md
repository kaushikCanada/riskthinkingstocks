# riskthinkingstocks
End to End ML solution developed for stocks prediction
I have tested the model deployment and it is working fine. I have attached the notebooks below for your reference. For the scheduling as  DAG part, Azure has recently updated their offering which is Data Factory, and it has some access issues I currently am short on time to solve. However, it would be my pleasure to walk you through what I did. 

For data ingestion and feature engineering we use Azure Synapse Analytics and serverless spark pools. The notebook for this is stocks_data_engg.

We have two areas in the storage called 'landing' and 'staging' where we are segregating the data stages.

image.png

Once the features are in staging, we invoke Azure Machine Learning. 
The order of notebooks is first stocks_modelling and then train_deploy.
main.py is the training python file. and sample-request.json file is self explanatory.
Here is the file structure below.


image.png

Use the score.py script ( also as final cell of model_deploy notebook) to get real time predictions. I have kept the endpoint dormant for now to save cost, but can be turned up on demand. THis is exactly liek production where I deploy scale to zero model servers. Here is a screenshot of the experiment.
image.png

 As a bonus, the endpoint also provides a swagger json for the model :-).
