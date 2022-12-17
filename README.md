# **Direct-Marketing-Optimization**
## _Propensity Modeling_

Direct Marketing Optimization of campaigns on customers based on propensity modelling.

- [Executive Summary](https://github.com/Bhanu-py/Direct-Marketing-Optimization/blob/master/executive-summary.pdf)
- [Technical Report](https://github.com/Bhanu-py/Direct-Marketing-Optimization/blob/master/technical_report.pdf)

Data: 
For the analysis, several tables are available:
- Social-demographical data (age, gender, tenure in a bank) 
- Products owned + actual volumes (current account, saving account, mutual funds, overdraft, 
credit card, consumer loan) 
- Inflow/outflow on C/A, aggregated card turnover (monthly average over past 3 months) 
- For 60 % of clients actual sales + revenues from these are available (training set) 
Conditions: 
- The bank has capacity to contact only 15 pct. of the clients (cca 100 people) with a marketing 
offer and each client can be targeted only once. 
Proposed steps: 
1. Create an analytical dataset (both training and targeting sets) 
2. Develop 3 propensity models (consumer loan, credit card, mutual fund) using training data set 
3. Optimize targeting clients with the direct marketing offer to maximize the revenue 
Expected result: 
- Which clients have higher propensity to buy consumer loan? 
- Which clients have higher propensity to buy credit card? 
- Which clients have higher propensity to buy mutual fund? 
- Which clients are to be targeted with which offer? General description. 
- What would be the expected revenue based on your strategy? 
The executive summary of the analysis should not be larger than two pages. Attach the technical 
report, list of clients to be contacted with which offer, data, algorithms and codes used.

## _Project Work-FLow_
![image](https://user-images.githubusercontent.com/57532016/208249823-558861a4-5f81-435c-99e1-75806f8df0b3.png)

## _Revenue Optimization_ 
Heuristic 1: average costs and revenues
![image](https://user-images.githubusercontent.com/57532016/208250042-8bcc71d9-8a16-4d38-8a2d-3ed72eb9a22f.png)

_**Reference**_:  
Optimization models for targeted offers in direct marketing: exact and heuristic algorithms: 
[Paper](https://doi.org/10.1016/j.ejor.2010.10.019)
