# FF15

By Leo Shi (A17033789) and Eric Gu (A16817621)

## Framing the Problem

As League of Legends continue to gain popularity, more and more Esports fan would like to watch streams on competitive games for the pro-series. We would like to know whether we can fairly accurately predict the length of the game using several features.

### Prediction Question
Given some team statistics of a pro-series League of Legends game, are we able to fairly accurately predict the duration of that game?

### Prediction Type
Regression. We choose regression instead of classification because we would like to use the available pro-series team statistics to predict the `gamelength` for all Legend of League games instead of classifying them into different groups.

### Response Variable
`gamelength`. We choose `gamelength` because players would want to finish a winning game early. With the given team statistics, especially while watching pro-series games, we would want to know when the game can finish while watching it. 

### Metrics
Accuracy. We choose accuracy because we are more interested in whether our result matches the true known value. We consider the false positive and false negative as equally important. Thus, we prefer it over F1-score, recall, and precision.

Additionally, we calculated the RMSE to provide a better idea of how our model is performing, as RMSE enables us to see its performance with respect to the time unit (seconds).

With our dataset, we only choose data that we already know and not include data after the game, such as the result of the game, win team MVP and so on. We predict our model based on the 2023 pro-series League of Legends game. We chose to use data like `dragons`, `teamname`, and `kills` that are either obtainable before, or generated continuously throughout the game.

We perform some data cleaning by dropping the missing values among the dataset because it seems difficult to perform the best approximation for the missing value. 


## Baseline Model

### Features Used
Our baseline model intends to predict the `gamelength` from two features `totalgold` and `damagetochampions`. 
 

`totalgold` (Discrete, Scaling) \
`damagetochampions` (Continuous, Scaling)

Both of the above are quantitative features. We didn’t perform further encoding for both features. We believe that both columns of data are meaningful since the amount of `totalgold` and `damagetochampions` tend to be positively related to the game length. Thus, both features are useful features that help to predict `gamelength`. 

We didn’t choose any categorical columns in the baseline model, which means that there are 0 ordinal features and 0 nominal features. 

### Model Decisions and Results

We chose the DecisionTreeRegressor as our model because it was the simplest to implement as we just used it for our latest lab. Although it may be easy to overfit, searching the best parameter is part of our plan so this would not be a significant problem later.

We perform GridSearchCV with a pipeline including just the DecisionTreeRegressor, the hyperparameters of max_depth between 1 to 20, and cross validation to be 5. When the max_depth is 6, the model has the smallest root mean square deviation(RMSE) around 161. On average each prediction is off by 2 minutes and 41 seconds. The coefficient of determination regression score is around 0.76. As for a baseline model, we think that the current model seems relatively ‘good’ because the accuracy is relatively high, and it intuitively shows that having higher `totalgold` and `damagetochampions` seems to help end the game early, which can become useful for fellow League of Legends players. 


## Final Model

### Features Used

Features added for the final model:
`teamname`, `visionscore`, `kills`, `deaths`, `barons`, `opp_barons`, `dragons`, `opp_dragons`

Used `kills`, `deaths`, `barons`, `opp_barons`, `dragons`, and `opp_dragons` to construct takedowns, total_barons, and total_dragons

`teamname` (Nominal, OneHot): We decided to add teamname as different teams may have different play styles (i.e. While T1 may tend to scale and wait for late game group fights, WBG likes to take fights early to secure an early lead and win faster). We used OneHotEncoder to transform this column and added it to our model as a feature.

`visionscore` (Discrete, Scaling): We added visionscore, which depends on the amount of wards (vision trinket for map visibility) placed and destroyed. Vision is a vital component of pro-plays as it is crucial to predict opponent movement and strategies. It is usually the case that vision score scales linearly as game progresses.

`takedowns` (Discrete, Scaling): We used `kills` and `deaths` to obtain the total takedowns of that specific game as a new column. We are interested in this stat the same reason we included visionscore. It is usually the case that the more takedowns there are, the longer the game would take.

`total_barons` (Discrete, Scaling): We used `barons` and `opp_barons` to obtain the total number of barons slayed during the game. Baron kills are good indicators for game lengths as the first baron always spawns at the 20 minutes mark, and respawns every 6 minutes after that. Baron is an objective and a strategic resource such that once slain, a team would be awarded a huge boost for a short amount of time that significantly increases the ability to take down opponent turrets (the goal of the game). In pro-play, barons are so crucial that games can be overturned by a single play around this objective. Not only is it a good indicator of lengths, it is likely that after the last baron kill, the game would end within 3 minutes.

`total_dragons` (Discrete, Scaling): We used `dragons` and `opp_dragons` to obtain the total number of dragons slayed during the game. Dragon kills may indicate whether a team is having a significant advantage over the other. Once a team has gotten 4 dragon kills, they gain a special boost called “drake soul”, if there were only 4 dragon kills in the game, it would most likely indicate that a team is a lot more dominant and gained significant advantage over the other in the early games, otherwise it would suggest that the game is quite balanced and may go on for a while.

`damagetochampions` (Contininous, Scaling): It is obvious that different teams have different playstyles, and some may prefer to focus on objective, and others to gain advantage by having more takedowns or be more aggressive during midgame (finding group fight chances and poking the enemy team). Additionally, teams may prefer different champions, which would even further vary this feature from team to team. As a result, we standardized this column with respect to the teams.

### Model Decisions and Results

For our algorithm, we continued using DecisionTreeRegressor as our model for prediction, with the same reason as before (explained in baseline). In addition to one-hot-encoding the `teamname`column, we standardized the `damagetochampions` column with respect to the team name. Most other features were quantitative and did not need further transformations.

As the `StdByGroup` transformer was manually implemented, the fitting of our regression model is quite slow. After testing a few parameters like `criterion` and `max_features`, it became apparent that the code would simply take too much time to run. We also limited our max_depth parameter to the range of 1 to 15, as we know that the deeper the max_depth gets, the more likely our model would overfit the training data and perform worse. As a result, we found out (from GridSearchCV) that our best parameter for max_depth was 7, which is what we used for our final model.

The score of the final model is approximately 0.88, which is certainly better than the score of 0.76. The root mean square error (RMSE) was reduced by around 45 seconds, landing at approximately 116.73 seconds for the final model. On average, the prediction is off by 1 minute and 56 seconds. Evidently, the model improved significantly from the baseline.

## Fairness Analysis

### Group Choice
Group X: games with at most 4 dragons
Group Y: games with more than 4 dragons

### Evaluation Metric
We chose 4 dragons because it is possible for a side to get all of the dragons whereas the opponent team doesn’t get any. It will cause a huge lead that will likely create outliers in features like `damagetochampions`, `totalgold`, and `takedowns`, making it more difficult to predict as it would tend to end early but with a higher-than-average stats on multiple columns.

### Null Hypothesis
The model is fair. The prediction for games with at most 4 dragons would be roughly the same as the prediction for games with more than 4 dragons.

### Alternative hypothesis
The model is unfair. The prediction for games with at most 4 dragons would be less accurate than the prediction for games with more than 4 dragons.

### Permutation Test Results

#### Test Statistic
Difference in accuracy measuring by R score (At most 4 dragons minus more than 4 dragons)

#### Significance Level
0.05

#### Distribution Histogram

<iframe src="fig.html" width=800 height=600 frameBorder=0></iframe>

#### P-value
0.0

### Conclusion

Based on the p_value 0.0, it is less than our significance level 0.05. Thus, we reject the null hypothesis. The model might be unfair and the prediction for games with at most 4 dragons would not be roughly the same as the prediction for games with more than 4 dragons. 

