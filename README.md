Command to run : python -m streamlit run app/main.py

For hybrid recommender:
Weight avg choose kiya kyoki
- Content has 170k datapoints (0.6 weighted)
- Collaborative ke bus 10k hai toh usse( 0.4 weight diya hai)

Fir we could've also selected rank fusion, usme dono recommender
ne jo rank diya usko add karke. Jiska least value hoga woh top ayega

Fir tha , Reciprocal Rank Fusion usme

Score=i∑ 1/ (​k+ranki​​), where k =60


Long story short :

A)Weighted Average
-Advantages
1)Very easy to understand.
2)Easy to tune.
3)Uses the actual confidence of each recommender.
-Disadvantages
1)Scores must be on comparable scales.

For example,Suppose:
Content similarity=>0.95

Collaborative similarity=> 0.42
That doesn't necessarily mean collaborative is "worse."
It may simply produce scores in a different range.


B)Rank Fusion
-Advantages
1)Doesn't care about score ranges.
2)If both recommenders think a song deserves a top position, it'll naturally rise to the top.
3)Much more robust.

-Disadvantages
1)Ignores confidence.

For example,Suppose:
These become identical, when 
Content similarity gives
0.99
and
Collaborative similarity gives
0.80

if both are ranked first.



Problem faced:
- 1915-2015 tak hi songs the, new songs spotify ke API se extract karne ka try kiya
- Weightage problem for hybrid recommender ( Hybrid= Content+collaborative)
- schema difference between content and collaborative, so resolve the hybrid ke internally kiya

Content returns
name
artists
year
popularity
Similarity Score

collaborative returns
title
artist_name
year
Similarity