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


1. Large Similarity Matrix Memory Issue

Problem: Initially computed a full cosine similarity matrix for all songs, which resulted in a MemoryError because it required over 200 GB of RAM.

Solution: Refactored the Content-Based Recommender to compute cosine similarity only between the selected song and the dataset on demand, reducing memory usage dramatically.

2. Hybrid Weight Selection

Problem: Determining appropriate weights for combining Content-Based and Collaborative recommendations.

Solution: Chose a weighted average of 0.6 (Content) and 0.4 (Collaborative) because the content dataset contains around 170k songs, whereas the collaborative model is trained on interaction data for approximately 10k songs. Also explored Rank Fusion and Reciprocal Rank Fusion before selecting the weighted approach.

3. Different Recommender Schemas

Problem: Content-Based and Collaborative recommenders returned different column names and metadata formats, making Hybrid recommendation difficult.

Solution: Standardized the output schema across all recommenders and used combine_first() after an outer merge to preserve metadata when songs appeared in only one recommender.

4. Outdated Dataset

Problem: The Spotify dataset contained songs only up to around 2015–2016, limiting recommendations for newer releases.

Solution: Planned Spotify Web API integration as an enrichment layer to fetch metadata and recommendations for recent songs while keeping the existing dataset as the primary knowledge base.

5. Cold Start Problem (This will naturally come later)

Problem: Collaborative Filtering cannot recommend songs that have no user interaction history.

Solution: Addressed this by introducing a Hybrid Recommender, allowing the Content-Based model to recommend songs based on audio features even when interaction data is unavailable.

6.One of the challenges was integrating the Content-Based and Collaborative recommenders into a single Hybrid Recommender. Although both systems recommended songs, they returned metadata using different schemas and sometimes recommended different sets of songs. During the merge operation, songs that were present in only one recommender resulted in missing (NaN) values for columns like name, artists, and year. To solve this, I used Pandas' combine_first() function, which fills missing values from one DataFrame using the corresponding values from another. This ensured that every recommended song retained complete metadata while allowing the Hybrid Recommender to combine recommendations from both models seamlessly.

7."While refactoring, I realized the Spotify dataset and the Million Song Dataset use different identifier systems. Instead of incorrectly treating them as the same key, I standardized the recommender interface and merged hybrid recommendations using normalized song title and artist names. In a production system, I'd replace this with a proper identifier mapping table or ISRC-based matching."