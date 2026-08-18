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

7."While refactoring, I realized the Spotify dataset and the Million Song Dataset use different identifier systems. Instead of incorrectly treating them as the same key, I standardized the recommender interface and merged hybrid recommendations using normalized song title and artist names. In a production system, I'd replace this with a proper identifier mapping table or ISRC-based matching."'

8.Challenge: The collaborative model only worked for songs present in the interaction dataset. Many Spotify tracks had no interaction history, causing the hybrid recommender to receive no collaborative candidates.

Solution: Standardized the recommender interface to always return a DataFrame (even when empty) and implemented graceful fallback logic in the hybrid recommender. This made the system more robust and prepared it for future cold-start handling strategies.

"While implementing the hybrid recommender, I found that the content-based and collaborative models were built on different datasets with inconsistent song titles and identifiers. Rather than forcing unreliable matches, I designed the system to gracefully handle missing collaborative candidates and planned a metadata enrichment layer using the Spotify API to create a common catalog."

About Recommendation Analytics
| Chart                   | Question Answered                        |
| ----------------------- | ---------------------------------------- |
| Top Artists             | Who is being recommended most?           |
| Year Distribution       | Which era do recommendations belong to?  |
| Popularity Distribution | Are recommendations mainstream or niche? |
| Recommendation Scores   | How confident is the recommender?        |


About Recommendation Diversity

| Diversity Metric         | What does it measure?                                                          |
| ------------------------ | ------------------------------------------------------------------------------ |
| Overall Diversity        | How diverse and balanced are the recommendations overall?                      |
| Artist Diversity         | Are recommendations coming from different artists or the same few artists?     |
| Year Diversity           | Do recommendations span different release years or focus on a specific era?    |
| Popularity Diversity     | Does the recommender balance mainstream hits with niche or less popular songs? |



                  MASTER MUSIC CATALOG
                 merged_dataset.csv
                     (312K songs)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Search UI         Content Engine      Metadata/XAI
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    Hybrid Recommender
                           │
              ┌────────────┴────────────┐
              │                         │
      Collaborative Dataset      No Interaction Data
 (triplets + song_data)           (new songs)
              │                         │
              ▼                         ▼
      Hybrid Recommendation     Content Recommendation



the final architecture becomes
                merged_dataset.csv
                       │
      ┌────────────────┴─────────────────┐
      │                                  │
 Search UI                     Content Recommender
      │                                  │
      └──────────────┐                   │
                     ▼                   │
             Hybrid Recommender ◄────────┘
                     ▲
                     │
      triplets_file.csv + song_data.csv


we created CatalogService?
Ans:-Because its main purpose is not to change the recommender.

It solves:

-duplicate titles in the UI
-formatted dropdowns
-metadata retrieval
-future search/autocomplete
-richer recommendation cards

The recommenders don't have to know about any of that.

Phase 1: Migration of data and stabilization . 
          After Realizing that our current 130k songs dataset contains mostly 1916 to 2016 songs and thus
          reduced the exprience of the user so to improve it we added 2 dataset from kaggle 
          900k (https://www.kaggle.com/datasets/olegfostenko/almost-a-million-spotify-tracks)
          114k (https://www.kaggle.com/datasets/saichaitanyareddyai/spotify-tracks-dataset-audio-features)

          and we preprocessed them by removing unneccesary columns, missing values. 114k dataset had significantly large value of missing dataset. To get a detailed report of every ETL of both dataset go throuh reports folder.

          Then, we then made some architectural changes in the system to ensure that recommenders get the same list of features
          from the new dataset created (merged_dataset.csv) and thenyaa

Phase 2: Added CatalogService for better, metadata enrichment to be displayed in recommendation cards
          Advantage of creating new dataset was that, we now have more features for every songs and we can add them to the recommendation cards to improve UX. Here, CatalogService acts as a middle man to smoothen the extraction of data from merged_data.csv and provide it to the recommender in much better way and yaa bus etna hi tha

Phase 3:
          In this phase we try to tackle the original problem about"What if a user searches a song which is not present in merged_dataset?"
                                             Phase 3.1(Live Spotify Search Fallback)
                                             If a song isn't found locally, query Spotify and display its metadata.
User types a song                                                ↓
↓                                            Phase 3.2(Album Artwork)
↓                                            Use the album_image URL your SpotifyClient already returns to show album covers in the UI.
Search local merged_dataset                                      ↓
↓                                            Phase 3.3(30-second Preview)
Found?                                       If preview_url is available, embed it with st.audio() so users can listen immediately
├── Yes                                                          ↓
│      │                                     Phase 3.4(Smarter Search)
│      ▼                                     Replace the giant dropdown with a searchable interface and optional artist filtering.
│  Recommendations                                               ↓
└── No                                     Phase 3.5(Metadata Enrichment)
       │                                 When Spotify provides richer information than our dataset (album art, release details)       ▼                                     display it alongside recommendations without changing the recommendation engine. 
Spotify Search
       │
       ▼
Display Spotify Metadata
       │
       ▼
Search again locally using title + artist
       │
       ▼
Recommendations

Phase 3 - Intelligent Spotify Fallback (2-Stage Matching)
Problem Statement:
Originally, the recommendation engine could only recommend songs that already existed in the local merged_dataset.csv (312K songs).
If a user searched for a newly released song that wasn't present in the dataset, the application simply returned:
"No songs found."

To solve this, we introduced a 2-stage matching approach that combines Spotify Search with our local recommendation engine.

Design Principle:
The Spotify API is not used to generate recommendations. It is used only as a live song discovery layer. Once a song is identified, the application maps it to the closest song in the local catalog and leverages the existing explainable recommendation engine to produce recommendations. This keeps the recommendation process fast, explainable, and independent of Spotify while allowing users to search for songs that may not yet exist in the local dataset.


User searches:
       APT
        │
        ▼
Search Local Dataset
        │
        ▼
     No Match
        │
        ▼
     Spotify Search
        │
        ▼
     Returns:
     APT.
     ROSÉ
     2024
        │
        ▼
SearchService.get_candidates()
        │
        ▼
Candidate Songs (~200)
        │
        ▼
MatchingService.find_best_match()
        │
        ▼
Best Local Match:APT (98.7%)
        │
        ▼
Content/Hybrid Recommender
        │
        ▼
Final Song Recommendations


Overall Architecture:
                User Search
                     │
                     ▼
            Search Local Dataset
                     │
          ┌──────────┴──────────┐
          │                     │
      Song Found           Song Not Found
          │                     │
          ▼                     ▼
 Recommendations         Spotify Search
                                │
                                ▼
                     Get Spotify Metadata
                                │
                                ▼
                     Candidate Generation
                                │
                                ▼
                      Fuzzy Matching
                                │
                                ▼
                  Closest Local Song Found
                                │
                                ▼
                 Existing Recommendation Engine
You may askWhy Two Stages?
Ans:Our local catalog contains approximately 312,000 songs.
Comparing every Spotify search result against all 312K songs would require:
1 Spotify Search
        ↓
312,000 similarity comparisons
This is inefficient.

Instead, we divide the problem into two stages:
Stage 1 → Filter                             Stage 2 → Rank
312,000 songs                                200 candidates
        ↓                        →                   ↓
200 likely candidates                        Top 5 best matches

This reduces the computation dramatically while maintaining high accuracy.

Stage 1 - Candidate Generation (SearchService.get_candidates())
Purpose:Reduce the search space from 312K songs to a few hundred likely matches.

Inputs
Spotify Song Title
Spotify Artist
Process
Convert title and artist to lowercase.
Split them into individual words (tokens).
Search the local dataset using these tokens.
Match against:
Song title
Artist name

Sort candidates by:
Popularity
Release Year
Return the top 200 candidates.

Example
Spotify returns:
Song   : Believer
Artist : Imagine Dragons

Candidate Generation returns:
Believer
Believer (Live)
Believer Remix
Thunder
Enemy
Natural

Instead of comparing against 312,000 songs, we now compare only these candidates.

Stage 2 - Fuzzy Matching (MatchingService.find_best_match())
Purpose:Find the closest local song from the candidate pool.

Inputs
Spotify Track Metadata
Candidate DataFrame (≈200 songs)
Similarity Calculation

Each candidate receives three scores:

1. Title Similarity (60%)
Calculated using Python's SequenceMatcher.

Example:
Believer
Believer (Live)

Similarity = 0.92
2. Artist Similarity (30%)

Example:
Imagine Dragons
Imagine Dragons

Similarity = 1.00
3. Release Year Similarity (10%)

Example:

Spotify:2017
Local:2018
Difference = 1 year
Year Score = 0.8

Final Score/ Matching Score =
0.60 × Title Similarity
+
0.30 × Artist Similarity
+
0.10 × Year Similarity

Example)
Spotify Song

Believer
Imagine Dragons
2017

Candidate 1

Believer
Imagine Dragons
2017

Scores
Title   = 1.00
Artist  = 1.00
Year    = 1.00

Final
100%

Candidate 2
Believer (Live)
Imagine Dragons
2018

Scores
Title   = 0.92
Artist  = 1.00
Year    = 0.80

Final
≈93%

Candidate 3
Believe
Cher
1998

Scores
Title   = 0.65
Artist  = 0.10
Year    = 0.00

Final
≈42%

The candidates are then sorted by Matching Score, and the highest-scoring song is selected as the closest local match.

Why Use SequenceMatcher?
Ans: Instead of exact string matching, SequenceMatcher computes a similarity ratio between two strings.

Examples:
Spotify Song	    Local Song	          Similarity
APT.                  APT	          ~0.95
Believer	          Believer (Live)	~0.92
Shape of You       	Shape Of You	     ~1.00

This makes the matching process robust against punctuation, case differences, and small variations in titles.



Advantages of the 2-Stage Approach
-Efficient: reduces comparisons from ~312K to ~200.
-Scalable: performance remains fast even as the catalog grows.
-Explainable: every recommendation is based on transparent similarity metrics.
-Independent of Spotify: recommendations are generated from the local catalog, while Spotify is used only to identify songs missing from the dataset.


spotify_fallback() Helper function: 
Workflow: 
                        main.py
                        ↓
                        spotify_fallback()
                        ↓
                        Spotify Search
                        ↓
                        Candidate Generation
                        ↓
                        Matching
                        ↓
                        Return local song
                        ↓
                        main.py continues normally



Q)If the interviewer asks me about the working of collaborative...
"How does your collaborative recommender work?"

Don't give the overly simplified:
A listens to X and Y, B listens to X, so recommend Y.

Instead say:
"I use implicit user-item interaction data, where each interaction contains a user, song, and listen count. I construct a song-user interaction matrix and represent each song by its listening pattern across users. I then use cosine similarity between these vectors to identify songs with similar user-consumption patterns. For example, if two songs are frequently listened to by similar users, their vectors will be similar and one can be recommended when the other is selected."

Q)"Why did you use multiple recommendation techniques?"

"I used three complementary approaches. Content-based filtering works on the audio characteristics of a song and therefore provides broad coverage. Collaborative filtering uses historical user-listening behavior and can capture patterns that audio similarity cannot. Hybrid combines both signals when both are available. 

However , because the collaborative interaction dataset has much lower coverage than the content catalog, I designed the system so that it falls back to content-based recommendations rather than forcing a collaborative or hybrid recommendation when sufficient interaction data isn't available."


"I optimized the hybrid pipeline by checking collaborative availability before invoking the collaborative recommender. Since the collaborative dataset has limited active-song coverage, this avoids unnecessary computation for songs that cannot participate in collaborative recommendation. For songs with interactions, both recommenders contribute to the weighted hybrid score."


In recommend()
"Because my collaborative dataset only covers a small subset of the content catalog, I added an availability check before invoking collaborative recommendation. This prevents unnecessary collaborative computation for songs that cannot produce collaborative recommendations, while retaining the full hybrid pipeline for songs with interaction data."