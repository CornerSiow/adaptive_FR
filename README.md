# Adaptive FR
It use GNG to store the tiny changes of face features to achieve an adaptive face recognition system

## GNG
User can use the GNG to act as a short-term memory feature to track the feature changes during time-series data.
# Step 1) Create GNG Network
# gng = GNG(512)

# Step 2) Initializing the network using data. (Optional)
# gng.initializing(f1, f2)

# continuous learn the inpu
# while True:
#     ...
#     Step 3) Learn the time-series data / temporal data
#     gng.pushData(feature)

# Step 4) Find max similarity
# similarity = gng.getMaxSimilarity(feature)
# print("Similarity after learning",similarity)

