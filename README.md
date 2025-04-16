# Adaptive FR
It use GNG to store the tiny changes of face features to achieve an adaptive face recognition system

## GNG
User can use the GNG to act as a short-term memory feature to track the feature changes during time-series data.
```
# Step 1) Create GNG Network
gng = GNG(512)

# Step 2) Initializing the network using data. (Optional)
gng.initializing(f1, f2)

# continuous learn the input data
while True:
     # Read the time series data
     ...
     #Step 3) Learn the time-series data / temporal data
     gng.pushData(feature)

# Step 4) Find max similarity
similarity = gng.getMaxSimilarity(feature)
print("Similarity after learning",similarity)
```
## Face Descriptor
User can use face descriptor to extract the face from the image, perform alignment using proposed method.
```
# Step 1) Create the descriptor
descriptor = FaceDescriptor( withAlignment=True, alignmentType='proposed')

# Step 2) Read the image
img = cv2.imread('material/test_face.jpg')
# must convert to RGB arrangement
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3) Extract the facial image from the image
face = descriptor.extractFacial(img)
# Display it, need to convert to cv2 color
face_img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR) 
cv2.imshow('img', face_img) 
cv2.waitKey(0)   
cv2.destroyAllWindows()

# Step 4) Extract the facial feature
feature = descriptor.extractFacialFeatures(face)
print(feature)
```
