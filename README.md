# MafQud-Recognition
The main code for the face recognition model used for helping in finding cases on MafQud mobile application.

One of the core parts of MafQud project is the tasks and work that are associated to the data science sub-team. So, their main big tasks are:

- Populating the data to train, validate and test the face recognition model.
- Build the core face recognition model.

And if there is another model (e.g., **GANs**, **Speech-to-Text**, …) as upcoming features.

## Data Scrapping:

We need to find a specific data for our model, our target data should contain at least: photos of the missing people, and their names associated to each photo.

We are able to find our target data and more by scrapping the Atfal Mafkoda | أطفال مفقودة [Website](https://atfalmafkoda.com/) and [Facebook Page](https://www.facebook.com/atfalmafkoda/). With this data we can capture a data of each missing person that contain:

- Name of the missing person.
- Photo(s) of the missing person.
- The government or the location of his residence.
- Date of absence.
- Expected current age.

You can see our scrapping codes for the [Website](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Data-Scraping/Atfal_Website_Scrapping.py) and the [Facebook Page](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Data-Scraping/Atfal_Facebook_Scrapping.py) and see how they save their final data in JSON or CSV file. 

![Scrapped Data CSV](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/csv_scrapped_data.jpg)
![Scrapped Data Json](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/json_scrapped_data_light.jpg)

## MafQud Recognition:
Our face recognition model is the second main part. The approach of recognizing a familiar face (i.e. trained or stored on the system database) has more than one step to be done:

### 1. Face detection model:
We need first to detect the location of the face (or many faces) on a photo (i.e. the coordinates of the face(s) in the photo). That can be done using one of the two algorithms:

- Histogram of Oriented Gradients (HOG) **[\[a method invented in 2005\]](hhttp://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)**: it is the older algorithm, so it is less accurate but yet faster than the other algorithm.

- Convolutional Neural Network (CNN): it is a way more accurate than HOG, but it is slower than it. **[based on ResNet-34 network architecture from “[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)” paper by He et al]**.

We added to them "**the mixed method**" which makes use of the best of the two algorithms.

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/hog.png" alt="HOG"/>
</p>


### 2. Face landmarks estimation: 
We need second to mark the main points in the face which called in our case: landmarks, the more landmarks we target and capture, the more accurate to recognize someone in the photo. So, we capture 68 landmarks in the face (such that: chin, eyebrows, eyes, lips, …) **[\[using face landmark estimation algorithm invented 2014\]](https://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf)**


<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/face_landmarks.png" alt="Ideal Landmarks"/>
</p>




### 3. Face alignment: 
According to the landmarks that are captured from the face, we compare them with an ideal landmark model and their ideal locations, then the photo is aligned to match and cope with the ideal standard landmarks with group of shearing (resizing and rotating).


### 4. Face encoding: 
Last step is to encode the 68 landmarks to be almost unique for the person. Unfortunately, they can’t be interpreted or said that they represent something in the photo as they are the output of a neural network. Now instead of storing the whole photo with its big size of pixels, we just store the 128 encoding numbers ranging from -1 to 1. That is more computational-saving approach and more efficient for space and time.

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/face_encoding.jpg" alt="Face Encoding Example"/>
</p>


### 5. Face comparing model: 
After storing each face as 128 encoding values ranging from -1 to 1, we now are able to compare our stored faces with the new photos that are queried, all what we need is to go through the last 4 steps with the new photo to find the 128-encoding value of the face on it, then we compare them with the stored ones by determining the distance between them and it is more probable that the nearest face of it is the targeted face. That is exactly what our K-Nearest Neighbors (KNN) model does.


## Final output:

Each face enters the system either for storing or for querying about it go through our 5 steps, finally it gives one of the outputs:

### - Matching:
If there is matching between the new photo and one (or some) of the stored photos, then it is the person we are searching for **(with accuracy = %82.0 in our prototype demo).**

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/matching_case1.jpg" alt="Matching Case Example 1"/>
</p>

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/matching_case2.jpg" alt="Matching Case Example 2"/>
</p>


### - Unknown:
If there is no matching happened, then it is a new unknown face to our system model, then we need to store and retrain our model on it in case of future cases.

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/unknown_case3.jpg" alt="Unknown Case Example/>
</p>

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/unknown_message.jpg" alt="Asking for storing"/>
</p>


### - No Face Detected:
There is a possibility that the user by mistake queries for a photo that doesn’t contain a human face at all (photo of dog, photo of nature, …) or even he/she queries for a low-pixeled (i.e., low resolution) photo that is difficult to capture the face on it.

<p align="center">
  <img src="https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/face_not_detcted.jpg" alt="No Face Detcted Case Example"/>
</p>


## MafQud Recognition Overview Chart
### Storing Images (creating encodings)

![Storing Images Flowchart](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/flow_chart1.jpg)


### Query New Images (searching for matching) 

![Query New Images Flowchart](https://github.com/yossef-elmahdy/Data-Science-Demo/blob/main/Screenshots/flow_chart2.jpg)


## References
- [Face recognition library documentation](https://face-recognition.readthedocs.io/en/latest/readme.html). 
- [Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).
