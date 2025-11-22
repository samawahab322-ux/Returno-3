####################imports####################
# Do not change

import cv2
import numpy as np
import tensorflow as tf

# Do not change
####################imports####################

#Following are the model and video capture configurations
# Do not change

model=tf.keras.models.load_model(
  'saved_model.h5', 
  custom_objects=None,
  compile=True, 
  options=None)

cap = cv2.VideoCapture(0)                                      # Using device's camera to capture video
if (cap.isOpened()==False):
  print("Please change defalut value of VideoCapture(k)(k = 0, 1, 2, 3, etc). Or no webcam device found")

text_color=(206,235,135)
org=(50,50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
thickness=3

class_list=['Alfred Enoch','Harry Potter','Hermione','Menna','Ron Weasley','Sama']               # List of all the classes 

# Do not change
###############################################

#This is the while loop block, computations happen here

while cap.isOpened():
  ret, image_np = cap.read()                                 # Reading the captured images
  if ret==False:
    print("Your camera might be open in some other application.")
    break
  image_np = cv2.flip(image_np, 1)               
  image_resized=cv2.resize(image_np,(224,224))   
  img_array = tf.expand_dims(image_resized, 0)               # Expanding the image array dimensions
  predict=model.predict(img_array)                           # Making an initial prediction using the model
  predict_index=np.argmax(predict[0], axis=0)                # Generating index out of the prediction
  predicted_class=class_list[predict_index]                  # Tallying the index with class list
    
  image_np = cv2.putText(
      image_np,
      "Image Classification Output: "+str(predicted_class),
      org,
      font,
			fontScale,
      text_color,
      thickness,
      cv2.LINE_AA)
    
  cv2.imshow("Image Classification Window",image_np)         # Displaying the classification window

  if cv2.waitKey(25) & 0xFF == ord('q'):                    # Press 'q' to close the classification window
    break
    
cap.release()                                                 # Stops taking video input 
cv2.destroyAllWindows()                                       # Closes input window
