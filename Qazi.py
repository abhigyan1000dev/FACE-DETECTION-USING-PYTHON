import cv2
from random import randrange

# LOADING SOME PRE TRAINED DATA ON FRONT FACES THROUGH  THE GITHUB FACE FRONTALS (haar cascade algo)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#STEP -1 
# CHOOSING AN IMAGE

#CHOOSE AN IMAGE TO DETECT FACES
# IT USES IMREAD FUNCTION TO READ IMAGES 
img=cv2.imread('bwood.jpg')

#DISPLAYING AN IMAGE USING IMSHOW
# cv2.imshow('Abhigyan Sharma',img)

# YOU HAVE TO MAKE THE IMAGE WAIT FOR THE KEYSTROKE
# cv2.waitKey()


#STEP -2 
# CHOOSING AN IMAGE
#  CONVERTING THE IMAGE TO GRAYSCALE AS OPEN DETECTS EVETRTHING IN GRAY B/W 

grayscaled_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow('Abhigyan Sharma',grayscaled_image)
# cv2.waitKey()

# DETECT FACE DETECTION 
face_coordinates=trained_face_data.detectMultiScale(grayscaled_image)
 
# print(face_coordinates)

 
 
 # DRAW THE RECTANGLE AROUND THE IMAGE
for (x,y,w,h) in face_coordinates:
 cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# 
cv2.imshow('Abhigyan Sharma',img)
cv2.waitKey()
print(" Code Completed")
