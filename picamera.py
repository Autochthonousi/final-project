#load the camera and color for the next step

from picamera import PiCamera, Color
from time import sleep

demoCamera = PiCamera()

demoCamera.start_preview()   
#set the  color for the  ground

demoCamera.annotate_background = Color('white')
demoCamera.annotate_foreground = Color('red') 
demoCamera.resolution = (480, 320)     
demoCamera.framate = 60                 

#get the name for the  camera and show on the screen
demoCamera.annotate_text = " Picamera"   
sleep(5)    
#save the  picture on the raspberry pi, and set the location of the picture
demoCamera.capture('/home/pi/Desktop/photo.jpg')  
demoCamera.stop_preview()      
