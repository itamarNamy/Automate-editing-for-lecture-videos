import pafy
import youtube_dl
import cv2

url = 'https://www.youtube.com/watch?v=crls61U3-gk'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")

# start the video
cap = cv2.VideoCapture(play.url)
while (True):
    ret,frame = cap.read()
    """
    your code....
    """
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()