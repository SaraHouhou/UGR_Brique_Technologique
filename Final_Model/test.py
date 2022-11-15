import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/shouhou/Downloads/img.jpg')
img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img)
print(img.shape)
#plt.plot(img)
#cv2.save.savefig(img)
#plt.imshow(img)
cv2.imshow(' image and resized', img)
if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()