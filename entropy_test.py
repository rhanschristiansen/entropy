import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def entropy(hist):
    entropy = 0
    for i in range(len(hist)):
        if hist[i] > 0:
            entropy -= np.log2(hist[i])*hist[i]
    return entropy


im = Image.open('lenagrey.gif')
im.save('lenagrey.png')
im = Image.open('lenacanny.gif')
im.save('lenacanny.png')
im = Image.open('lenacannylowradius.gif')
im.save('lenacannylowradius.png')

img_names = ['lena grey scale', 'lena canny', 'lena canny low radius']
lenagrey = cv2.imread('lenagrey.png')
lenacanny = cv2.imread('lenacanny.png')
lenacannylowradius = cv2.imread('lenacannylowradius.png')
imgs = [lenagrey, lenacanny, lenacannylowradius]
hists = []
fig, axes = plt.subplots(2, 3, figsize=(10,7))
for i, (img, name) in enumerate(zip(imgs, img_names)):
    hist, _, _ = axes[1,i].hist(img.ravel(), bins=256, density=True)
    ax2 = axes[0,i]
    ax2.imshow(img)
    e = entropy(hist)
    axes[1,i].set_title(name + ' Entropy: ' + str(e))


plt.show()
a = 1