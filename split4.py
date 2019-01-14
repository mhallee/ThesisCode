from PIL import Image
import sys
import re

#adjustable parameters
newHeight = 512
newWidth = 512

# Open picture
im = Image.open(sys.argv[1])

# Extract file name (before .xxx extension)
print(sys.argv[1])
removeExtension = re.compile('[^\.]*')
file = removeExtension.match(sys.argv[1])
prefix = file[0]

#prepare to loop over image
ogWidth, ogHeight = im.size
print("Height: %d"%ogHeight)
print("Width: %d"%ogWidth)
k = 0

#slice image
for i in range(0,ogHeight-newHeight,newHeight):
    for j in range(0,ogWidth-newWidth,newWidth):
        box = (j, i, j+newWidth, i+newHeight)
        a = im.crop(box)  
        a.save("%s - %03d.jpg" % (prefix, k))
        k += 1