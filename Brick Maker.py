
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# ## Important Parameters

img_height = 60
img_width = 60

brick_height = 8
brick_width = 16

mortar_width = 2

brick_rows = img_height / (brick_height + mortar_width)
brick_cols = img_width / (brick_width + mortar_width)

print(brick_cols)
print(brick_rows)

for counter in range(0,10):
    # ## Start Image


    sample_image = np.zeros([img_height,img_width])


    # ## Compute Brick Corner Points

    tops = np.linspace(0,img_height,brick_rows+1).astype(int)
    lefts = np.linspace(0,img_width,brick_cols+1).astype(int)

    #add noise

    row_noise_scale = mortar_width -1
    col_noise_scale = mortar_width -1

    # ## Draw Bricks

    offset = brick_width / 2
    offset_row = False

    for t in tops:
        for l in lefts:
            
            top = t + int(np.abs(np.random.normal(loc=0.0, scale=row_noise_scale)))
            bottom = top + brick_height
            
            left = l + int(np.abs(np.random.normal(loc=0.0, scale=col_noise_scale)))
            if offset_row:
                left += offset  
            right = left + brick_width
                              
            sample_image[top:bottom,left:right] = 1
            
        offset_row = not offset_row




    plt.figure()

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_image, cmap=plt.cm.binary)
    plt.savefig("bricks{}.png".format(counter))




