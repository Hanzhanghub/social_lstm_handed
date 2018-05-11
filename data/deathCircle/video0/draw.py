

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = plt.imread('./reference.jpg')
# implot = plt.imshow(im)
#
# cir1 = Circle((0, 1248), 1000, alpha=0.4, color='r')
# cir2 = Circle((1630, 800), 900, alpha=0.4, color='b')
# ax.add_patch(cir1)
# ax.add_patch(cir2)
# # plt.axis('off')
# plt.savefig('./background.png', bbox_inches='tight', pad_inches=0.0, transparent=True)
# plt.show()

fig = plt.figure()
bgimg = plt.imread('./reference.jpg')
implot = plt.imshow(bgimg)
# fig.figimage(bgimg, resize=True)
plt.savefig('./dsf.png')
plt.show()