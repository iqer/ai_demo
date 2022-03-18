from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer = SummaryWriter('logs')
image_path = 'data/train/ants/20935278_9190345f6b.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image(tag='train', img_tensor=img_array, global_step=1,  dataformats='HWC')
# y = 2x
for i in range(100):

    writer.add_scalar('y=2x', 2*i, i)


writer.close()
