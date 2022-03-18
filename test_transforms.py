from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去看两个问题

# 2.为什么我们需要Tensor数据类型
'/Users/arthurchen/Documents/code/py_proj/ai_demo/data/train/ants/0013035.jpg'
img_path = 'data/train/ants/0013035.jpg'
img = Image.open(img_path)
print(img)

writer = SummaryWriter('logs')

# 1.transforms该如何使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image('Tensor_img', tensor_img)
writer.close()

