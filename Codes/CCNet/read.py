——import os
import numpy as np
import skimage.io as io
from skimage.transform import resize
import torch.utils.data.dataset as Dataset

def normor(image): #[n,d,w,h]
    image -=image.mean()
    image /=image.std()
    return image

def convert_to_one_hot(seg):
    shape = seg.shape
    outs = np.zeros((35,shape[0],shape[1]), seg.dtype)
    for i in range(34):
        outs[i][seg == i] = 1
        
    outs[34][seg==-1] = 1

    out = np.zeros((4,shape[0],shape[1]), seg.dtype)

    out[3] = outs[23]
    out[2] = outs[22] + outs[21]
    out[1] = outs[16] + outs[15] + outs[14] + outs[13] + outs[12] + outs[11]

    for i in range(11):
        out[0] = out[0] + outs[i]
    for i in range(17,21):
        out[0] = out[0] + outs[i]
    for i in range(24,35):
        out[0] = out[0] + outs[i]
    return out

def convert_to_one_hot1(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

def resize_label(image):
    edsize = image.shape
    result = np.zeros((edsize[0],edsize[1]//2,edsize[2]//2), image.dtype)
    for i in range(len(image)):
        result[i,:,:] =resize(image[i].astype(float), (edsize[1] // 2, edsize[2] // 2), order=1, mode='edge')[None]
    # image = vals[np.vstack(result).argmax(0)]
    return result


# 原来这里data文件夹下面的image是用于train的
Path_img_train = './data/train/'

# +++++++++++++++++++++++++++++++++++
Path_img_test = './data/test/'

# 原来这的lable是训练集的lable
Path_lab = './data/label/'

img_train = []
img_test = []

for root,dirs,files in os.walk(Path_img_train):
    for file in files:
        found_path = os.path.join(root, file)
        img_train.append(found_path)
        
for root,dirs,files in os.walk(Path_img_test):
    for file in files:
        found_path = os.path.join(root, file)
        img_test.append(found_path)

# image = io.imread(img[0]).astype(float)
# image = image.transpose(2,0,1)
# imagenorm = normor(image)
# labelname = Path_lab + img[0][-36:-15] + 'gtCoarse_labelIds.png'
# label = io.imread(labelname)

# label_one = convert_to_one_hot(label)

class Data(Dataset.Dataset):
    def __init__(self,img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image = io.imread(self.img[index]).astype(float)
        image = image.transpose(2, 0, 1)
        ins = img[index].rfind('/')
        # print(img[index])
        # print(img[index][ins:-15] )
        labelname = Path_lab + img[index][ins:-15] + 'gtFine_labelIds.png'
        label = io.imread(labelname)
        label_one = convert_to_one_hot(label)

        edsize = image.shape

        image = resize(image, (edsize[0], edsize[1]//2, edsize[2] // 2), order=3,
                       mode='edge').astype(np.float32)

        imagenorm = normor(image)
        label_one = resize_label(label_one)

        return imagenorm,label_one

train_data = Data(img_train)
test_data = Data(img_test)
