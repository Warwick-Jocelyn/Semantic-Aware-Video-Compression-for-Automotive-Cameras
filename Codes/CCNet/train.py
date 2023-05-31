import torch
import read as data
import torch.utils.data as Datas
import CCNet as Network
import metrics as criterion

from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
import skimage.io as io

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_ids = [0]

# model = Module()

device = torch.device("cuda:0")
data = data.train_data

dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)

fusenet = Network.SegNetwork().to(device)
opt = torch.optim.Adam(fusenet.parameters(),lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)

#######
pretrained_dict = torch.load('./pkl/net_epoch_27-fuseNetwork.pkl')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pretrained_num = 27
model_dict = fusenet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
fusenet.load_state_dict(model_dict)

#Some criterion
criterion_CE = criterion.crossentry()
criterion_dice = criterion.DiceMeanLoss()
criterion_dice1 = criterion.DiceMeanLoss1()
criterion_iou = criterion.IOU()

for epoch in range(200):
    meansegdice = 0
    for step, (img, label) in enumerate(dataloder):
        # print(img.shape)
        # print(label.shape)

        img = img.to(device).float()
        label = label.to(device).float()
        b, c, w, h = img.shape

        segresult = fusenet(img)

        # print(segresult.shape)
        # print(label.shape)
        lossseg_ed_es = criterion_dice1(segresult, label)
        lossseg_ce = criterion_CE(segresult, label)

        if step% 100==0:
            segresulti = segresult[0,0,:,:,]*0+segresult[0,1,:,:,]*1+segresult[0,2,:,:,]*2+segresult[0,3,:,:,]*3
            io.imsave('./seg'+str(step)+'.png', segresulti.data.cpu().numpy())

            segresulti = label[0, 0, :, :, ] * 0 + label[0, 1, :, :, ] * 1 + label[0, 2, :,
                                                                                     :, ] * 2 + label[0, 3, :,
                                                                                                :, ] * 3
            io.imsave('./label' + str(step) + '.png', segresulti.data.cpu().numpy())

            segresulti = img[0, :, :,:, ]
            segresulti = segresulti.data.cpu().numpy()
            segresulti = np.transpose(segresulti, (1, 2, 0))
            io.imsave('./img' + str(step) + '.png', segresulti)

        loss = lossseg_ed_es + lossseg_ce
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2 == 0:
            torch.save(fusenet.state_dict(), './pkl/net_epoch_' + str(epoch+pretrained_num) + '-fuseNetwork.pkl')

        meansegdice += lossseg_ed_es.data.cpu().numpy()
        meaniou += criterion_iou.data.cpu.numpy()

        print('EPOCH:', epoch, '|Step:', step,
              '|loss_seg:', loss.data.cpu().numpy(),'|lossseg_ce:', lossseg_ce.data.cpu().numpy(),'|lossseg_f1:', lossseg_ed_es.data.cpu().numpy())

    print('epoch', epoch, '|meansegdice:',(meansegdice / step), '|Mean_iou:',(meaniou/ step))

