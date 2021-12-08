from .sim_ghost import sim_ghost
import torch
import numpy as np
import pickle
from torchvision import transforms

from typing import List

classid_list = [0,1,2,3]

model : sim_ghost = sim_ghost(classid_list=classid_list, pretrain=False)

state_dict = torch.load('Step3/sim_ghost_params.pth')

model.load_state_dict(state_dict=state_dict)

with open('Step3/thresholds.txt', 'rb') as fr:
    model.classwise_thresholds = pickle.load(fr)

model.cuda()


id_map = ['sly_bjbmyw', 'yw_gkxfw', 'sly_dmyw', 'yw_nc']


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def api(cropped_list: List[np.ndarray]) -> List[int]:
    inputs = list(map(test_transform, cropped_list))

    inputs = np.stack(inputs, axis=0)
    return model.predict(torch.from_numpy(inputs).cuda())
