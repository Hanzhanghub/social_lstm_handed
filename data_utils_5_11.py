# coding:utf-8

'''
date: 2018/5/11
content:
复现social lstm。
5.10
第一步处理数据，我的思路是每个数据集创建一个dataset对象，然后再通过这个dataset去创建dataloader

5.11
修改一下5月10号的思路，在getitem的函数中不应该返回每个人的序列位置，
而是应该返回这个序列中所有的帧ID.
'''

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataProcess(Dataset):
    def __init__(self, dataset, is_test=False):
        super(DataProcess, self).__init__()

        self.is_test = is_test
        # 内部建立一个映射表，对应datasets中的数据集
        self.dataset_mapping = {0: './data/eth/hotel/pixel_pos_interpolate.csv',
                                1: './data/eth/univ/pixel_pos_interpolate.csv',
                                2: './data/ucy/univ/pixel_pos_interpolate.csv',
                                3: './data/ucy/zara/zara01/pixel_pos_interpolate.csv',
                                4: './data/ucy/zara/zara02/pixel_pos_interpolate.csv'}
        # 根据datasets的下标加载相应数据
        self.load_data(dataset)

        # 记录训练总数的长度
        self.length = 0

    def load_data(self, dataset_num):

        # 加载数据
        data = np.genfromtxt(self.dataset_mapping[dataset_num], delimiter=',')
        data = pd.DataFrame(data)

        # 找到这个数据集的所有帧ID
        self.frame_numbers = data.ix[0].unique()
        print('数据集{}共有{}帧'.format(dataset_num, self.frame_numbers.shape[0]))

        # 找到这个数据集中所有的行人ID
        self.id_numbers = data.ix[1].unique()
        print('数据集{}共有{}人'.format(dataset_num, self.id_numbers.shape[0]))

        # 找到每一帧中存在的行人ID以及每个行人的位置
        self.ped_per_frame = []
        self.ped_pos_per_frame = []
        for frame in self.frame_numbers:
            # 找到这一帧中的所有行人ID
            peds_in_frame = data.ix[1, data.ix[0] == frame]
            self.ped_per_frame.append(peds_in_frame)

            pos_info = []  # 表示这一帧中所有人的位置信息--(pedID, x, y)
            # 根据这些行人的ID找到他们的位置
            for id in peds_in_frame:
                x_of_ped = data.ix[3, data.ix[1] == id]
                y_of_ped = data.ix[2, data.ix[1] == id]
                pos_info.append((int(id), x_of_ped, y_of_ped))
            self.ped_pos_per_frame.append(pos_info)

        # 找到这个数据集中每个序列帧
        self.frame_seq = []  # 找法：还是通过每个人的id去找他的序列，
        # 但是这回不是保存每帧中的位置，而是保存每个帧ID

        for ped in self.id_numbers:
            frame_number = data.ix[0, data.ix[1] == ped]
            # 保存这个帧ID
            self.frame_seq.append((ped, frame_number))
        self.length = len(self.frame_seq)
        print('数据集{}中共有{}条轨迹序列'.format(dataset_num, len(self.frame_seq)))

        print('数据装载完毕')

    def __len__(self):
        return len(self.frame_seq)

    def __getitem__(self, item):
        ped_ID, frames = self.frame_seq[item]
        frames = frames.tolist()

        # 根据文章中的要求，
        # 对于训练数据，我们是取一帧一帧进行学习
        # 对于测试数据，我们选取前8帧，并预测后来的12帧
        if self.is_test:
            return ped_ID, frames
        else:
            # 对于一个固定的下标，我返回的就是训练和测试集的下标
            train_data = frames[:-1]
            valid_data = frames[1:]

            return train_data, valid_data


if __name__ == '__main__':
    dataset = DataProcess(4, False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    '''
    统计：
    0 - 390人
    1 - 360
    2 - 434
    3 - 148
    4 - 204
    '''

    # for ibatch, (train, valid) in enumerate(dataloader):
    #     print('train data is :', len(train))
    #     print('valid data is :', len(valid))
