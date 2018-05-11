# coding:utf-8

'''
date: 2018/5/10
content:
复现social lstm。
第一步处理数据，我的思路是每个数据集创建一个dataset对象，然后再通过这个dataset去创建dataloader
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
        self.raw_data = self.load_data(dataset)

        # 记录训练总数的长度
        self.length = 0

    def load_data(self, dataset_num):

        # 加载数据
        data = np.genfromtxt(self.dataset_mapping[dataset_num], delimiter=',')

        # 找到这个数据集的所有帧ID
        self.frame_numbers = data.ix[0].unique()

        # 找到这个数据集中所有的行人ID
        self.id_numbers = data.ix[1].unique()

        # 找到每一帧中存在的行人ID
        self.ped_per_frame = []
        self.ped_pos_per_fram = []
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
            self.ped_pos_per_fram.append(pos_info)

        # 找到这个数据集中每个人的序列
        self.ped_seq = []  # 每一个人的轨迹序列
        self.length = id.number.shape[0]
        for ped in self.id_numbers:
            ped_x = data.ix[3, data.ix[1] == ped]
            ped_y = data.ix[2, data.ix[1] == ped]

            # todo: 关于序列的返回
            # 我暂时先用两种方法，其一是返回一个DataFrame
            # 其二是返回一个zip后的对象，然后在getitem的时候在unzip
            # x_y = pd.concat([ped_x, ped_y], axis=1)
            x_y = zip(ped_x, ped_y)
            self.ped_seq.append((ped, x_y))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        ped_ID, xy = self.ped_seq[item]  # 这里xy是一个DataFrame对象(-1, 2)

        # 根据文章中的要求
        # 对于训练数据，我们是取一帧一帧进行学习，
        # 对于测试数据，我们选取前8帧，并预测后来的12帧
        if self.is_test:
            pass
        else:
            # todo: 这里我就有一个问题了，
            # 我应该怎么加载数据集：1.返回训练数据和验证数据
            seq_data = []
            for ped_x, ped_y in xy:
                seq_data.append((ped_x, ped_y))

            train_data = seq_data[: -1]
            valid_data = seq_data[1:]

        return train_data, valid_data


if __name__ == '__main__':
    dataset = DataProcess([0, 1, 2], False)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=-1)

    for train, valid in dataloader:
        print('train data is :', train)
        print('valid data is :', valid)
