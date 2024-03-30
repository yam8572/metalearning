import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))

        """
        - Meta Train : office 辦公室(156個)、hallway 走廊(61個)、storage 儲藏室(19個)、WC 衛生間(11個)、conference room 會議室(11個)   
        - Meta Test :  auditorium 禮堂(2個)、open space開放空間、lobby 大廳(3個)、lounge 休息室(3個)、, pantry 食品儲藏室(3個)、copy room 影印室(2個)
        """
        rooms_name = ['office', 'conferenceRoom', 'hallway', 'auditorium', 'openspace', 'lobby', 'lounge', 'pantry', 'copyRoom', 'storage', 'WC']
        self.train_rooms_name = ['office', 'hallway','storage','WC','conferenceRoom']
        self.test_rooms_name = list(set(rooms_name) - set(self.train_rooms_name))

        if split == 'train':
            rooms_split = [room for room in rooms if any(name in room for name in self.train_rooms_name)]

        else:
            rooms_split = [room for room in rooms if any(name in room for name in self.test_rooms_name)]

        # rooms = [room for room in rooms if 'Area_' in room]
        # pretrain_area = 'Area_5'
        # if split == 'train':
        #     rooms_split = [room for room in rooms if pretrain_area in room]
        # else:
        #     rooms_split = [room for room in rooms if not pretrain_area in room]


        # if split == 'train':
        #     rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        # else:
        #     rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            print("------------->", room_path)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        # print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)
    
class S3DISDatasetFewShot(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, block_size=1.0, num_episode=50000, n_way=3, k_shot=5, n_queries=1, num_class=13):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.n_way = n_way # should small than n_way <= 5
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.num_episode = num_episode
        self.split = split
        self.data_root = data_root
        self.NUM_CLASSES=num_class
        self.labelweights = np.zeros(self.NUM_CLASSES)

        rooms = sorted(os.listdir(data_root))
        """
        - Meta Train : office 辦公室(156個)、hallway 走廊(61個)、storage 儲藏室(19個)、WC 衛生間(11個)、conference room 會議室(11個)   
        - Meta Test :  auditorium 禮堂(2個)、open space開放空間、lobby 大廳(3個)、lounge 休息室(3個)、, pantry 食品儲藏室(3個)、copy room 影印室(2個)
        """
        rooms_name = ['office', 'conferenceRoom', 'hallway', 'auditorium', 'openspace', 'lobby', 'lounge', 'pantry', 'copyRoom', 'storage', 'WC']
        # meta train & meta test 皆用 S3DIS >> sample 數多的當meta train 
        self.train_rooms_name = ['office', 'hallway','storage','WC','conferenceRoom']
        # cross dataset S3DIS:meta train ScanNet: meta test >> 改選13物件都有的
        # self.train_rooms_name = ['office', 'hallway','storage','lobby','conferenceRoom']

        self.test_rooms_name = list(set(rooms_name) - set(self.train_rooms_name))
        assert n_way <= min(len(self.train_rooms_name),len(self.test_rooms_name)), "n_way too big"

        if split == 'train':
            self.rooms_split = [room for room in rooms if any(name in room for name in self.train_rooms_name)]

        else:
            self.rooms_split = [room for room in rooms if any(name in room for name in self.test_rooms_name)]

    def __getitem__(self, idx, n_way_train_scenes=None, n_way_test_scenes=None): 

        if n_way_train_scenes is not None: # n_way_scenes 指定 scenes
            self.meta_train_rooms_name = np.array(n_way_train_scenes)
        else:
            self.meta_train_rooms_name = sorted(np.random.choice(self.train_rooms_name, self.n_way ,replace=False),key=str.casefold)
        
        if n_way_test_scenes is not None: # n_way_scenes 指定 scenes
            self.meta_test_rooms_name = np.array(n_way_test_scenes)
        else:
            self.meta_test_rooms_name = sorted(np.random.choice(self.test_rooms_name, self.n_way ,replace=False),key=str.casefold)
        
        if self.split == 'train':
            support_ptclouds, support_labels, query_ptclouds, query_labels = self.generate_one_episode(self.meta_train_rooms_name)
        else:
            support_ptclouds, support_labels, query_ptclouds, query_labels = self.generate_one_episode(self.meta_test_rooms_name)
        return support_ptclouds, support_labels, query_ptclouds, query_labels

    def __len__(self):
        return self.num_episode
        # return len(self.room_idxs)
    def generate_one_episode(self, meta_rooms_name):
        support_ptclouds = []
        support_labels = []
        query_ptclouds = []
        query_labels = []

        selected_roomnames=[]
        for i in range(self.n_way):
            tmp=[room for room in self.rooms_split if meta_rooms_name[i] in room]
            # 場景數不夠需重複取
            while len(tmp) < (self.k_shot+self.n_queries):
                tmp=tmp*2
            selected_roomnames.append(np.random.choice(tmp, self.k_shot + self.n_queries ,replace=False))
        selected_roomnames=np.array(selected_roomnames)
        selected_roomnames=selected_roomnames.reshape(self.n_way*(self.k_shot+self.n_queries)).flatten()
        """
        順序 ex. 2shot 1query
        selected_roomnames = [support,support,query]
        """
        c=0
        labelweights = np.zeros(self.NUM_CLASSES)
        for room_name in tqdm(selected_roomnames, total=len(selected_roomnames)):
            room_path = os.path.join(self.data_root, room_name)
            # print("------------->", room_path)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            coord_max = np.amax(points, axis=0)[:3]
            select_points,select_labels=self.sample_block_point(points, labels, coord_max)
            tmp, _ = np.histogram(labels, range(self.NUM_CLASSES+1))
            labelweights += tmp
            # print("select_points",len(select_points))
            # print("select_labels",len(select_labels))
            if c < self.k_shot:
                # print("support:",room_name)
                support_ptclouds.append(select_points)
                support_labels.append(select_labels) 
            else:
                # print("query:",room_name)
                query_ptclouds.append(select_points)
                query_labels.append(select_labels)
            c+=1
            if(c % (self.k_shot+self.n_queries) == 0):
                c=0
        labelweights = labelweights.astype(np.float32)
        self.labelweights = labelweights / np.sum(labelweights)
        # # Checking for zero values
        # zero_mask = labelweights == 0
        # # Adding a small epsilon to avoid division by zero
        # epsilon = 1e-10
        # labelweights[zero_mask] += epsilon
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        # print(self.labelweights)
        return support_ptclouds, support_labels, query_ptclouds, query_labels
    """
    隨機找出一塊寬高均為block_size大小的區域，並且要求該區域的點數要大於1024 ，否則就得重新選擇新的區域。
    因為隨機選取中心點的時候有可能找到角落上的點，第一個會導致選取的這塊區域不是block_size大小的正方形區域，
    第二是如果該區域的點太少了，即便後面可以重複取點到4096 ，但是網路取得到的有效點就少了，影響網路的訓練。
    """
    def sample_block_point(self, points, labels, coord_max):
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

class ScannetDatasetFewShot(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, block_size=1.0, num_episode=50000, n_way=4, k_shot=2, n_queries=2, num_class=21):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.n_way = n_way # should small than n_way <= 5
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.num_episode = num_episode
        self.split = split
        self.data_root = data_root
        self.NUM_CLASSES=num_class
        self.labelweights = np.zeros(self.NUM_CLASSES)


        rooms = sorted(os.listdir(data_root))
        """
        ['Apartment' 'Bathroom' 'Bedroom' 'Bookstore' 'Classroom' 'Closet'
        'ComputerCluster' 'ConferenceRoom' 'CopyRoom' 'DiningRoom' 'GameRoom'
        'Gym' 'Hallway' 'Kitchen' 'LaundryRoom' 'LivingRoom' 'Lobby' 'Mailboxes'
        'Misc' 'Office' 'Stairs' 'Storage']
        - Meta Train : Apartment 公寓(40個)、Bathroom 浴室(211個)、Bedroom 寢室(273個)、DiningRoom 客廳(14個)??、Kitchen 廚房(108個)、LaundryRoom 洗衣房(27個)?、Lobby 大堂(54個)
        - Meta Test : Bookstore 書店(67個)、Classroom 教室(37個)、Closet衣櫥(8個)、ComputerCluster電腦教室(5個)、conferenceRoom 會議室(111個)、copyRoom 影印室(42個)、GameRoom 遊戲室(11個)、Gym 健身房(4個)、Hallway 門廳(32個)、LivingRoom 客廳(221個)、Mailboxes 信箱(8個)、Misc雜物間(35個)、Office辦公室(172個) 、Stairs樓梯(14個)、Storage 儲藏室(19個)
        """
        rooms_name = ['Apartment','Bathroom','Bedroom', 'Bookstore','Classroom','Closet',
        'ComputerCluster','ConferenceRoom','CopyRoom','DiningRoom','GameRoom',
        'Gym','Hallway','Kitchen','LaundryRoom','LivingRoom','Lobby','Mailboxes',
        'Misc','Office','Stairs','Storage']
        # self.train_rooms_name = ['Apartment','Bathroom','Bookstore','Classroom','Gym','Hallway','Kitchen','LivingRoom','Office']
        # 和 S3DIS 不同標籤的場景
        # self.train_rooms_name = ['Apartment','Bathroom','Bedroom','DiningRoom','Kitchen','LaundryRoom','Lobby']
        # self.train_rooms_name = ['Apartment','Bathroom','Bedroom','Bookstore','Classroom','ConferenceRoom','DiningRoom','Kitchen','LivingRoom','Lobby','Office']
        self.train_rooms_name = ['Apartment','Bathroom','Bedroom','Bookstore','Classroom','ConferenceRoom','CopyRoom','Kitchen','LivingRoom','Lobby','Office']
        
        # 找多 floor wall window door table chair sofa bookcase clutter 和S3DIS多相同的場景測試相似資料集的泛化性
        # self.train_rooms_name = ['Apartment','Bookstore','Bedroom','Classroom','ConferenceRoom','DiningRoom','Kitchen','LivingRoom','Misc','Office','LaundryRoom','Lobby']
        self.test_rooms_name = list(set(rooms_name) - set(self.train_rooms_name))
        assert n_way <= min(len(self.train_rooms_name),len(self.test_rooms_name)), "n_way too big"

        if split == 'train':
            self.rooms_split = [room for room in rooms if any(name in room for name in self.train_rooms_name)]

        else:
            self.rooms_split = [room for room in rooms if any(name in room for name in self.test_rooms_name)]

    def __getitem__(self, idx, n_way_train_scenes=None, n_way_test_scenes=None): 

        if n_way_train_scenes is not None: # n_way_scenes 指定 scenes
            self.meta_train_rooms_name = np.array(n_way_train_scenes)
        else:
            self.meta_train_rooms_name = sorted(np.random.choice(self.train_rooms_name, self.n_way ,replace=False),key=str.casefold)
        
        if n_way_test_scenes is not None: # n_way_scenes 指定 scenes
            self.meta_test_rooms_name = np.array(n_way_test_scenes)
        else:
            self.meta_test_rooms_name = sorted(np.random.choice(self.test_rooms_name, self.n_way ,replace=False),key=str.casefold)
        
        if self.split == 'train':
            support_ptclouds, support_labels, query_ptclouds, query_labels = self.generate_one_episode(self.meta_train_rooms_name)
        else:
            support_ptclouds, support_labels, query_ptclouds, query_labels = self.generate_one_episode(self.meta_test_rooms_name)

        # print("meta_train_rooms_name=",self.meta_train_rooms_name)
        # print("meta_test_rooms_name=",self.meta_test_rooms_name)

        return support_ptclouds, support_labels, query_ptclouds, query_labels

    def __len__(self):
        return self.num_episode
        # return len(self.room_idxs)
    def generate_one_episode(self, meta_rooms_name):
        support_ptclouds = []
        support_labels = []
        query_ptclouds = []
        query_labels = []

        selected_roomnames=[]
        for i in range(self.n_way):
            tmp=[room for room in self.rooms_split if meta_rooms_name[i] in room]
            # 場景數不夠需重複取
            while len(tmp) < (self.k_shot+self.n_queries):
                tmp=tmp*2
            selected_roomnames.append(np.random.choice(tmp, self.k_shot + self.n_queries ,replace=False))
        selected_roomnames=np.array(selected_roomnames)
        selected_roomnames=selected_roomnames.reshape(self.n_way*(self.k_shot+self.n_queries)).flatten()
        """
        順序 ex. 2shot 1query
        selected_roomnames = [support,support,query]
        """
        c=0
        labelweights = np.zeros(self.NUM_CLASSES)
        for room_name in tqdm(selected_roomnames, total=len(selected_roomnames)):
            room_path = os.path.join(self.data_root, room_name)
            # print("------------->", room_path)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            coord_max = np.amax(points, axis=0)[:3]
            select_points,select_labels=self.sample_block_point(points, labels, coord_max)
            tmp, _ = np.histogram(labels, range(self.NUM_CLASSES+1))
            labelweights += tmp
            # print("select_points",len(select_points))
            # print("select_labels",len(select_labels))
            if c < self.k_shot:
                # print("support:",room_name)
                support_ptclouds.append(select_points)
                support_labels.append(select_labels) 
            else:
                # print("query:",room_name)
                query_ptclouds.append(select_points)
                query_labels.append(select_labels)
            c+=1
            if(c % (self.k_shot+self.n_queries) == 0):
                c=0
        labelweights = labelweights.astype(np.float32)
        self.labelweights = labelweights / np.sum(labelweights)
        # # Checking for zero values
        # zero_mask = labelweights == 0
        # # Adding a small epsilon to avo/home/g111056119/Documents/7111056426/Reptile-Pytorch/log/sem_seg/8batches_4ways_2shots_scannet_v2id division by zero
        # epsilon = 1e-10
        # labelweights[zero_mask] += epsilon
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        # print(self.labelweights)
        return support_ptclouds, support_labels, query_ptclouds, query_labels
    """
    隨機找出一塊寬高均為block_size大小的區域，並且要求該區域的點數要大於1024 ，否則就得重新選擇新的區域。
    因為隨機選取中心點的時候有可能找到角落上的點，第一個會導致選取的這塊區域不是block_size大小的正方形區域，
    第二是如果該區域的點太少了，即便後面可以重複取點到4096 ，但是網路取得到的有效點就少了，影響網路的訓練。
    """
    def sample_block_point(self, points, labels, coord_max):
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001, num_class=13):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        self.NUM_CLASSES = num_class
        
        rooms = sorted(os.listdir(root))
        rooms = [room for room in rooms if 'Area_' in room]
        """
        - Meta Train : office 辦公室(156個)、hallway 走廊(61個)、storage 儲藏室(19個)、WC 衛生間(11個)、conference room 會議室(11個)   
        - Meta Test :  auditorium 禮堂(2個)、open space開放空間、lobby 大廳(3個)、lounge 休息室(3個)、, pantry 食品儲藏室(3個)、copy room 影印室(2個)
        """
        rooms_name = ['office', 'conferenceRoom', 'hallway', 'auditorium', 'openspace', 'lobby', 'lounge', 'pantry', 'copyRoom', 'storage', 'WC']


        self.train_rooms_name = ['office', 'hallway','storage','WC','conferenceRoom']
        self.test_rooms_name = list(set(rooms_name) - set(self.train_rooms_name))
        print("self.test_rooms_name",self.test_rooms_name)

        if split == 'train':
            self.file_list = [room for room in rooms if any(name in room for name in self.train_rooms_name)]

        else:
            self.file_list = [room for room in rooms if any(name in room for name in self.test_rooms_name)]
        # print("self.file_list",self.file_list)

        # assert split in ['train', 'test']
        # if self.split == 'train':
        #     self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        # else:
        #     self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(self.NUM_CLASSES)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(self.NUM_CLASSES+1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        print("labelweights",labelweights)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetWholeScene_scannet():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', stride=0.5, block_size=1.0, padding=0.001, num_class=21):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        self.NUM_CLASSES = num_class
        rooms = sorted(os.listdir(root))
        """
        - Meta Train : Apartment 公寓(40個)、Bathroom 浴室(211個)、Bedroom 寢室(273個)、DiningRoom 客廳(14個)??、Kitchen 廚房(108個)、LaundryRoom 洗衣房(27個)?、Lobby 大堂(54個)
        - Meta Test : Bookstore 書店(67個)、Classroom 教室(37個)、Closet衣櫥(8個)、ComputerCluster電腦教室(5個)、conferenceRoom 會議室(111個)、copyRoom 影印室(42個)、GameRoom 遊戲室(11個)、Gym 健身房(4個)、Hallway 門廳(32個)、LivingRoom 客廳(221個)、Mailboxes 信箱(8個)、Misc雜物間(35個)、Office辦公室(172個) 、Stairs樓梯(14個)、Storage 儲藏室(19個)
        """
        rooms_name = ['Apartment','Bathroom','Bedroom', 'Bookstore','Classroom','Closet',
        'ComputerCluster','ConferenceRoom','CopyRoom','DiningRoom','GameRoom',
        'Gym','Hallway','Kitchen','LaundryRoom','LivingRoom','Lobby','Mailboxes',
        'Misc','Office','Stairs','Storage']

        # 找多 floor wall window door table chair sofa bookcase clutter 和S3DIS多相同的場景測試相似資料集的泛化性
        # self.train_rooms_name = ['Apartment','Bookstore','Bedroom','Classroom','ConferenceRoom','DiningRoom','Kitchen','LivingRoom','Misc','Office','LaundryRoom','Lobby']
        # 和 S3DIS 不同標籤的場景
        # self.train_rooms_name = ['Apartment','Bathroom','Bedroom','Bookstore','Classroom','ConferenceRoom','DiningRoom','Kitchen','LivingRoom','Lobby','Office']
        self.train_rooms_name = ['Apartment','Bathroom','Bedroom','Bookstore','Classroom','ConferenceRoom','CopyRoom','Kitchen','LivingRoom','Lobby','Office']
        self.test_rooms_name = list(set(rooms_name) - set(self.train_rooms_name))

        if split == 'train':
            self.file_list = [room for room in rooms if any(name in room for name in self.train_rooms_name)]

        else:
            self.file_list = [room for room in rooms if any(name in room for name in self.test_rooms_name)]
        

        # assert split in ['train', 'test']
        # if self.split == 'train':
        #     self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        # else:
        #     self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(self.NUM_CLASSES)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(self.NUM_CLASSES+1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()