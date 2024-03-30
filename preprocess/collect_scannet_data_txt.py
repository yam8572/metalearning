""" Collect point clouds and the corresponding labels from original ScanNetV2 dataset, and save into numpy files.

Author: Zhao Na, 2020
"""
import os
import sys
import json
import numpy as np
from plyfile import PlyData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def get_raw2scannet_label_map(label_mapping_file):
    lines = [line.rstrip() for line in open(label_mapping_file)]
    lines = lines[1:]
    raw2scannet = {}
    label_classes_set = set(CLASS_NAMES)
    print("label_classes_set=",label_classes_set)
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            # raw2scannet[raw_name] = 'unannotated'
            raw2scannet[raw_name] = 'clutter'

            if(raw_name=='shower doors'):raw2scannet[raw_name] = 'door'
            elif(raw_name=='windowsill'):raw2scannet[raw_name] = 'window'
            elif(raw_name=='bookshelf'):raw2scannet[raw_name] = 'bookcase' # 因依照s3dis一起命名 bookcase改成scannet bookshelf 才會標注到
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = np.round(plydata['vertex'].data['x'],5)
        vertices[:,1] = np.round(plydata['vertex'].data['y'],5)
        vertices[:,2] = np.round(plydata['vertex'].data['z'],5)
        vertices[:,3] = np.round(plydata['vertex'].data['red'],1)
        vertices[:,4] = np.round(plydata['vertex'].data['green'],1)
        vertices[:,5] = np.round(plydata['vertex'].data['blue'],1)
    return vertices


def collect_point_label(scene_path, scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    mesh_seg_filename = os.path.join(scene_path, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))
    # print mesh_seg_filename
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        # print len(seg)
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    # Raw points in XYZRGBA
    ply_filename = os.path.join(scene_path, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgb(ply_filename)
    
    print('{0}: {1} points'.format(scene_name, points.shape[0]))

    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(scene_path, '%s.aggregation.json' % (scene_name))
    # print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])

    # print len(instance_segids)
    # print labels

    # Each instance's points
    instance_points_list = []
    # instance_labels_list = []
    semantic_labels_list = []

    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids), :]
        instance_points_list.append(instance_points)
        # instance_labels_list.append(np.ones((instance_points.shape[0], 1)) * i)
        if labels[i] not in RAW2SCANNET:
            # label = 'unannotated'
            label = 'clutter'

        else:
            label = RAW2SCANNET[labels[i]]
        # print("label=",label)
        # print("CLASS_NAMES=",CLASS_NAMES)

        label = int(CLASS_NAMES.index(label))
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1)) * label)

    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:, 0:6]  # XYZRGB, disregarding the A
    # instance_labels = np.concatenate(instance_labels_list, 0)
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    # data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)
    data = np.concatenate((scene_points, semantic_labels), 1)
    # print("out_filename=",out_filename)
    # if not os.path.exists(out_filename): os.makedirs(out_filename)
    np.savetxt(out_filename, data)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/ScanNet/scans',
                        help='Directory to dataset')
    args = parser.parse_args()

    """
    DATA_PATH= datasets/ScanNet/scans
    DST_PATH= /home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/ScanNet
    SAVE_PATH= /home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/ScanNet/scenes/data
    """
    
    # DATA_PATH = args.data_path
    # DST_PATH = os.path.join(ROOT_DIR, 'datasets/ScanNet')
    # SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')

    DATA_PATH='/media/g111056119/SP PHD U3/datasets/ScanNet/scans'
    SAVE_PATH='/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_21_txt'
    print("DATA_PATH=",DATA_PATH)
    print("SAVE_PATH=",SAVE_PATH)

    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    # meta_path = os.path.join(DST_PATH, 'meta')
    meta_path='/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/ScanNet/meta/'
    # CLASS_NAMES = [x.rstrip() for x in open(os.path.join(meta_path, 'scannet_classnames_yun.txt'))]
    # CLASS_NAMES = [x.rstrip() for x in open('/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/class_names.txt')]
    # CLASS_NAMES = [x.rstrip() for x in open('/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/ScanNet/meta/scannet_classnames_13.txt')]
    CLASS_NAMES = [x.rstrip() for x in open('/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/ScanNet/meta/scannet_classnames_yun.txt')]
    # CLASS_NAMES = [x.rstrip() for x in open('/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/class_names_9_s3dis_scannet.txt')]
    print("class_name",CLASS_NAMES)


    label_mapping_file = os.path.join(meta_path, 'scannetv2-labels.combined.tsv')
    RAW2SCANNET = get_raw2scannet_label_map(label_mapping_file)
    # print("RAW2SCANNET=",RAW2SCANNET)
    
    scene_paths = sorted([os.path.join(DATA_PATH, o) for o in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, o))])
    # print(scene_paths[:51])
    # n_scenes = len(scene_paths[:101])
    n_scenes = len(scene_paths)

    if (n_scenes == 0):
        raise ValueError('%s is empty' % DATA_PATH)
    else:
        print('%d scenes to be processed...' % n_scenes)
    
    all_scene_names = {'Apartment': ['scene0000_00', 'scene0000_01', 'scene0000_02', 'scene0046_00', 'scene0046_01','scene0046_02','scene0054_00','scene0101_00','scene0101_01','scene0101_02','scene0101_03','scene0101_04','scene0101_05','scene0207_00','scene0207_01','scene0207_02','scene0231_00','scene0231_01','scene0231_02','scene0276_00','scene0276_01','scene0288_00','scene0288_01','scene0288_02','scene0327_00','scene0392_00','scene0392_01','scene0392_02','scene0419_00','scene0419_01','scene0419_02','scene0472_00','scene0472_01','scene0472_02','scene0504_00','scene0515_00','scene0515_01','scene0515_02','scene0673_04','scene0673_05'],
                   'LivingRoom': ['scene0001_00','scene0001_01','scene0002_00','scene0002_01','scene0012_00','scene0012_01','scene0012_02','scene0013_00','scene0013_01','scene0013_02','scene0021_00','scene0022_00','scene0022_01','scene0024_00','scene0024_01','scene0024_02','scene0027_00','scene0027_01','scene0027_02','scene0031_00','scene0031_01','scene0031_02','scene0033_00','scene0036_00','scene0036_01','scene0045_00','scene0045_01','scene0049_00','scene0050_00','scene0050_01','scene0050_02','scene0059_00','scene0059_01','scene0059_02','scene0061_00','scene0061_01','scene0081_00','scene0082_00','scene0087_00','scene0087_01','scene0087_02','scene0091_00','scene0113_00','scene0113_01','scene0125_00','scene0128_00','scene0129_00','scene0148_00','scene0150_00','scene0150_01','scene0150_02','scene0151_00','scene0151_01','scene0152_00','scene0152_01','scene0152_02','scene0160_00','scene0160_01','scene0160_02','scene0160_03','scene0160_04','scene0161_00','scene0161_01','scene0161_02','scene0162_00','scene0165_00','scene0165_01','scene0165_02','scene0192_00','scene0192_01','scene0192_02','scene0196_00','scene0203_00','scene0203_01','scene0203_02','scene0212_00','scene0212_01','scene0212_02','scene0213_00','scene0226_00','scene0234_00','scene0239_00','scene0239_01','scene0239_02','scene0244_00','scene0244_01','scene0247_00,''scene0247_01','scene0248_00','scene0248_01','scene0248_02','scene0273_00','scene0273_01','scene0275_00','scene0281_00','scene0294_00','scene0294_01','scene0294_02','scene0297_00','scene0297_01','scene0297_02','scene0299_00','scene0299_01','scene0301_00','scene0301_01','scene0301_02','scene0313_00','scene0313_01','scene0313_02','scene0315_00','scene0316_00','scene0319_00','scene0322_00','scene0324_00','scene0324_01','scene0326_00','scene0330_00','scene0334_00','scene0334_01','scene0334_02','scene0339_00','scene0344_00','scene0344_01','scene0362_00','scene0362_01','scene0362_02','scene0362_03','scene0367_00','scene0367_01','scene0368_00','scene0368_01','scene0371_00','scene0371_01','scene0385_00','scene0385_01','scene0385_02','scene0386_00','scene0387_00','scene0387_01','scene0387_02','scene0407_00','scene0407_01','scene0423_00','scene0423_01','scene0423_02','scene0432_00','scene0432_01','scene0437_00','scene0437_01','scene0442_00','scene0467_00','scene0480_00','scene0480_01','scene0497_00','scene0499_00','scene0503_00','scene0513_00','scene0514_00','scene0514_01','scene0518_00','scene0520_00','scene0520_01','scene0526_00','scene0526_01','scene0537_00','scene0559_00','scene0559_01','scene0559_02','scene0568_00','scene0568_01','scene0568_02','scene0572_00','scene0572_01','scene0572_02','scene0587_00','scene0587_01','scene0587_02','scene0587_03','scene0588_00','scene0588_01','scene0588_02','scene0588_03','scene0593_00','scene0593_01','scene0594_00','scene0603_00','scene0603_01','scene0604_00','scene0604_01','scene0604_02','scene0608_00','scene0608_01','scene0608_02','scene0609_00','scene0609_01','scene0609_02','scene0609_03','scene0616_00','scene0616_01','scene0617_00','scene0622_00','scene0622_01','scene0623_00','scene0623_01','scene0638_00','scene0646_00','scene0646_01','scene0646_02','scene0647_00','scene0647_01','scene0660_00','scene0668_00','scene0672_00','scene0672_01','scene0673_00','scene0673_01','scene0673_02','scene0673_03','scene0694_00','scene0694_01','scene0701_00','scene0701_01','scene0701_02'],
                   'Kitchen': ['scene0003_00','scene0003_01','scene0003_02','scene0011_00','scene0011_01','scene0047_00','scene0063_00','scene0068_00','scene0068_01','scene0076_00','scene0094_00','scene0097_00','scene0111_00','scene0111_01','scene0111_02','scene0117_00','scene0149_00','scene0158_00','scene0158_01','scene0158_02','scene0164_00','scene0164_01','scene0164_02','scene0164_03','scene0181_00','scene0181_01','scene0181_02','scene0181_03','scene0197_00','scene0197_01','scene0197_02','scene0198_00','scene0201_00','scene0201_01','scene0201_02','scene0210_00','scene0210_01','scene0220_00','scene0220_01','scene0220_02','scene0310_00','scene0310_01','scene0310_02','scene0323_00','scene0323_01','scene0328_00','scene0335_00','scene0335_01','scene0335_02','scene0347_00','scene0347_01','scene0347_02','scene0348_00','scene0348_01','scene0348_02','scene0390_00','scene0409_00','scene0409_01','scene0412_00','scene0412_01','scene0433_00','scene0436_00','scene0439_00','scene0439_01','scene0465_00','scene0465_01','scene0479_00','scene0479_01','scene0479_02','scene0486_00','scene0487_00','scene0487_01','scene0488_00','scene0488_01','scene0498_00','scene0498_01','scene0498_02','scene0522_00','scene0529_00','scene0529_01','scene0529_02','scene0532_00','scene0532_01','scene0540_00','scene0540_01','scene0540_02','scene0542_00','scene0552_00','scene0552_01','scene0573_00','scene0573_01','scene0586_00','scene0586_01','scene0586_02','scene0607_00','scene0607_01','scene0649_00','scene0649_01','scene0651_00','scene0651_01','scene0651_02','scene0654_00','scene0654_01','scene0670_00','scene0670_01','scene0681_00','scene0682_00','scene0687_00'],
                   'Hallway': ['scene0004_00','scene0007_00','scene0019_00','scene0019_01','scene0043_00','scene0043_01','scene0060_00','scene0060_01','scene0071_00','scene0079_01','scene0118_00','scene0118_01','scene0118_02','scene0133_00','scene0242_00','scene0242_01','scene0242_02','scene0251_00','scene0267_00','scene0290_00','scene0293_00','scene0293_01','scene0374_00','scene0431_00','scene0473_00','scene0473_01','scene0530_00','scene0543_00','scene0543_01','scene0543_02','scene0703_00','scene0703_01'],
                   'Misc': ['scene0005_00','scene0005_01','scene0023_00','scene0099_00','scene0099_01','scene0159_00','scene0183_00','scene0191_00','scene0191_01','scene0191_02','scene0194_00','scene0284_00','scene0325_00','scene0325_01','scene0337_00','scene0337_01','scene0337_02','scene0343_00','scene0370_00','scene0370_01','scene0370_02','scene0405_00','scene0443_00','scene0444_00','scene0444_01','scene0459_00','scene0459_01','scene0491_00','scene0531_00','scene0565_00','scene0629_00','scene0629_01','scene0629_02','scene0632_00','scene0636_00'],
                   'Bedroom': ['scene0006_00','scene0006_01','scene0006_02','scene0032_00','scene0032_01','scene0035_00','scene0035_01','scene0039_00','scene0039_01','scene0051_00','scene0051_01','scene0051_02','scene0051_03','scene0055_00','scene0055_01','scene0055_02','scene0070_00','scene0072_00','scene0072_01','scene0072_02','scene0073_00','scene0073_01','scene0073_02','scene0073_03','scene0078_00','scene0078_01','scene0078_02','scene0085_00','scene0085_01','scene0092_00','scene0092_01','scene0092_02','scene0092_03','scene0092_04','scene0107_00','scene0110_00','scene0110_01','scene0110_02','scene0127_00','scene0127_01','scene0137_00','scene0137_01','scene0137_02','scene0143_00','scene0143_01','scene0143_02','scene0144_00','scene0144_01','scene0147_00','scene0147_01','scene0157_00','scene0157_01','scene0172_00','scene0172_01','scene0178_00','scene0184_00','scene0193_00','scene0193_01','scene0200_00','scene0200_01','scene0200_02','scene0209_00','scene0209_01','scene0209_02','scene0217_00','scene0219_00','scene0221_00','scene0221_01','scene0222_00','scene0222_01','scene0223_00','scene0223_01','scene0223_02','scene0224_00','scene0233_00','scene0233_01','scene0235_00','scene0236_00','scene0236_01','scene0243_00','scene0246_00','scene0253_00','scene0256_02','scene0259_00','scene0259_01','scene0270_00','scene0270_01','scene0270_02','scene0271_00','scene0271_01','scene0277_00','scene0277_01','scene0277_02','scene0279_00','scene0279_01','scene0279_02','scene0280_00','scene0280_01','scene0280_02','scene0282_00','scene0282_01','scene0282_02','scene0286_00','scene0286_01','scene0286_02','scene0286_03','scene0287_00','scene0295_00','scene0295_01','scene0296_00','scene0296_01','scene0298_00','scene0306_00','scene0306_01','scene0308_00','scene0321_00','scene0331_00','scene0331_01','scene0332_00','scene0332_01','scene0332_02','scene0340_00','scene0340_01','scene0340_02','scene0353_00','scene0353_01','scene0353_02','scene0356_00','scene0356_01','scene0356_02','scene0359_00','scene0359_01','scene0364_00','scene0364_01','scene0376_00','scene0376_01','scene0376_02','scene0382_00','scene0382_01','scene0389_00','scene0394_00','scene0394_01','scene0397_00','scene0397_01','scene0398_00','scene0398_01','scene0400_00','scene0400_01','scene0416_00','scene0416_01','scene0416_02','scene0416_03','scene0416_04','scene0418_00','scene0418_01','scene0418_02','scene0425_00','scene0425_01','scene0426_00','scene0426_01','scene0426_02','scene0426_03','scene0435_00','scene0435_01','scene0435_02','scene0435_03','scene0438_00','scene0440_00','scene0440_01','scene0440_02','scene0445_00','scene0445_01','scene0457_00','scene0457_01','scene0457_02','scene0468_00','scene0468_01','scene0468_02','scene0477_00','scene0477_01','scene0506_00','scene0507_00','scene0517_00','scene0517_01','scene0517_02','scene0521_00','scene0541_00','scene0541_01','scene0541_02','scene0545_00','scene0545_01','scene0545_02','scene0547_00','scene0547_01','scene0547_02','scene0548_00','scene0548_01','scene0548_02','scene0554_00','scene0554_01','scene0556_00','scene0556_01','scene0557_00','scene0557_01','scene0557_02','scene0560_00','scene0567_00','scene0567_01','scene0576_00','scene0576_01','scene0576_02','scene0577_00','scene0580_00','scene0580_01','scene0589_00','scene0589_01','scene0589_02','scene0590_00','scene0590_01','scene0600_00','scene0600_01','scene0600_02','scene0613_00','scene0613_01','scene0613_02','scene0615_00','scene0615_01','scene0620_00','scene0620_01','scene0631_01','scene0631_02','scene0633_00','scene0633_01','scene0640_00','scene0640_01','scene0640_02','scene0642_00','scene0642_01','scene0642_02','scene0642_03','scene0645_00','scene0645_01','scene0645_02','scene0648_00','scene0648_01','scene0652_00','scene0656_00','scene0656_01','scene0656_02','scene0656_03','scene0658_00','scene0674_00','scene0674_01','scene0675_00','scene0675_01','scene0677_00','scene0677_01','scene0677_02','scene0679_00','scene0679_01','scene0688_00','scene0695_00','scene0695_01','scene0695_02','scene0695_03','scene0697_00','scene0697_01','scene0697_02','scene0697_03','scene0698_00','scene0698_01','scene0699_00','scene0706_00'],
                   'Lobby': ['scene0008_00','scene0028_00','scene0048_00','scene0048_01','scene0053_00','scene0081_01','scene0081_02','scene0132_00','scene0132_01','scene0132_02','scene0134_00','scene0134_01','scene0134_02','scene0145_00','scene0199_00','scene0202_00','scene0228_00','scene0229_00','scene0229_01','scene0229_02','scene0269_00','scene0269_01','scene0269_02','scene0292_00','scene0292_01','scene0312_00','scene0312_01','scene0312_02','scene0314_00','scene0317_00','scene0317_01','scene0329_00','scene0329_01','scene0329_02','scene0355_00','scene0355_01','scene0363_00','scene0381_00','scene0381_01','scene0381_02','scene0450_00','scene0461_00','scene0475_00','scene0475_01','scene0475_02','scene0536_00','scene0536_01','scene0536_02','scene0549_00','scene0549_01','scene0637_00','scene0641_00','scene0690_00','scene0690_01'],
                   'Bathroom': ['scene0009_00','scene0009_01','scene0009_02','scene0014_00','scene0026_00','scene0029_00','scene0029_01','scene0029_02','scene0034_00','scene0034_01','scene0034_02','scene0042_00','scene0042_01','scene0042_02','scene0062_00','scene0062_01','scene0062_02','scene0065_00','scene0065_01','scene0065_02','scene0067_00','scene0067_01','scene0067_02','scene0080_00','scene0080_01','scene0080_02','scene0083_00','scene0083_01','scene0084_00','scene0084_01','scene0084_02','scene0086_00','scene0086_01','scene0086_02','scene0090_00','scene0096_00','scene0096_01','scene0096_02','scene0100_00','scene0100_01','scene0100_02','scene0105_00','scene0105_01','scene0105_02','scene0112_00','scene0112_01','scene0112_02','scene0116_00','scene0116_01','scene0116_02','scene0119_00','scene0124_00','scene0124_01','scene0135_00','scene0146_00','scene0146_01','scene0146_02','scene0153_00','scene0153_01','scene0170_00','scene0170_01','scene0170_02','scene0175_00','scene0195_00','scene0195_01','scene0195_02','scene0214_00','scene0214_01','scene0214_02','scene0232_00','scene0232_01','scene0232_02','scene0237_00','scene0237_01','scene0250_00','scene0250_01','scene0250_02','scene0254_00','scene0254_01','scene0260_00','scene0260_01','scene0260_02','scene0265_0','scene0265_01','scene0265_02','scene0268_00','scene0268_01','scene0268_02','scene0346_00','scene0346_01','scene0350_00','scene0350_01','scene0350_02','scene0365_00','scene0365_01','scene0365_02','scene0366_00','scene0375_00','scene0375_01','scene0375_02','scene0396_00','scene0396_01','scene0396_02','scene0399_00','scene0399_01','scene0406_00','scene0406_01','scene0406_02','scene0410_00','scene0410_01','scene0413_00','scene0415_00','scene0415_01','scene0415_02','scene0429_00','scene0434_00','scene0434_01','scene0434_02','scene0441_00','scene0446_00','scene0446_01','scene0447_00','scene0447_01','scene0447_02','scene0448_00','scene0448_01','scene0448_02','scene0449_00','scene0449_01','scene0449_02','scene0458_00','scene0458_01','scene0460_00','scene0471_00','scene0471_01','scene0471_02','scene0484_00','scene0484_01','scene0485_00','scene0492_00','scene0492_01','scene0501_00','scene0501_01','scene0501_02','scene0509_00','scene0509_01','scene0509_02','scene0510_00','scene0510_01','scene0510_02','scene0519_00','scene0523_00','scene0523_01','scene0523_02','scene0527_00','scene0538_00','scene0539_00','scene0539_01','scene0539_02','scene0546_00','scene0550_00','scene0553_00','scene0553_01','scene0553_02','scene0564_00','scene0570_00','scene0570_01','scene0570_02','scene0571_00','scene0571_01','scene0574_00','scene0574_01','scene0574_02','scene0579_00','scene0579_01','scene0579_02','scene0581_00','scene0581_01','scene0581_02','scene0583_00','scene0583_01','scene0583_02','scene0602_00','scene0610_00','scene0610_01','scene0610_02','scene0612_00','scene0612_01','scene0625_00','scene0625_01','scene0631_00','scene0639_00','scene0659_00','scene0659_01','scene0662_00','scene0662_01','scene0662_02','scene0664_00','scene0664_01','scene0664_02','scene0676_00','scene0676_01','scene0683_00','scene0686_00','scene0686_01','scene0686_02','scene0693_00','scene0693_01','scene0693_02','scene0702_00','scene0702_01','scene0702_02'],
                   'Office': ['scene0010_00','scene0010_01','scene0017_01','scene0017_02','scene0025_00','scene0025_01','scene0025_02','scene0040_00','scene0040_01','scene0058_00','scene0058_01','scene0077_00','scene0077_01','scene0089_00','scene0089_01','scene0089_02','scene0093_00','scene0093_01','scene0093_02','scene0098_00','scene0098_01','scene0106_02','scene0108_00','scene0114_00','scene0114_01','scene0114_02','scene0126_00','scene0126_01','scene0126_02','scene0131_00','scene0131_01','scene0131_02','scene0138_00','scene0141_00','scene0141_01','scene0141_02','scene0142_00','scene0142_01','scene0166_00','scene0166_01','scene0166_02','scene0168_00','scene0168_01','scene0168_02','scene0174_00','scene0174_01','scene0177_00','scene0177_01','scene0177_02','scene0179_00','scene0186_00','scene0186_01','scene0206_00','scene0206_01','scene0206_02','scene0211_00','scene0211_01','scene0211_02','scene0211_03','scene0230_00','scene0238_00','scene0238_01','scene0241_00','scene0241_01','scene0241_02','scene0255_00','scene0255_01','scene0255_02','scene0256_00','scene0256_01','scene0262_00','scene0262_01','scene0263_00','scene0263_01','scene0264_00','scene0264_01','scene0264_02','scene0272_00','scene0272_01','scene0274_00','scene0274_01','scene0274_02','scene0305_00','scene0305_01','scene0309_00','scene0309_01','scene0333_00','scene0336_00','scene0336_01','scene0341_00','scene0341_01','scene0351_00','scene0351_01','scene0352_00','scene0352_01','scene0352_02','scene0369_00','scene0369_01','scene0369_02','scene0378_00','scene0378_01','scene0378_02','scene0379_00','scene0380_00','scene0380_01','scene0380_02','scene0383_00','scene0383_01','scene0383_02','scene0395_00','scene0395_01','scene0395_02','scene0424_00','scene0424_01','scene0424_02','scene0464_00','scene0469_00','scene0469_01','scene0469_02','scene0474_00','scene0474_01','scene0474_02','scene0474_03','scene0474_04','scene0474_05','scene0476_00','scene0476_01','scene0476_02','scene0505_00','scene0505_01','scene0505_02','scene0505_03','scene0505_04','scene0525_00','scene0525_01','scene0525_02','scene0535_00','scene0582_00','scene0582_01','scene0582_02','scene0591_00','scene0591_01','scene0591_02','scene0614_00','scene0614_01','scene0614_02','scene0619_00','scene0626_00','scene0626_01','scene0626_02','scene0628_00','scene0628_01','scene0628_02','scene0630_00','scene0630_01','scene0630_02','scene0630_03','scene0630_04','scene0630_05','scene0630_06','scene0643_00','scene0644_00','scene0653_00','scene0653_01','scene0663_00','scene0663_01','scene0663_02','scene0684_00','scene0684_01','scene0700_00','scene0700_01','scene0700_02'],
                   'Classroom': ['scene0015_00','scene0030_00','scene0030_01','scene0030_02','scene0041_00','scene0041_01','scene0095_00','scene0095_01','scene0167_00','scene0173_00','scene0173_01','scene0173_02','scene0185_00','scene0245_00','scene0257_00','scene0283_00','scene0285_00','scene0342_00','scene0411_00','scene0411_01','scene0411_02','scene0420_00','scene0420_01','scene0420_02','scene0466_00','scene0466_01','scene0483_00','scene0562_00','scene0566_00','scene0569_00','scene0569_01','scene0584_00','scene0584_01','scene0584_02','scene0621_00','scene0624_00','scene0657_00'],
                   'Gym': ['scene0016_00','scene0016_01','scene0016_02','scene0428_00,','scene0428_01'],
                   'CopyRoom': ['scene0018_00','scene0136_00','scene0136_01','scene0136_02','scene0303_00','scene0303_01','scene0303_02','scene0338_00','scene0338_01','scene0338_02','scene0462_00','scene0534_00','scene0534_01','scene0585_00','scene0585_01','scene0597_00','scene0597_00','scene0597_01','scene0597_02','scene0704_00','scene0104_00','scene0109_00','scene0109_01','scene0120_00','scene0120_01','scene0205_00','scene0205_01','scene0205_02','scene0240_00','scene0278_00','scene0278_01','scene0311_00','scene0478_00','scene0478_01','scene0482_00','scene0482_01','scene0511_00','scene0511_01','scene0544_00','scene0685_00','scene0685_01','scene0685_02','scene0704_01'],
                   'Bookstore': ['scene0020_00','scene0020_01','scene0037_00','scene0038_00','scene0038_01','scene0038_02','scene0064_00','scene0064_01','scene0079_00','scene0123_00','scene0123_01','scene0123_02','scene0154_00','scene0155_00','scene0155_01','scene0155_02','scene0187_00','scene0187_01','scene0189_00','scene0208_00','scene0218_00','scene0218_01','scene0258_00','scene0266_00','scene0266_01','scene0300_00','scene0300_01','scene0318_00','scene0320_00','scene0320_01','scene0320_02','scene0320_03','scene0361_00','scene0361_01','scene0361_02','scene0391_00','scene0408_00','scene0408_01','scene0453_00','scene0453_01','scene0481_00','scene0481_01','scene0489_00','scene0489_01','scene0489_02','scene0512_00','scene0558_00','scene0558_01','scene0558_02','scene0598_00','scene0598_01','scene0598_02','scene0661_00','scene0665_00','scene0665_01','scene0666_00','scene0666_01','scene0666_02','scene0667_00','scene0667_01','scene0667_02','scene0669_00','scene0669_01','scene0692_00','scene0692_01','scene0692_02','scene0692_03'],
                   'Stairs': ['scene0044_00','scene0044_01','scene0044_02','scene0069_00','scene0182_00','scene0182_01','scene0182_02','scene0357_00','scene0357_01','scene0524_00','scene0524_01','scene0561_00','scene0561_01','scene0692_04'],
                   'Mailboxes': ['scene0052_00','scene0052_01','scene0052_02','scene0190_00','scene0304_00','scene0596_00','scene0596_01','scene0596_02'],
                   'ComputerCluster': ['scene0056_00','scene0056_01','scene0106_00','scene0592_00','scene0592_01'],
                   'Storage': ['scene0057_00','scene0057_01','scene0307_00','scene0307_01','scene0307_02','scene0360_00','scene0377_00','scene0377_01','scene0377_02','scene0384_00','scene0516_00','scene0516_01','scene0605_00','scene0605_01','scene0635_00','scene0635_01','scene0696_00','scene0696_01','scene0696_02'],
                   'ConferenceRoom': ['scene0066_00','scene0075_00','scene0088_00','scene0088_01','scene0088_02','scene0088_03','scene0103_00','scene0103_01','scene0122_00','scene0122_01','scene0130_00','scene0140_00','scene0140_01','scene0156_00','scene0163_00','scene0163_01','scene0169_00','scene0169_01','scene0171_00','scene0171_01','scene0180_00','scene0188_00','scene0204_00','scene0204_01','scene0204_02','scene0215_00','scene0215_01','scene0225_00','scene0227_00','scene0249_00','scene0252_00','scene0289_00','scene0289_01','scene0291_00','scene0291_01','scene0291_02','scene0302_00','scene0302_01','scene0345_00','scene0345_01','scene0354_00','scene0358_00','scene0358_01','scene0358_02','scene0372_00','scene0373_00','scene0373_01','scene0388_00','scene0388_01','scene0393_00','scene0393_01','scene0393_02','scene0401_00','scene0402_00','scene0403_00','scene0403_01','scene0414_00','scene0422_00','scene0427_00','scene0430_00','scene0430_01','scene0452_00','scene0452_01','scene0452_02','scene0455_00','scene0456_00','scene0456_01','scene0463_00','scene0463_01','scene0470_00','scene0470_01','scene0490_00','scene0493_00','scene0493_01','scene0494_00','scene0495_00','scene0496_00','scene0500_00','scene0500_01','scene0508_00','scene0508_01','scene0508_02','scene0528_00','scene0528_01','scene0551_00','scene0563_00','scene0575_00','scene0575_01','scene0575_02','scene0578_00','scene0578_01','scene0578_02','scene0599_00','scene0599_01','scene0599_02','scene0611_00','scene0611_01','scene0627_00','scene0627_01','scene0634_00','scene0655_00','scene0655_01','scene0655_02','scene0680_00','scene0680_01','scene0689_00','scene0691_00','scene0691_01','scene0705_00','scene0705_01','scene0705_02'],
                   'Closet': ['scene0074_00','scene0074_01','scene0074_02','scene0121_00','scene0121_01','scene0121_02','scene0404_00','scene0454_00'],
                   'DiningRoom': ['scene0102_00','scene0102_01','scene0451_00','scene0451_01','scene0451_02','scene0451_03','scene0451_04','scene0451_05','scene0555_00','scene0606_00','scene0606_01','scene0606_02','scene0671_00','scene0671_01'],
                   'GameRoom': ['scene0115_00','scene0115_01','scene0115_02','scene0176_00','scene0216_00','scene0502_00','scene0502_01','scene0502_02','scene0533_00','scene0533_01','scene0618_00'],
                   'LaundryRoom': ['scene0139_00','scene0417_00','scene0595_00','scene0601_00','scene0601_01','scene0650_00','scene0678_00','scene0678_01','scene0678_02','scene0261_00','scene0261_01','scene0261_02','scene0261_03','scene0349_00','scene0349_01','scene0404_01','scene0404_02','scene0421_00','scene0421_01','scene0421_02']
                   }
    
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        # Check if the scene_name matches any predefined category
        for category, scenes in all_scene_names.items():
            if scene_name in scenes:
                out_filename = f"{category}_{scene_name}.txt"
                break
        try:
            # out_filename = scene_name+'.npy' # scene0000_00.npy
            # collect_point_label(scene_path, scene_name, os.path.join(SAVE_PATH, category , out_filename))
            collect_point_label(scene_path, scene_name, os.path.join(SAVE_PATH, out_filename))

        except:
            raise ValueError('ERROR {}!!'.format(scene_path))
