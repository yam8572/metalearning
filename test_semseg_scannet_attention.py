"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene_scannet
# from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def get_class2Label(class_txt_path):
	# class_txt_path='datasets/class_names.txt'
	with open(class_txt_path,'r') as file:
		line = file.read()
	class2Label_dict={index: value for index, value in enumerate(line.splitlines())}
	return class2Label_dict

# class2Label_dict = get_class2Label('datasets/class_names_9_s3dis_scannet.txt')
# class2Label_dict = get_class2Label('datasets/class_names_9_s3dis_scannet_noclutter.txt')
# class2Label_dict = get_class2Label('datasets/ScanNet/meta/scannet_classnames_13_noclutter.txt')
class2Label_dict = get_class2Label('datasets/ScanNet/meta/scannet_classnames_yun.txt')
print("class2Label_dict = ",class2Label_dict)

# g_classes = ['floor','wall','window','door','table','chair','sofa','bookcase','clutter']
# g_class2label = {cls: i for i,cls in enumerate(g_classes)}
# g_class2color = {'floor':	[0,0,255],
#                  'wall':	[0,255,255],
#                  'window':      [100,100,255],
#                  'door':        [200,200,100],
#                  'table':       [170,120,200],
#                  'chair':       [255,0,0],
#                  'sofa':        [200,100,100],
#                  'bookcase':    [10,200,100],
#                  'clutter':     [50,50,50]} 
# g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

# g_classes = ['floor','wall','window','door','table','chair','sofa','bookcase']
# g_class2label = {cls: i for i,cls in enumerate(g_classes)}
# g_class2color = {'floor':	[0,0,255],
#                  'wall':	[0,255,255],
#                  'window':      [100,100,255],
#                  'door':        [200,200,100],
#                  'table':       [170,120,200],
#                  'chair':       [255,0,0],
#                  'sofa':        [200,100,100],
#                  'bookcase':    [10,200,100],
#                  } 
# g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

# g_classes = ['desk','bed','sink','bathtub','toilet','curtain','counter','shower curtain','refridgerator','picture','cabinet','otherfurniture']
# g_class2label = {cls: i for i,cls in enumerate(g_classes)}
# g_class2color = {
#         'desk': [247, 182, 210],
#         'bed': [255, 187, 120],
#         'sink': [112, 128, 144],
#         'bathtub': [227, 119, 194],
#         'toilet': [44, 160, 44],
#         'curtain': [219, 219, 141],
#         'counter': [23, 190, 207],
#         'shower curtain': [158, 218, 229],
#         'refridgerator': [255, 127, 14],
#         'picture': [196, 156, 148],
#         'cabinet': [31, 119, 180],
#         'otherfurniture': [82, 84, 163],
#     }
# g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

g_classes = ['bed','floor','wall','sink','bathtub','window','door','table','chair','sofa','bookshelf','desk','clutter','toilet','curtain','counter','shower curtain','refridgerator','picture','cabinet','otherfurniture']
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {
        'bed': [255, 187, 120],
        'floor':	[0,0,255],
        'wall':	[0,255,255],
        'sink': [112, 128, 144],
        'bathtub': [227, 119, 194],
        'window':      [100,100,255],
        'door':        [200,200,100],
        'table':       [170,120,200],
        'chair':       [255,0,0],
        'sofa':        [200,100,100],
        'bookshelf':    [10,200,100],
        'desk': [247, 182, 210],
        'clutter':     [50,50,50],
        'toilet': [44, 160, 44],
        'curtain': [219, 219, 141],
        'counter': [23, 190, 207],
        'shower curtain': [158, 218, 229],
        'refridgerator': [255, 127, 14],
        'picture': [196, 156, 148],
        'cabinet': [31, 119, 180],
        'otherfurniture': [82, 84, 163],
    }
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    # parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--type', type=str, default='ECANet', help='attention type name [default: scSE][scSE,BAM,CBAM,ECANet,SENet,EMCA]')
    
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # experiment_dir = '/media/g111056119/SP PHD U3/datasets/log/sem_seg/' + args.log_dir
    experiment_dir = '/media/g111056119/SP PHD U3/datasets/log/pointnet/scannet/' + args.log_dir
    visual_dir = experiment_dir + '/visual_test/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval_test_lastModel.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = len(class2Label_dict)
    print("NUM_CLASSES = ",NUM_CLASSES)
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # root = '/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_9_noclutter/'
    # root = '/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_13_noclutter/'
    root = '/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_21/'
    # root = '/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_21_val_test/'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene_scannet(root, split='train', block_points=NUM_POINT, num_class=NUM_CLASSES)
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene_scannet(root, split='test', block_points=NUM_POINT, num_class=NUM_CLASSES)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model_name = 'pointnet2_sem_seg_msg_attention'
    print("Model Name: ", model_name)
    
    from models.pointnet2_sem_seg_msg_attention import get_model                        
    classifier = get_model(NUM_CLASSES, args.type).cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/last_model.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                # fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            # filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            # with open(filename, 'w') as pl_save:
            #     for i in pred_label:
            #         pl_save.write(str(int(i)) + '\n')
            #     pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    # fout_gt.write(
                    #     'v %f %f %f %d %d %d\n' % (
                    #         whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                    #         color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                # fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                class2Label_dict[l] + ' ' * (14 - len(class2Label_dict[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]) + 1e-6)
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / (float(np.sum(total_seen_class) + 1e-6))))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
