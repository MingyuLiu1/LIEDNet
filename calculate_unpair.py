import os
import csv
import numpy as np
import torch
import pyiqa
import argparse
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def load_test_img_batch(img_dir, ref_dir, all_metrics):

    img_list = [x for x in sorted(os.listdir(img_dir))]
    ref_list = [x for x in sorted(os.listdir(ref_dir))]
    all_metrics['input_path'] = img_list
    all_metrics['gt_path'] = ref_list
    img_batch = []
    ref_batch = []
    for img_name, ref_name in zip(img_list, ref_list):
        img_path = os.path.join(img_dir, img_name)
        ref_path = os.path.join(ref_dir, ref_name)

        img_tensor = imread2tensor(img_path).unsqueeze(0)
        ref_tensor = imread2tensor(ref_path).unsqueeze(0)
        img_batch.append(img_tensor)
        ref_batch.append(ref_tensor)

    return img_batch, ref_batch, all_metrics


def dict2csv(dic, filename):
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()


def run_test(img_dir, ref_dir, test_metric_names, use_cpu):
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'============> Testing on {device}')
    all_metrics = dict()
    img_batch, ref_batch, all_metrics = load_test_img_batch(img_dir, ref_dir, all_metrics)

    for metric_name in test_metric_names:
        if metric_name == 'fid':
            continue
        print(f'============> Testing {metric_name} ... ')
        iqa_metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)
        for idx, img in enumerate(img_batch):
            img_batch[idx] = img.to(device)
            ref_batch[idx] = ref_batch[idx].to(device)
            img_batch[idx].requires_grad_()

        metric_mode = DEFAULT_CONFIGS[metric_name]['metric_mode']
        if metric_mode == 'FR':
            score = []
            for i in range(len(img_batch)):
                b,c,h,w = img_batch[i].shape
                score.append(iqa_metric(img_batch[i][:,:,:h,:w], ref_batch[i][:,:,:h,:w]).squeeze().data.cpu().numpy())
        else:
            score = []
            for i in range(len(img_batch)):
                score.append(iqa_metric(img_batch[i]).squeeze().data.cpu().numpy())
        our_score = np.mean(score)
        our_score_std = np.std(score)
        print(f'============> {metric_name} Results Avg score is {our_score}')
        print(f'============> {metric_name} Results Std score is {our_score_std}')
        all_metrics[metric_name] = score

    if metric_name == 'fid':
        fid_metric = pyiqa.create_metric('fid', device='cuda')
        FID = fid_metric(img_dir, ref_dir)
        print(f'============> {metric_name} Results score is {FID}')

    dict2csv(all_metrics, img_dir+'/Metrics_result.csv')


if __name__ == '__main__':
    import sys
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric_names', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu for test')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--gt_path', type=str)
    args = parser.parse_args()
    
    result_path = args.result_path
    gt_path = args.gt_path 

    if args.metric_names is not None:
        test_metric_names = args.metric_names
    else:
        test_metric_names = pyiqa.list_models()

    run_test(result_path, gt_path, test_metric_names, args.use_cpu)
    