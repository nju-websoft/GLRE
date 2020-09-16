
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch', type=int, default=16)
inp = parser.parse_args()


output_path = './results/cdr-dev/cdr_basebert/'
config_path = './configs/cdr_basebert.yaml'
os.system('CUDA_VISIBLE_DEVICES=' + inp.gpu + ' python ./src/main.py --train --batch=' + str(inp.batch)+ ' --test_data=./data/CDR/processed/dev_filter.data'
          ' --config_file=' + config_path + ' --save_pred=dev --output_path=' + output_path)

input_theta = 0.5
with open(os.path.join(output_path, "train_finsh.ok"), 'r') as f:
    for line in f.readlines():
        input_theta = line.strip().split("\t")[1]
        break

os.system('CUDA_VISIBLE_DEVICES=' + inp.gpu+ ' python ./src/main.py --test --batch ' + str(inp.batch)+ ' --test_data ./data/CDR/processed/dev_filter.data'
          ' --config_file=' +  config_path + ' --output_path=' + output_path
          + ' --save_pred=dev_test --input_theta='+str(input_theta) + ' --remodelfile='+output_path)

os.system('CUDA_VISIBLE_DEVICES=' + inp.gpu+ ' python ./src/main.py --test --batch ' + str(inp.batch) + ' --test_data ./data/CDR/processed/test_filter.data'
          ' --config_file=' + config_path + ' --output_path=' + output_path
          + ' --save_pred=test --input_theta=' + str(input_theta) + ' --remodelfile=' + output_path)