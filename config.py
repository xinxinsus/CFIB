import torch

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else'cpu')#
#DEVICE = torch.device('cpu')


# import random
#
# # 针对每组实验设置不同的种子
# num_experiments = 3
#
# seeds = [random.randint(1, 100000) for _ in range(num_experiments)]
#
# for seed in seeds:
#     random.seed(seed)



#[8308, 5993, 7912, 3478, 1544, 752]

# TORCH_SEED = 128 #129

#for eng
# TORCH_SEED = 3478 #129   128 130

#for split 20
TORCH_SEED = 129 #130 #125

class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = '/home/ubuntu/models/roberta-large' #/home/ubuntu/models '/media/tiffany/新加卷/PycharmProjects/models/roberta-large'

        # hyper parameter
        self.epochs = 20

        self.batch_size = 2
        self.lr = 1e-5
        self.tuning_bert_rate = 3.6284e-06 # 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1
        self.TORCH_SEED=TORCH_SEED
        # gnn
        self.feat_dim = 1024
        self.att_heads = '4'

