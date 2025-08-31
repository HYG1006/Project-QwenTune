# coding: utf-8
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import os
import os.path as osp

import optuna
from loguru import logger

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--use_pe', action="store_true")
parser.add_argument('--cuda', action='store_true', help='use CUDA device', default=True)
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--model', type=str)

args = parser.parse_args()
logger.info(f'CURRENT_MODEL:{args.model}')
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    # torch.cuda.set_device(args.gpu_id)
    # device = torch.device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

def evaluate(model):
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model.eval()
    idx = 0
    avg_loss = 0
    # print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            decode, _ = model(data)

            # Calculate cross-entropy loss
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    # print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss.item() / idx), avg_loss / idx

criterion = nn.CrossEntropyLoss()


# Train Function
def train(model, optimizer):
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model.train()
    idx = 0
    avg_loss = 0
    hidden = None
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        # data: (seq_len, bs)
        target = target.to(device)
        decode, _ = model(data)

        # Calculate cross-entropy loss
        optimizer.zero_grad()
        loss = criterion(decode.view(decode.size(0)*decode.size(1), -1), target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        # if (idx+1) % 50 == 0:
            # print(f"The loss is {loss}")
        idx += 1
        avg_loss += loss
    return math.exp(avg_loss.item() / idx), avg_loss / idx

# def objective(trial):
#     hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
#     num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
#     dropout = trial.suggest_float('dropout', 0.0, 0.6)
#     lr = trial.suggest_float('lr', 1e-3, 4e-3, log=True)
#     if args.model == 'lstm':
#         model_ = model.LMModel_LSTM(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = num_layers, hidden_size=hidden_dim, dropout=dropout)
#     elif args.model == 'rnn':
#         model_ = model.LMModel_RNN(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = num_layers, hidden_size=hidden_dim, dropout=dropout)
#     elif args.model == 'transformer':
#         model_ = model.LMModel_transformer(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = num_layers, hidden_size=hidden_dim, dropout=dropout)
#     model_ = model_.to(device)
#     optimizer = optim.Adam(model_.parameters(), lr=lr)
#     train_perplexity = []
#     valid_perplexity = []
#     for epoch in range(1, args.epochs+1):
#         # print(f"Start training epoch {epoch}")
#         train_ppl, train_loss = train(model_, optimizer)
#         train_perplexity.append(train_ppl)
        
#         valid_ppl, valid_loss = evaluate(model_)
#         valid_perplexity.append(valid_ppl)
#     return min(valid_perplexity)


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)
# for k, v in study.best_params.items():
#     logger.info(f'{k}: {v}')
# exit()

########################################
# Build LMModel model (bulid your language model here)
# model = model.LMModel_transformer(nvoc = len(data_loader.vocabulary), num_layers = args.num_layers,
#                       dim = args.emb_dim, nhead = args.num_heads)
if args.model == 'lstm':
    model_ = model.LMModel_LSTM(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = 2, hidden_size=256, dropout=0.4538)
    optimizer = optim.Adam(model_.parameters(), lr=0.001878)
elif args.model == 'rnn':
    model_ = model.LMModel_RNN(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = 2, hidden_size=256, dropout=0.5756)
    optimizer = optim.Adam(model_.parameters(), lr=0.002007)
elif args.model == 'transformer':
    model_ = model.LMModel_transformer(nvoc = len(data_loader.vocabulary), dim = args.emb_dim, num_layers = 4, hidden_size=512, dropout=0.2008)
    optimizer = optim.Adam(model_.parameters(), lr=0.001331)
model_ = model_.to(device)

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.




# Loop over epochs.
train_perplexity = []
valid_perplexity = []
train_loss_list = []
valid_loss_list = []
for epoch in range(1, args.epochs+1):
    print(f"Start training epoch {epoch}")
    train_ppl, train_loss = train(model_, optimizer)
    train_perplexity.append(train_ppl)
    train_loss_list.append(train_loss.detach().cpu())
    
    valid_ppl, valid_loss = evaluate(model_)
    valid_perplexity.append(valid_ppl)
    valid_loss_list.append(valid_loss.detach().cpu())

print(f"Train Perpelexity {train_perplexity}")
print(f"Valid Perpelexity {valid_perplexity}")
print(min(valid_perplexity))

# save ckpt
torch.save(model_.state_dict(), f'ckpts/{args.model}_{args.epochs}.pth')

# plot loss vs. epoch
import matplotlib.pyplot as plt

t = range(1, args.epochs+1, 1)
plt.plot(t, train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid()
plt.title(f'Model: {args.model}')
plt.savefig(f'images/train_loss_{args.model}_epoch{args.epochs}.png')

plt.cla()
plt.plot(t, valid_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.grid()
plt.title(f'Model: {args.model}')
plt.savefig(f'images/valid_loss_{args.model}_epoch{args.epochs}.png')






## transformer
##Train Perpelexity [1093.058536984991, 446.73140622980503, 301.52450338320034, 246.26376606407698, 214.64299704889373, 192.99575344050672, 176.80442279813906, 164.02186558697892, 153.5047097003694, 144.73608891921225, 137.18168103884878, 130.70923073012537, 125.04380849718291, 120.81010959545705, 116.09009338903756, 112.07989914997275, 108.67646540748619, 105.87390635984161, 103.12351166847573, 100.41740827910725, 97.94114063183589, 95.58049159659934, 93.29935789172667, 91.41218092537845, 89.69534270973497, 88.18531717000073, 86.49113246722683, 84.92576714405752, 83.6108162123822, 82.21022757201196, 80.88834905697544, 79.76059550830294, 78.5046538796179, 77.26560679667993, 76.4028940453166, 75.52448865056043, 74.65149921930217, 73.67881096874149, 72.87680871150808, 72.2834156039246, 71.63370870931435, 70.77280808260397, 69.972951334495, 69.50273251233686, 68.6820835612053, 68.07348860281, 67.77206381346691, 67.04683158040281, 66.74084325395272, 66.1217261463782]
##Valid Perpelexity [471.95198099109376, 276.04429093731466, 228.5046823159407, 203.25022718952528, 186.0105467837613, 177.37231430047913, 168.32827191112543, 161.71853588832985, 152.76365250101256, 149.43264358476756, 143.01507228298814, 140.57097175146382, 137.15968731269845, 134.8223400202423, 133.22031820365928, 130.0433174395789, 128.15806692823546, 125.83682217247679, 124.80598709377034, 123.94843538679024, 121.66099759450307, 119.74962627493214, 119.30968139696714, 118.62718993775586, 118.11481103324657, 118.18190097707756, 116.9078223851364, 117.64207940713467, 115.74835977158367, 115.02792821422683, 113.80044013854625, 113.39551352259511, 112.44257275949286, 112.80437762097799, 111.97069616713435, 112.30992651029509, 111.35269706518743, 111.89841177281846, 110.78500880231782, 109.53848881656094, 109.39692712592273, 109.8063033234072, 108.88079571164664, 108.8392689053919, 109.2428671941576, 108.0125805400334, 108.73593596254595, 107.49114701870928, 106.63633736437735, 108.25843295038214]

## rnn
##Train Perpelexity [995.3672611273553, 434.70922590450584, 307.9347987843869, 255.6003595669633, 226.7636382089111, 207.35282141559208, 194.13007197120803, 183.73432053612598, 174.8992338702511, 167.69495861400665, 161.4085290527775, 155.74372056637998, 151.55545369227883, 148.22657238201785, 144.71786996201607, 141.50342384488843, 138.8143850996657, 137.04725715069975, 134.24047374601173, 131.83823654820347, 130.31834728701398, 128.77612863367, 126.64021731340242, 125.00136232258193, 123.10241849085276, 121.66673636163405, 120.43964011805079, 118.89392710641064, 118.13731318099777, 117.51230722919725, 116.79106543954097, 116.16308792989959, 114.7975819291587, 113.96076868545357, 112.91864640413263, 112.20880855945985, 111.10073927934316, 110.48935475746691, 110.17051855183189, 109.56546044142526, 108.72046456343934, 108.14523265088539, 107.78344453181698, 107.03137140069325, 108.22804763576138, 106.84666528860784, 106.28116731753234, 105.74989908074868, 105.063837579403, 105.08004842669699]
##[435.2040181268229, 289.56558792293094, 236.29404703928913, 211.51063791937943, 196.146426552119, 185.57689518893113, 177.53758359948736, 171.50098592370475, 167.81115109065695, 161.63897442976267, 158.67377893084134, 154.37119300575847, 153.60972136030108, 151.35379022724297, 149.02886864096385, 147.13194041651582, 147.40255777852164, 146.93979304808678, 144.49167813153036, 145.40821289002955, 146.15419466036315, 143.73420670382617, 142.28961017097777, 141.4828361059593, 139.86941681642338, 139.30361922887678, 138.7214366358244, 138.23280746639008, 137.96308500089495, 138.3178822232333, 139.9596937131379, 138.62274183945317, 138.836997238954, 138.86476729424632, 137.9829349822438, 136.9222751758489, 137.1810663626515, 137.20857997281368, 137.50117816270438, 137.0734574328815, 135.4130253601956, 135.43929865901725, 134.1035725828687, 134.15663962985872, 134.83446350214297, 135.7127397977331, 135.30075708854062, 135.1008670954774, 135.49909678980202, 135.38373221478753]

## lstm
##Train Perpelexity [987.6074376651504, 526.1384487730347, 381.02554927472613, 315.6692678040235, 275.7551112480772, 247.9056881018885, 227.57813391722826, 211.0375218450999, 197.87254030317513, 187.09857621707351, 177.87652204068183, 169.99360813390612, 162.95624018252315, 156.651324279555, 150.91763572054174, 146.23471412304139, 141.87534914239805, 137.9210635933939, 134.21805759706234, 130.6435960888447, 127.53294006114048, 124.93340427542661, 121.93932603936824, 119.79702047691012, 117.67743773784014, 115.46001220854548, 113.62411289602328, 111.70002408060478, 109.92335225071842, 108.0000240037901, 106.44513262745603, 104.45950375108245, 103.050935835863, 101.63305554260339, 100.2272492994706, 99.20061688357356, 98.02357834720246, 96.88111736796658, 95.661104523539, 94.55808078021877, 93.51956900346184, 92.52543893601911, 91.47423357548674, 90.79665878299264, 89.93353605972138, 89.23642878993036, 88.39289580705763, 87.67465074503238, 86.9714203286989, 86.22751172833857]
##Valid Perpelexity [614.0582059877147, 400.2866680385491, 318.23466314585863, 276.6622048121467, 248.54567584663116, 229.56935464961305, 214.22263798270998, 202.13999580513743, 193.39127479362526, 186.4511048837397, 178.27062183011552, 172.26494726212266, 167.75820999763224, 163.51491504310383, 159.98390912175557, 155.8878082327453, 152.89949293260239, 150.36904702384422, 148.24171436866231, 146.2168908559041, 143.92098578799693, 141.84065571393765, 139.95328701344587, 138.16577921982977, 136.98145902077968, 135.36617412304676, 134.42368186351027, 133.3891458904422, 132.56996899109802, 131.4618381148071, 130.2872609529188, 129.36920276642115, 128.62664398627274, 127.65244324042912, 127.30550763557245, 126.60894275042597, 126.45386490249038, 126.31652922975658, 125.70282821441455, 124.99725246318152, 124.43443739146217, 124.06629254407606, 123.43251092817852, 123.0074546245982, 122.74924390267881, 123.20992891790628, 122.14282296211354, 121.89287041405467, 121.89147546883822, 121.62314619413253]