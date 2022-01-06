import argparse
import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

from tqdm import trange
import copy

from pFedGP.Learner import pFedGPIPComputeLearner, pFedGPIPDataLearner

from experiments.backbone import CNNTarget
from experiments.heterogeneous_class_dist.clients import BaseClients
from utils import get_device, set_logger, set_seed, str2bool, detach_to_numpy, save_experiment, \
                  print_calibration, calibration_search, offset_client_classes, calc_metrics
from experiments.calibrate import ECELoss

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

#############################
#       Dataset Args        #
#############################
parser.add_argument(
    "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'cinic10'],
)
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--num-clients", type=int, default=50, help="number of simulated clients")

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
parser.add_argument("--num-client-agg", type=int, default=5, help="number of kernels")

################################
#       Model Prop args        #
################################
parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

parser.add_argument('--method', type=str, default='pFedGP-compute',
                    choices=['pFedGP-data', 'pFedGP-compute'],
                    help='Inducing points method')
parser.add_argument('--embed-dim', type=int, default=84)
parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                    help='kernel function')
parser.add_argument('--objective', type=str, default='predictive_likelihood',
                    choices=['predictive_likelihood', 'marginal_likelihood'])
parser.add_argument('--predict-ratio', type=float, default=0.5,
                    help='ratio of samples to allocate for test part when optimizing the predictive_likelihood')
parser.add_argument('--num-inducing-points', type=int, default=100, help='number of inducing points per class')
parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
parser.add_argument('--outputscale', type=float, default=8., help='output scale')
parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--outputscale-increase', type=str, default='constant',
                    choices=['constant', 'increase', 'decrease'],
                    help='output scale increase/decrease/constant along tree')
parser.add_argument('--balance-classes', type=str2bool, default=True, help='Balance classes dist. per client in PredIP')

#############################
#       General args        #
#############################
parser.add_argument('--color', type=str, default='darkblue', help='color for calibration plot')
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument("--eval-every", type=int, default=25, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output/pFedGP-IP", help="dir path for output file")  # change
parser.add_argument("--seed", type=int, default=42, help="seed value")

args = parser.parse_args()

set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
num_classes = 10 if args.data_name in ('cifar10', 'cinic10') else 100
classes_per_client = 2 if args.data_name == 'cifar10' else 10 if args.data_name == 'cifar100' else 4

exp_name = f'pFedGP-IP_{args.data_name}_method_{args.method}_num_clients_{args.num_clients}_seed_{args.seed}_' \
           f'num_steps_{args.num_steps}_inner_steps_{args.inner_steps}_num_inducing_{args.num_inducing_points}' \
           f'_objective_{args.objective}_predict_ratio_{args.predict_ratio}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

# change
logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

ECE_module = ECELoss()

@torch.no_grad()
def eval_model(global_model, GPs, X_bar, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    targets = []
    preds = []

    global_model.eval()

    for client_id in range(args.num_clients):
        is_first_iter = True
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        GPs[client_id], label_map, X_train, Y_train = build_tree(clients, client_id, X_bar)
        GPs[client_id].eval()
        client_X_bar = X_bar[list(label_map.keys()), ...]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
                                         device=label.device)
            X_test = global_model(img)
            loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, client_X_bar, is_first_iter)

            running_loss += loss.item()
            running_correct += pred.argmax(1).eq(Y_test).sum().item()
            running_samples += len(Y_test)

            is_first_iter = False
            targets.append(Y_test)
            preds.append(pred)

        # erase tree (no need to save it)
        GPs[client_id].tree = None

        results[client_id]['loss'] = running_loss / (batch_count + 1)
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples

        # if client_id > 3:
        #     break

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return results, labels_vs_preds


###############################
# init net and GP #
###############################
clients = BaseClients(args.data_name, args.data_path, args.num_clients,
                    classes_per_client=classes_per_client,
                    batch_size=args.batch_size)

# NN
net = CNNTarget(n_kernels=args.n_kernels, embedding_dim=args.embed_dim)
net = net.to(device)

# GPs
ip_method = pFedGPIPDataLearner if args.method == 'pFedGP-data'\
            else pFedGPIPComputeLearner

GPs = torch.nn.ModuleList([])
for client_id in range(args.num_clients):
    # GP instance
    GPs.append(ip_method(args, classes_per_client))

# Inducing locations
X_bar = nn.Parameter(torch.randn((num_classes, args.num_inducing_points, args.embed_dim), device=device) * 0.01,
                     requires_grad=True)

def get_optimizer(network, curr_X_bar):
    params = [
        {'params': curr_X_bar},
        {'params': network.parameters()}
    ]
    return torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9) \
           if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def build_tree(clients, client_id, curr_X_bar):
    """
    Build GP tree per client
    :return: List of GPs
    """
    for k, batch in enumerate(clients.train_loaders[client_id]):
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)
    client_X_bar = curr_X_bar[list(label_map.keys()), ...]

    GPs[client_id].build_base_tree(X, offset_labels, client_X_bar)  # build tree
    return GPs[client_id], label_map, X, offset_labels


criteria = torch.nn.CrossEntropyLoss()
results = defaultdict(list)

################
# init metrics #
################
last_eval = -1
best_step = -1
best_acc = -1
test_best_based_on_step, test_best_min_based_on_step = -1, -1
test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
step_iter = trange(args.num_steps)

best_model = copy.deepcopy(net)
best_X_bar = copy.deepcopy(X_bar)
best_labels_vs_preds_val = None
best_val_loss = -1

for step in step_iter:

    # print tree stats every 100 epochs
    to_print = True if step % 100 == 0 else False

    # select several clients
    client_ids = np.random.choice(range(args.num_clients), size=args.num_client_agg, replace=False)

    # initialize global model params
    params = OrderedDict()
    for n, p in net.named_parameters():
        params[n] = torch.zeros_like(p.data)

    # initialize inducing points
    X_bar_step = torch.zeros_like(X_bar.data, device=device)

    # iterate over each client
    train_avg_loss = 0
    num_samples = 0

    for j, client_id in enumerate(client_ids):

        curr_global_net = copy.deepcopy(net)
        curr_global_net.train()

        curr_X_bar = copy.deepcopy(X_bar)
        optimizer = get_optimizer(curr_global_net, curr_X_bar)

        # build tree at each step
        GPs[client_id], label_map, _, __ = build_tree(clients, client_id, curr_X_bar)
        GPs[client_id].train()

        for i in range(args.inner_steps):

            # init optimizers
            optimizer.zero_grad()

            for k, batch in enumerate(clients.train_loaders[client_id]):
                batch = (t.to(device) for t in batch)
                img, label = batch

                z = curr_global_net(img)
                X = torch.cat((X, z), dim=0) if k > 0 else z
                Y = torch.cat((Y, label), dim=0) if k > 0 else label

            offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                         device=Y.device)
            client_X_bar = curr_X_bar[list(label_map.keys()), ...]

            loss = GPs[client_id](X, offset_labels, client_X_bar, to_print=to_print)
            loss *= args.loss_scaler

            # propagate loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
            optimizer.step()

            train_avg_loss += loss.item() * offset_labels.shape[0]
            num_samples += offset_labels.shape[0]

            step_iter.set_description(
                f"Step: {step+1}, client: {client_id}, Inner Step: {i}, Loss: {loss.item()}"
            )

        for n, p in curr_global_net.named_parameters():
            params[n] += p.data
        X_bar_step += curr_X_bar.data
        # erase tree (no need to save it)
        GPs[client_id].tree = None

    train_avg_loss /= num_samples

    # average parameters
    for n, p in params.items():
        params[n] = p / args.num_client_agg
    # update new parameters
    net.load_state_dict(params)
    X_bar.data = X_bar_step.data / args.num_client_agg

    if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
        val_results, labels_vs_preds_val = eval_model(net, GPs, X_bar, clients, split="val")
        val_avg_loss, val_avg_acc = calc_metrics(val_results)
        logging.info(f"Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)

        if best_acc < val_avg_acc:
            best_val_loss = val_avg_loss
            best_acc = val_avg_acc
            best_step = step

            best_model = copy.deepcopy(net)
            best_X_bar = copy.deepcopy(X_bar)
            best_labels_vs_preds_val = labels_vs_preds_val

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)

net = best_model
X_bar = best_X_bar

test_results, labels_vs_preds_test = eval_model(net, GPs, X_bar, clients, split="test")
avg_test_loss, avg_test_acc = calc_metrics(test_results)

logging.info(f"\nStep: {step + 1}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_acc:.4f}")
logging.info(f"\nStep: {step + 1}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

best_temp = calibration_search(ECE_module, out_dir, best_labels_vs_preds_val, args.color, 'calibration_val.png')
logging.info(f"best calibration temp: {best_temp}")
print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_temp1.png', args.color, temp=1.0)
print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_best.png', args.color, temp=best_temp)

results['best_step'].append(best_step)
results['best_val_acc'].append(best_acc)
results['test_loss'].append(avg_test_loss)
results['test_acc'].append(avg_test_acc)

save_path = Path(args.save_path)
save_path.mkdir(parents=True, exist_ok=True)
with open(str(save_path / f"results_{args.inner_steps}_inner_steps_seed_{args.seed}.json"), "w") as file:
    json.dump(results, file, indent=4)