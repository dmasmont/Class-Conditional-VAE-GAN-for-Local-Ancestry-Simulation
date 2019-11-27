import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from VCFDataset import VCFDataset
from cvaegan.CVAELAI import CVAELAI

import torch.optim as optim

IS_GAN = True

def KLD_loss(mu, logvar):
    n = mu.shape[1]
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * n
    return KLD

GPU = 0
device = torch.device('cuda:{}'.format(GPU))

_CHROMOSOME = 20
_GENERATIONS_TRAIN = [0]
_GENERATIONS_TEST =  [0]


IS_REAL_DATA = False
if IS_REAL_DATA:
    DATASET_ROOT = '/dataset-real/admixed-simulation-output/'
else:
    DATASET_ROOT = '/dataset-simulated/admixed-simulation-output/'


train_dataset_list = []
for gen in _GENERATIONS_TRAIN:
    for ancestry in ['AFR', 'EUR', 'EAS']:
        vcf_path_train = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen),
                                      '{}-train'.format(ancestry)) + '/test-admix.query.vcf'
        map_path_train = None #os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen), '{}-train'.format(ancestry)) + '/test-admix.result'
        out_path_train = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen), '{}-train'.format(ancestry))
        train_dataset_ = VCFDataset(vcf_path_train, map_path_train, out_path_train, single_ancestry=True, ancestry=ancestry)
        train_dataset_list.append(train_dataset_)


val_dataset_list = []
for gen in _GENERATIONS_TEST:
    for ancestry in ['AFR', 'EUR', 'EAS']:
        vcf_path_val = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen),
                                    '{}-val'.format(ancestry)) + '/test-admix.query.vcf'
        map_path_val = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen),
                                    '{}-val'.format(ancestry)) + '/test-admix.result'
        out_path_val = os.path.join(DATASET_ROOT, 'chm{}'.format(_CHROMOSOME), 'gen_{}'.format(gen), '{}-val'.format(ancestry))
        val_dataset_ = VCFDataset(vcf_path_val, map_path_val, out_path_val, single_ancestry=True, ancestry=ancestry)  # , is_missing_data=False, missing_percent=MISSING_PERCENTAGE, balance_dataset=False)
        val_dataset_list.append(val_dataset_)


WINDOWS_SIZE_LIST = [1000]
WINDOWS_SIZE = WINDOWS_SIZE_LIST[0]
IS_MISSING_LABELS = True
IS_NOISY_LABELS = False
IS_RESIDUAL = True
MISSING_PERCENTAGE = 0.1
NOISY_PERCENTAGE = 0.1
SIMULATION_LOSS = False


_train_dataset_list = []
for dataset in train_dataset_list:
    dataset.windows_size = WINDOWS_SIZE
    _train_dataset_list.append(dataset)
train_dataset = torch.utils.data.ConcatDataset(_train_dataset_list)

_val_dataset_list = []
for dataset in val_dataset_list:
    dataset.windows_size = WINDOWS_SIZE
    _val_dataset_list.append(dataset)
val_dataset = torch.utils.data.ConcatDataset(_val_dataset_list)

gen_train, anc_train, win_train = train_dataset[0]
gen_test, anc_test, win_test = train_dataset[0]
assert gen_train.shape == gen_test.shape

INPUT_DIMENSION = gen_train.shape[0]
BATCH_SIZE = 2

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=20)

valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         shuffle=False, num_workers=20)


average_haploid = None
for i, data in enumerate(trainloader):
    inputs, _, labels = data
    inputs, labels = inputs.to(device).float(), labels.to(device)  # .float()
    if average_haploid is None:
        average_haploid = torch.mean(inputs, dim=0)
    else:
        average_haploid += torch.mean(inputs, dim=0)

_average_haploid = average_haploid / i
average_haploid = average_haploid.sign()


std_haploid = None
for i, data in enumerate(trainloader):
    inputs, _, labels = data
    inputs, labels = inputs.to(device).float(), labels.to(device)  # .float()
    if std_haploid is None:
        std_haploid = torch.mean((inputs - _average_haploid)**2, dim=0)
    else:
        std_haploid += torch.mean((inputs - _average_haploid)**2, dim=0)

_std_haploid = torch.sqrt(std_haploid / i)
print(_std_haploid)
print(_std_haploid.shape)
_std_haploid = None


HIDDEN_SIZE = 100
EMBEDDING_SIZE = 10

net= CVAELAI(WINDOWS_SIZE, INPUT_DIMENSION, hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, use_batch_norm=True, is_residual=IS_RESIDUAL, residual_avg=_average_haploid, residual_var=_std_haploid ,is_GAN=IS_GAN)

LAST_WINDOW_SIZE = net.CVAEList[-1].feature_size


## Output folder
folder_path_name = '/cvae-models/chm{}/{}_{}_{}/'.format(_CHROMOSOME,WINDOWS_SIZE,  HIDDEN_SIZE, EMBEDDING_SIZE)
if not os.path.exists(folder_path_name):
    os.makedirs(folder_path_name)

with open(folder_path_name + 'log.txt', 'a') as file:
    file.writelines('Window Size:{}, Generations train:{}, Missing Labels: {}, {} \n'.format(
        WINDOWS_SIZE,  _GENERATIONS_TRAIN, IS_MISSING_LABELS, MISSING_PERCENTAGE))



BCE_cirterion = nn.BCELoss()
CE_criterion = nn.CrossEntropyLoss()
MSE_criterion = nn.MSELoss()

optimizer = optim.Adam(net.CVAEList.parameters(), lr=0.01)
optimizer_dis = optim.Adam(net.DiscriminatorList.parameters(), lr=0.01)

print('Start training')
for epoch in range(200):
    running_loss, running_loss_clf, running_loss_rec, running_loss_rec_sign, running_loss_norm, running_loss_kld, running_loss_clf3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in enumerate(trainloader):
        net.train()
        inputs, _, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device)
        labels_one_hot = F.one_hot(labels, num_classes=3).float()
        # zero the parameter gradients
        optimizer.zero_grad()
        optimizer_dis.zero_grad()

        if IS_GAN:
            batch_size = inputs.shape[0]

            ### Train Discriminator
            sim, output_c = net.simulate(device=device, single_ancestry=True, batch_size=batch_size)
            sim = sim.sign()
            dis_out_fake = net.forward_discriminator(sim, output_c)
            dis_out_real = net.forward_discriminator(inputs, output_c)

            ones_label = torch.ones((batch_size, labels.shape[1]), device=device)
            zeros_label = torch.zeros((batch_size, labels.shape[1]), device=device)

            discriminator_loss = BCE_cirterion(dis_out_fake, zeros_label) + BCE_cirterion(dis_out_real, ones_label)
            discriminator_loss.backward()
            optimizer_dis.step()

            sim, output_c = net.simulate(device=device, single_ancestry=True, batch_size=batch_size)
            sim = sim.sign()
            dis_out_fake = net.forward_discriminator(sim, output_c)
            simulation_loss = -1 * BCE_cirterion(dis_out_fake, zeros_label)
            simulation_loss = torch.clamp(simulation_loss, 0, 10)

        if IS_NOISY_LABELS:
            _inputs = (torch.ones_like(inputs) - 2*F.dropout(torch.ones_like(inputs), p=(1-NOISY_PERCENTAGE))) * inputs
        else:
            _inputs = inputs

        if IS_MISSING_LABELS:
            _inputs = F.dropout(_inputs, p=MISSING_PERCENTAGE)
        else:
            _inputs = _inputs


        ## AutoEncoder + Classifier
        output_decoder, output_mu, output_logvar, res_out = net(_inputs, labels_one_hot)
        _x, _x_out = res_out

        loss_rec = MSE_criterion( _x, _x_out)
        loss_rec_sign = MSE_criterion(inputs, output_decoder.sign())
        loss_KLD = KLD_loss(output_mu, output_logvar)

        if IS_GAN:
            loss = 1.0 * loss_rec_sign + 0.01 * loss_rec + 1 * loss_KLD + 0.1 * simulation_loss
        else:
            loss = 1.0 * loss_rec_sign + 0.1 * loss_rec + 1* loss_KLD

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_rec += loss_rec.item()
        running_loss_rec_sign += loss_rec_sign.item()
        running_loss_kld += loss_KLD.item()

        if i % 2 == 0:
            print('[%d, %5d] loss: %.3f = %.3f + %.3f + %.3f' %
                  (epoch + 1, i + 1, running_loss / 2, running_loss_rec / 2, running_loss_rec_sign / 2,  running_loss_kld/ 2))
            running_loss, running_loss_rec, running_loss_rec_sign, running_loss_norm, running_loss_kld, running_loss_clf3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if epoch % 1 == 0:

        correct, total, correct_win, total_win = 0, 0, 0, 0

        correct_recons = 0
        correct_avg_haploid=0
        correct_simulation=0

        correct_pre = 0
        label_hist = [0,0,0]

        total_removed_elems = 0
        correct_recons_removed_elems = 0
        correct_avg_haploid_removed_elems = 0
        with torch.no_grad():
            net.eval()
            print('NET IS TRAINING: ', net.training)
            # Eval with batch size = 1. TODO: bigger batch size support needs to be added
            all_pred_list = []
            all_labels_list = []
            for j, data in enumerate(valloader):
                inputs, all_labels, labels = data
                inputs, all_labels, labels = inputs.to(device).float(), all_labels.to(device), labels.to(device)

                labels_one_hot = F.one_hot(labels, num_classes=3).float()

                if IS_NOISY_LABELS:
                    _inputs = (torch.ones_like(inputs) - 2 * F.dropout(torch.ones_like(inputs), p=(1 - NOISY_PERCENTAGE))) * inputs
                else:
                    _inputs = inputs

                if IS_MISSING_LABELS:
                    _inputs_before = _inputs
                    _inputs = F.dropout(_inputs, p=MISSING_PERCENTAGE)
                    removed_elems = torch.abs(_inputs_before.sign()) - torch.abs(_inputs.sign())
                else:
                    _inputs = _inputs

                output_decoder, output_mu, output_logvar, res_out = net(_inputs, labels_one_hot)
                labels, all_labels = torch.squeeze(labels), torch.squeeze(all_labels)
                total += all_labels.size(0)
                correct_recons += (output_decoder.sign() == inputs).sum().item()
                correct_avg_haploid +=(average_haploid == inputs).sum().item()
                print(correct_recons / total, correct_avg_haploid / total)

                if IS_MISSING_LABELS:
                    correct_recons_removed_elems += torch.clamp(torch.abs(output_decoder.sign() + inputs)[removed_elems>0],0,1).sum().item()
                    correct_avg_haploid_removed_elems += torch.clamp(torch.abs(average_haploid + inputs)[removed_elems>0],0,1).sum().item()
                    total_removed_elems += torch.sum(removed_elems).item()
                    print('Missin labels: ', correct_recons_removed_elems / total_removed_elems, correct_avg_haploid_removed_elems / total_removed_elems, total_removed_elems)


        with open(folder_path_name+'log.txt', 'a') as file:
            file.writelines('{} {}\n'.format(epoch, correct_recons/total))

        net_path_name = 'cvae_{}.pth'.format(epoch)
        torch.save(net, folder_path_name+net_path_name)

print('Finished Training')



