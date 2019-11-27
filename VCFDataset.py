from __future__ import print_function, division
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import vcf
import numpy as np
import os

MAP_KEY_ASCII_MAP = [u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u':',u';',u'<',u'=',u'>',u'?',u'@',u'A']

def get_vcf_headers(vcf_path):
    vcf_reader = vcf.Reader(open(vcf_path, 'r'))
    return vcf_reader.samples

def generate_map_from_single_ancestry_vcf(vcf_path, out_path, ancestry):
    samples = get_vcf_headers(vcf_path)
    with open(out_path, 'a') as out_file:
        for sample in samples:
            line = sample + '\t' + ancestry + '\n'
            out_file.write(line)


class VCFDataset(Dataset):
    """VCF dataset."""

    def __init__(self, vcf_path, map_path, out_path, windows_size=-1, is_haploid=True, random_diploid_switch=True, phase_switch=0.1, use_subset_positions=False, subset_positions_file=None, norm_output=False, norm_params=None, is_missing_data=False, missing_percent=0, balance_dataset=False, single_ancestry=False, ancestry=0, printout=False, is_hilbert=False):
        """
        Args:
            TODO
        """

        self.is_haploid = is_haploid
        self.random_diploid_switch = random_diploid_switch
        self.phase_switch = phase_switch

        self.is_missing_data = is_missing_data
        self.missing_percent = missing_percent
        self.windows_size = windows_size

        self.single_ancestry = single_ancestry

        self.use_subset_positions = use_subset_positions
        if self.use_subset_positions:
            self.subset_positions = np.load(subset_positions_file)


        ## Load VCF file
        print('loading vcf file...')
        mat_vcf_path = os.path.join(out_path, 'mat_vcf.npy')
        if os.path.exists(mat_vcf_path):
            mat_vcf_3d = np.load(mat_vcf_path)
            if mat_vcf_3d[0,0].dtype is not np.uint8:
                mat_vcf_3d = np.uint8(mat_vcf_3d)
                np.save(mat_vcf_path, mat_vcf_3d)
        else:
            print('reading vcf file...')
            vcf_reader = vcf.Reader(open(vcf_path, 'r'))

            mat_list = []

            for i, record in enumerate(vcf_reader):
                p_0 = record.samples[0]['GT'][0]
                p_1 = record.samples[0]['GT'][2]
                if printout:
                    print(i, record, p_0, p_1, record.samples[0]['GT'][1])
                record_list = []

                for j, sample in enumerate(record.samples):
                    p_0 = sample['GT'][0]
                    p_1 = sample['GT'][2]
                    record_list.append([p_0, p_1])

                mat_list.append(record_list)


            mat_vcf = np.array(mat_list)
            mat_vcf = mat_vcf.swapaxes(0, 1)

            mat_vcf_3d = mat_vcf
            mat_vcf_3d = np.uint8(mat_vcf_3d)
            np.save(mat_vcf_path, mat_vcf_3d)


        print('done...')
        self.mat_vcf_3d = mat_vcf_3d

        mat_vcf_path_2d = os.path.join(out_path, 'mat_vcf_2d.npy')
        if os.path.exists(mat_vcf_path_2d):
            mat_vcf_2d = np.load(mat_vcf_path_2d)
        else:
            mat_vcf_2d = np.zeros((int(mat_vcf_3d.shape[0]*2), mat_vcf_3d.shape[1]))
            mat_vcf_2d[::2, :] = mat_vcf_3d[:, :, 0]
            mat_vcf_2d[1::2, :] = mat_vcf_3d[:, :, 1]
            np.save(mat_vcf_path_2d, mat_vcf_2d)

        self.mat_vcf_2d = mat_vcf_2d

        self.mat_vcf_2d = (self.mat_vcf_2d-0.5)*2
        self.mat_vcf_3d = (self.mat_vcf_3d - 0.5) * 2

        if not self.single_ancestry:
            ## Loading MAP file #################################
            print('loading map file...')
            mat_map_path = os.path.join(out_path, 'mat_map.npy')
            if os.path.exists(mat_map_path):
                mat_map_2d = np.load(mat_map_path)
            else:
                print('reading map file...')
                mat_map_str = np.loadtxt(map_path, skiprows=1, dtype=np.unicode_)
                mat_map = np.zeros_like(mat_map_str, dtype=np.int32)-1
                for k in MAP_KEY_ASCII_MAP:
                    print(k, ord(k) - 49)
                    mat_map[mat_map_str == k] = ord(k) - 49 ## When using more than 9 ancestries, parsing to int does not work, ASCII parsing needed
                print(np.max(mat_map), np.min(mat_map), set(mat_map.flatten()))
                mat_map_2d = mat_map[:, 2:]
                mat_map_2d = mat_map_2d.swapaxes(0, 1).astype(int)
                print(np.max(mat_map_2d), np.min(mat_map_2d), set(mat_map_2d.flatten()))
                np.save(os.path.join(out_path, 'mat_map.npy'), mat_map_2d)

            mat_map_3d = np.zeros((int(mat_map_2d.shape[0] / 2), mat_map_2d.shape[1], 2))
            mat_map_3d[:, :, 0] = mat_map_2d[::2, :]
            mat_map_3d[:, :, 1] = mat_map_2d[1::2, :]

            self.mat_map_2d = mat_map_2d #- 1
            self.mat_map_3d = mat_map_3d #- 1

        else:
            self.mat_map_2d = np.zeros((int(mat_vcf_2d.shape[0]), mat_vcf_2d.shape[1]), dtype=np.int)
            self.mat_map_3d = np.zeros((int(mat_vcf_2d.shape[0] / 2), mat_vcf_2d.shape[1], 2), dtype=np.int)
            # if ancestry is 'AFR':
            #     self.mat_map_2d += 0
            #     self.mat_map_3d += 0
            # elif ancestry is 'EUR':
            #     self.mat_map_2d += 2
            #     self.mat_map_3d += 2
            # elif ancestry is 'EAS':
            #     self.mat_map_2d += 1
            #     self.mat_map_3d += 1
            self.mat_map_2d += ancestry
            self.mat_map_3d += ancestry


        self.norm_output = norm_output
        if self.norm_output:
            if norm_params is None:
                self.mean = self.mat_vcf_2d.mean(axis=0)
                self.var = self.mat_vcf_2d.std(axis=0)
            else:
                self.mean, self.var = norm_params
        else:
            self.mean = None
            self.var = None


        print(self.mat_map_2d.shape, self.mat_map_3d.shape, self.mat_vcf_2d.shape, self.mat_vcf_3d.shape)

        if self.use_subset_positions:
            self.mat_map_2d = self.mat_map_2d[:, self.subset_positions]
            self.mat_map_3d = self.mat_map_3d[:, self.subset_positions,:]
            self.mat_vcf_2d = self.mat_vcf_2d[:, self.subset_positions]
            self.mat_vcf_3d = self.mat_vcf_3d[:, self.subset_positions, :]

        print(self.mat_map_2d.shape, self.mat_map_3d.shape, self.mat_vcf_2d.shape, self.mat_vcf_3d.shape)


    def __len__(self):
        if self.is_haploid:
            return self.mat_map_3d.shape[0]*2
        else:
            return self.mat_map_3d.shape[0]


    def anc2win(self, anc, windows_size):
        input_dimension = len(anc)
        num_windows = int(np.floor(input_dimension/windows_size))
        anc_win = []
        for j in range(num_windows):
            if j == num_windows-1:
                _x = anc[j*windows_size:]
            else:
                _x = anc[j * windows_size:(j + 1) * windows_size]
            (values, counts) = np.unique(_x, return_counts=True)
            ind = np.argmax(counts)
            anc_win.append(values[ind])

        return np.array(anc_win)

    def __getitem__(self, idx):
        if self.is_haploid:
            return self.get_item_haploid(idx)
        else:
            return self.get_item_diploid(idx)


    def get_item_haploid(self, idx):
        gen = self.mat_vcf_2d[idx, :]
        anc = self.mat_map_2d[idx, :]

        if self.windows_size > 1:
            win = self.anc2win(anc, self.windows_size)
        else:
            win = anc


        if self.is_missing_data:
            gen = torch.tensor(gen)
            gen = F.dropout(gen, p=self.missing_percent)

        if self.norm_output:
            gen = (gen - self.mean) / self.var



        return gen, anc, win


    def get_item_diploid(self, idx):
        gen = self.mat_vcf_3d[idx, :, :]
        anc = self.mat_map_3d[idx, :, :]

        if self.random_diploid_switch and np.random.random()>0.5:
            gen = np.stack([gen[:,1],gen[:,0]], axis=1)
            anc = np.stack([anc[:,1],anc[:,0]], axis=1)

        if self.phase_switch > 0.0:
            diff = gen[:, 0] * gen[:, 1]
            rand = np.sign(np.random.rand(*diff[diff==-1].shape) - self.phase_switch)
            gen[diff==-1,:] = gen[diff==-1,:] * np.stack([rand, rand], axis=1)

        if self.windows_size > 1:
            win = np.stack([self.anc2win(anc[:,0], self.windows_size), self.anc2win(anc[:,1], self.windows_size)], axis=1)
        else:
            win = anc


        if self.is_missing_data:
            gen = torch.tensor(gen)
            gen = F.dropout(gen, p=self.missing_percent)

        if self.norm_output:
            gen = (gen - self.mean) / self.var

        return gen, anc, win



class RFMixOutputDataset(Dataset):
    """RFMixOutputDataset dataset."""

    def __init__(self, tsv_path, map_path, out_path, num_categories=3):
        """
        Args:
            TODO
        """

        NUM_CATEGORIES = num_categories

        ## Loading MAP file #################################
        print('loading map file...')
        mat_map_path = os.path.join(out_path, 'mat_map_RF.npy')
        mat_map_path_pos = os.path.join(out_path, 'mat_map_pos_RF.npy')
        if os.path.exists(mat_map_path) and os.path.exists(mat_map_path_pos):
            mat_map_2d = np.load(mat_map_path)
            mat_map_POS = np.load(mat_map_path_pos)
        else:
            print('reading map file...')
            # mat_map = pd.read_csv(map_path, delimiter=r'\t')
            mat_map = np.loadtxt(map_path, skiprows=1)
            mat_map_POS = mat_map[:, 1]
            mat_map_2d = mat_map[:, 2:]
            mat_map_2d = mat_map_2d.swapaxes(0, 1).astype(int)
            np.save(os.path.join(out_path, 'mat_map_RF.npy'), mat_map_2d)
            np.save(os.path.join(out_path, 'mat_map_pos_RF.npy'), mat_map_POS)

        mat_map_3d = np.zeros((int(mat_map_2d.shape[0] / 2), mat_map_2d.shape[1], 2))
        mat_map_3d[:, :, 0] = mat_map_2d[::2, :]
        mat_map_3d[:, :, 1] = mat_map_2d[1::2, :]

        self.mat_map_2d = mat_map_2d - 1
        self.mat_map_3d = mat_map_3d - 1

        elems, counts = np.unique(self.mat_map_2d, return_counts=True)
        print('ELEMS ARE: ,', elems)
        print('COUNTS ARE: ,', counts)
        self.counts = counts

        ## Load TSV file
        print('loading tsv file...')
        mat_tsv_path = os.path.join(out_path, 'mat_tsv.npy')
        if os.path.exists(mat_tsv_path):
            mat_tsv_3d = np.load(mat_tsv_path)
        else:
            with open(tsv_path, 'r') as f:
                i = 0
                for line in f.readlines():
                    print(line)
                    i+=1
                    if i>5:
                        break
            mat_tsv = np.loadtxt(tsv_path, skiprows=2)
            mat_tsv = mat_tsv.swapaxes(0,1)
            POS_tsv = mat_tsv[1,:]
            GMI_tsv = mat_tsv[3, :]
            mat_tsv = mat_tsv[4:,:]
            mat_tsv = np.round(mat_tsv)

            pos_map = mat_map_POS.squeeze()
            pos_tsv = POS_tsv.squeeze()
            pos_map_set = set(pos_map)

            pos_tsv_set = set(pos_tsv)
            sorting_idx = np.argsort(pos_map)

            plt.plot(sorting_idx)
            plt.show()

            indexes = np.searchsorted(pos_map[sorting_idx], pos_tsv)

            self.mat_map_2d = self.mat_map_2d[:, sorting_idx[indexes]]

            _reshape = mat_tsv.reshape((int(mat_tsv.shape[0]/NUM_CATEGORIES),NUM_CATEGORIES,mat_tsv.shape[1]))
            argm = np.argmax(_reshape, axis=1) #+ 1

            acc2 = np.mean((argm == self.mat_map_2d))
            acc = np.sum(np.sum(1-np.clip(np.abs(argm - self.mat_map_2d),0,1)))/(self.mat_map_2d.shape[0]*self.mat_map_2d.shape[1])
            print('Accuracy is ,', acc, acc2)

            self.accuracy = acc

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        return None
