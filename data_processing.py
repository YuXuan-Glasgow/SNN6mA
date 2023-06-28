
import random
import numpy as np
import torch
import xlrd


def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_pos = workbook.sheet_by_index(0)
    nrows_pos = booksheet_pos.nrows

    booksheet_neg = workbook.sheet_by_index(1)
    nrows_neg = booksheet_neg.nrows

    seq = []
    label = []
    for i in range(nrows_pos):
        seq.append(booksheet_pos.row_values(i)[0])
        label.append(int(booksheet_pos.row_values(i)[1]))
    for j in range(nrows_neg):
        seq.append((booksheet_neg.row_values(j)[0]))
        label.append(int(booksheet_neg.row_values(j)[1]))


    return seq, torch.tensor(label)


def ACGTto0123(filename):
    seq, label = read_seq_label(filename)
    seq0123 = []
    for i in range(len(seq)):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq0123.append(one_seq)
    return seq0123, label


def seq_to01_to0123(filename):
    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='float32')
    seq_0123 = np.zeros((nrows, seq_len), dtype='float32')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_01, seq_0123, label


def load_data(filename):
    seq01, seq_0123, label = seq_to01_to0123(filename)

    r = random.random
    random.seed(2)
    a = np.linspace(0, len(label) - 1, len(label)).astype(int)
    random.shuffle(a, random=r)

    num_total = len(label)
    num_train = int(len(label) * 0.8)
    num_val = int(len(label) * 0.1)
    num_test = num_total - num_train - num_val

    train_index = a[:num_train]
    valid_index = a[num_train:num_train + num_val]
    test_index = a[num_train + num_val:num_total]

    x_train = seq01[train_index, :, :]
    x_val = seq01[valid_index, :, :]
    x_test = seq01[test_index, :, :]

    y_train = label[train_index]
    y_val = label[valid_index]
    y_test = label[test_index]

    return x_train, y_train, x_val, y_val, x_test, y_test


def ACGTto00011011(filename):
    seq, label = read_seq_label(filename)
    seq00011011 = []
    for i in range(len(seq)):
        one_seq = seq[i]

        one_seq = one_seq.replace('A', '00')
        one_seq = one_seq.replace('C', '01')
        one_seq = one_seq.replace('G', '10')
        one_seq = one_seq.replace('T', '11')

        seq00011011.append(one_seq)
    return seq00011011, label


def autoimg(filename, n):
    seq, label = ACGTto00011011(filename)
    nrows = len(seq)
    seq_len = len(seq[0])

    dict3to1 = {'111': 0, '110': 1, '101': 0, '100': 1,
                '011': 0, '010': 1, '001': 0, '000': 0}

    seq_autoimg = np.zeros((nrows, seq_len, n), dtype='float32')
    for i in range(nrows):

        one_seq = seq[i]
        seq_autoimg[i, :, 0] = np.array([one_seq[q] for q in range(seq_len)])
        one_seq = '1' + one_seq + '1'
        one_seq = one_seq[-1] + one_seq + one_seq[0]

        for j in range(1, n):

            temp = np.array([dict3to1[one_seq[k:k + 3]] for k in range(seq_len)])
            seq_autoimg[i, :, j] = temp

            arr = temp.tolist()
            str2 = ''.join(str(i) for i in arr)
            one_seq = '1' + str2 + '1'
            one_seq = str2[-1] + str2 + str2[0]


    return seq_autoimg, label


def load_img_data(filename, n):
    seq, label = autoimg(filename, n)

    r = random.random
    random.seed(2)
    a = np.linspace(0, len(label) - 1, len(label)).astype(int)
    random.shuffle(a, random=r)

    num_total = len(label)
    num_train = int(len(label) * 0.8)
    num_val = int(len(label) * 0.1)
    num_test = num_total - num_train - num_val

    train_index = a[:num_train]
    valid_index = a[num_train:num_train + num_val]
    test_index = a[num_train + num_val:num_total]

    x_train = seq[train_index, :, :]
    x_val = seq[valid_index, :, :]
    x_test = seq[test_index, :, :]

    y_train = label[train_index]
    y_val = label[valid_index]
    y_test = label[test_index]
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    filename = 'data/test.xlsx'
    load_img_data(filename, n=100)