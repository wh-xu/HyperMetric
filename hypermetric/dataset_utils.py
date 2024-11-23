import struct

import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


DATASET_ROOT = "./dataset"


def readChoirDat(filename):
	""" Parse a choir_dat file """
	with open(filename, 'rb') as f:
		nFeatures = struct.unpack('i', f.read(4))[0]
		nClasses = struct.unpack('i', f.read(4))[0]
		
		X, y = [], []
		while True:
			newDP = []
			for i in range(nFeatures):
				v_in_bytes = f.read(4)
				if v_in_bytes is None or len(v_in_bytes) == 0:
					return nFeatures, nClasses, X, y

				v = struct.unpack('f', v_in_bytes)[0]
				newDP.append(v)

			l = struct.unpack('i', f.read(4))[0]
			X.append(newDP)
			y.append(l)

	return nFeatures, nClasses, X, y


def load_dataset(dataset_name, batch_size, device='cuda'):
    if dataset_name in ['isolet', 'mnist', 'face', 'cardio3', 'pamap2', 'ucihar']:
        nFeatures, nClasses, x_train, y_train = readChoirDat(filename=DATASET_ROOT+'/'+dataset_name+'_train.choir')
        x_train, y_train = torch.Tensor(x_train).to(device), torch.LongTensor(y_train).to(device)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        nFeatures, nClasses, x_test, y_test = readChoirDat(filename=DATASET_ROOT+'/'+dataset_name+'_test.choir')
        x_test, y_test = torch.Tensor(x_test).to(device), torch.LongTensor(y_test).to(device)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    else:
        raise Exception("Unknown dataset name: ", dataset_name)

    return nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader
