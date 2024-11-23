
import torch.nn.functional as F
import torch.nn as nn
import torch
from pytorch_metric_learning import testers

import numpy as np
import sys
from copy import deepcopy
from scipy.linalg import qr

np.random.seed(0)
torch.manual_seed(0)


def binarize_hard(x):
    return torch.where(x > 0, 1.0, -1.0)

def binarize_soft(x):
    return torch.tanh(x)


def HD_test(model, x_test, y_test):
	out = model(x_test, embedding=False)
	preds = torch.argmax(out, dim=-1)

	acc = torch.mean((preds==y_test).float())	
	print("Testing accuracy: ", acc.item())
	return acc

def get_Hamming_margin(model, x_test, y_test=None):
	def Hamming_distance(a, b):
		D = a.size()[1]
		return (D - a @ b.T)/2

	# Compute mean Hamming distance between class HVS
	class_hvs = binarize_hard(model.class_hvs.data)
	class_Hamming_distance = Hamming_distance(class_hvs, class_hvs)
	mean_class_Hamming_distance = torch.mean(class_Hamming_distance).item()
	

	# Compute test samples' Hamming distance
	test_enc_hvs = binarize_hard(model(x_test, True)) 
	test_Hamming_dist = Hamming_distance(test_enc_hvs, class_hvs)

	sorted_test_Hamming_distance, _ = torch.sort(test_Hamming_dist, dim=-1, descending=False)
	test_enc_hvs_Hamming_margin = (sorted_test_Hamming_distance[:,1:]-sorted_test_Hamming_distance[:,0].unsqueeze(dim=1)).mean(dim=1).cuda()
	mean_test_Hamming_margin = torch.mean(test_enc_hvs_Hamming_margin).item()

	res_dict = {
		"avg_class_Hamming_dist": mean_class_Hamming_distance,
		"avg_test_Hamming_margin": mean_test_Hamming_margin
	}
	return res_dict



class HDC(nn.Module):
	def __init__(self, dim, D, num_classes, enc_type='RP', binary=True, device='cuda', kargs=None):
		super(HDC, self).__init__()
		self.enc_type, self.binary = enc_type, binary	
		self.device = device

		if enc_type in ['RP', 'RP-COS']:
			self.rp_layer = nn.Linear(dim, D).to(device)
			self.class_hvs = torch.zeros(num_classes, D).float().to(device)
			self.class_hvs_nb = torch.zeros(num_classes, D).float().to(device)
		else:
			pass
		
	def encoding(self, x):
		if self.enc_type == 'RP':
			out = self.rp_layer(x)
		elif self.enc_type == 'RP-COS':
			out = torch.cos(self.rp_layer(x))
		elif self.enc_type == 'SVD':
			pass
		elif self.enc_type == 'ID':
			pass
		else:
			raise Exception("Error encoding type: {}".format(self.enc_type))
		
		return binarize_soft(out) if self.binary else out

	def init_class(self, x_train, labels_train):
		out = self.encoding(x_train)
		if self.binary:
			out = binarize_hard(out)

		for i in range(x_train.size()[0]):
			self.class_hvs[labels_train[i]] += out[i]
		
		if self.binary:
			self.class_hvs = binarize_hard(self.class_hvs)
			
	def HD_train_step(self, x_train, y_train, lr=1.0):
		shuffle_idx = torch.randperm(x_train.size()[0])
		x_train = x_train[shuffle_idx]
		train_labels = y_train[shuffle_idx]

		enc_hvs = binarize_hard(self.encoding(x_train))
		for i in range(enc_hvs.size()[0]):
			sims = self.similarity(self.class_hvs, enc_hvs[i].unsqueeze(dim=0))
			predict = torch.argmax(sims, dim=1)
			
			if predict != train_labels[i]:
				self.class_hvs_nb[predict] -= lr * enc_hvs[i]
				self.class_hvs_nb[train_labels[i]] += lr * enc_hvs[i]
			
			self.class_hvs.data = binarize_hard(self.class_hvs_nb)

	
	def similarity(self, class_hvs, enc_hv):
		# class_hvs = torch.div(class_hvs, torch.norm(class_hvs, dim=1, keepdim=True))
		# enc_hv = torch.div(enc_hv, torch.norm(enc_hv, dim=1, keepdim=True))
		return torch.matmul(enc_hv, class_hvs.t())/class_hvs.size()[1]

	def forward(self, x, embedding=True):
		out = self.encoding(x)
		if embedding:
			out = out
		else:
			out = self.similarity(class_hvs=binarize_hard(self.class_hvs), enc_hv=out)      
		return out


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def metric_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data.reshape(data.size()[0],-1))
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def metric_test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


def binarize(base_matrix):
	return np.where(base_matrix < 0, -1, 1).astype(float)

def gen_base_matrix(m, n, is_binary=True):
    Q = np.random.rand(m, n)-0.5
    Q, R = qr(Q)
    if is_binary:
        return np.where(Q[:m, :n] > 0.0, 1, -1).astype(float)
    else:
        return Q[:m, :n]

def encoding_rp(X_data, base_matrix, signed=False):
	enc_hvs = []
	for i in range(len(X_data)):
		hv = np.matmul(base_matrix, X_data[i])
		if signed:
			hv = binarize(hv)
		enc_hvs.append(hv)
	return enc_hvs

def encoding_Kron(X_data, base_submat, signed=True, log=True):
	enc_hvs = []
	Kron_dim = base_submat[0].shape
	for i in range(len(X_data)):
		hv = X_data[i].reshape(Kron_dim[1], -1).dot(base_submat[1].T)
		hv = base_submat[0].dot(hv).flatten()
  
		if signed:
			hv = binarize(hv)
		enc_hvs.append(hv)
	return enc_hvs

def max_match(class_hvs, enc_hv, class_norms, binary=False):
		max_score = -np.inf
		max_index = -1
		for i in range(len(class_hvs)):
			if binary:
				score = np.matmul(class_hvs[i], enc_hv)
			else:
				score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
			
			if score > max_score:
				max_score = score
				max_index = i
		return max_index


from RRAM import RRAM
def test_RRAM_HD(model, x_test, y_test, kargs):
	class_hvs = binarize_hard(model.class_hvs)
	test_hvs = binarize_hard(model(x_test, embedding=True))
	
	preds = torch.argmax(test_hvs@class_hvs.T, dim=-1)
	fp_acc = torch.mean((preds==y_test).float()).item()

	unipolar_class_hvs = torch.where(class_hvs>0, 1, 0).cpu().numpy()
	unipolar_test_hvs = torch.where(test_hvs>0, 1, 0).cpu().numpy()
	
	rram_chip = RRAM(S_ou=kargs['S_ou'], R=kargs['R'], R_deviation=kargs['R_deviation'])
	
	res_dict = {
        "S_ou": kargs['S_ou'],
		"R": kargs['R'], 
		"R_deviation": kargs['R_deviation'],
        "fp_acc": fp_acc,
        "test_acc": []
    }
	
	for i in range(5):
		rram_chip.rram_write_binary(unipolar_class_hvs)
		
		preds, Hamming_sim_cim = rram_chip.rram_hd_am(unipolar_test_hvs, collect_stats=True)
		
		acc = torch.mean((torch.tensor(preds).cuda() == y_test).float()).item()
		res_dict["test_acc"].append(acc)
	
	return res_dict


def train_binary(X_train, y_train, X_test, y_test, D=1024, alg='rp', Kron_shape=[32,28], epoch=20, lr=1.0, L=64, quantize=0):
	# randomly select 20% of train data as validation
	permvar = np.arange(0, len(X_train))
	np.random.shuffle(permvar)
	X_train = [X_train[i] for i in permvar]
	y_train = [y_train[i] for i in permvar]
	cnt_vld = int(0.2 * len(X_train))
	X_validation = X_train[0:cnt_vld]
	y_validation = y_train[0:cnt_vld]
	X_train = X_train[cnt_vld:]
	y_train = y_train[cnt_vld:]

	# encodings
	if alg in ['rp', 'rp-sign']:
		# create base matrix
		base_matrix = gen_base_matrix(D, len(X_train[0]))
		
		print('\nEncoding ' + str(len(X_train)) + ' train data')
		train_enc_hvs = encoding_rp(X_train, base_matrix, signed=(alg == 'rp-sign'))
  
		print('\nEncoding ' + str(len(X_validation)) + ' validation data')
		validation_enc_hvs = encoding_rp(X_validation, base_matrix, signed=(alg == 'rp-sign'))
	elif alg in ['rp-Kron']:
		# create base matrix for Kronecker product
		base_matrix = []
		base_matrix.append(gen_base_matrix(Kron_shape[0], Kron_shape[1]))
		base_matrix.append(gen_base_matrix(D//Kron_shape[0], len(X_train[0])//Kron_shape[1]))

		print('\nEncoding ' + str(len(X_train)) + ' train data')
		train_enc_hvs = encoding_Kron(X_train, base_matrix, signed=True)
  
		print('\nEncoding ' + str(len(X_validation)) + ' validation data')
		validation_enc_hvs = encoding_Kron(X_validation, base_matrix, signed=True)
 
	# training, initial model
	class_hvs_fp = np.array([[0.] * D] * (max(y_train) + 1))
	for i in range(len(train_enc_hvs)):
		class_hvs_fp[y_train[i]] += train_enc_hvs[i]
	
	class_hvs_bin = deepcopy(binarize(class_hvs_fp))
	class_hvs_best = deepcopy(class_hvs_bin)
    
	# retraining
	if epoch > 0:
		acc_max = -np.inf
		print('\n' + str(epoch) + ' retraining epochs')
		for i in range(epoch):
			sys.stdout.write('epoch ' + str(i) + ': ')
			sys.stdout.flush()
			# shuffle data during retraining
			pickList = np.arange(0, len(train_enc_hvs))
			np.random.shuffle(pickList)
			class_hvs_bin = deepcopy(binarize(class_hvs_fp))
			for j in pickList:
				predict = max_match(class_hvs_bin, train_enc_hvs[j], class_norms=None, binary=True)
				if predict != y_train[j]:
					class_hvs_fp[predict] -= np.multiply(lr, train_enc_hvs[j])
					class_hvs_fp[y_train[j]] += np.multiply(lr, train_enc_hvs[j])

			correct = 0
			for j in range(len(validation_enc_hvs)):
				predict = max_match(class_hvs_bin, validation_enc_hvs[j], class_norms=None, binary=True)
				if predict == y_validation[j]:
					correct += 1
			acc = float(correct)/len(validation_enc_hvs)
			sys.stdout.write("%.4f " %acc)
			sys.stdout.flush()
			if i > 0 and i%5 == 0:
				print('')
			if acc > acc_max:
				acc_max = acc
				class_hvs_best = deepcopy(class_hvs_bin)			

	del X_train, X_validation, train_enc_hvs, validation_enc_hvs

	if alg == 'rp' or alg == 'rp-sign':
		test_enc_hvs = encoding_rp(X_test, base_matrix, signed=(alg == 'rp-sign'))
	elif alg in ['rp-Kron']:
		test_enc_hvs = encoding_Kron(X_test, base_matrix, signed=True)
  
	correct = 0
	for i in range(len(test_enc_hvs)):
		predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms=None, binary=True)
		if predict == y_test[i]:
			correct += 1
	acc = float(correct)/len(test_enc_hvs)
	return acc, class_hvs_best, base_matrix


def run_test(class_hvs, X_test, y_test, base_matrix, alg='rp-Kron'):
    if alg == 'rp' or alg == 'rp-sign':
        test_enc_hvs = encoding_rp(X_test, base_matrix, signed=(alg == 'rp-sign'))
    elif alg in ['rp-Kron']:
        test_enc_hvs = encoding_Kron(X_test, base_matrix, signed=True)
    
    correct = 0
    for i in range(len(test_enc_hvs)):
        predict = max_match(class_hvs, test_enc_hvs[i], class_norms=None, binary=True)
        if predict == y_test[i]:
            correct += 1
    
    acc = float(correct)/len(test_enc_hvs)
    print("Testing accuracy: ", acc)
    return acc
    

def run_train():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    train_images = np.array(train_images).reshape(60000,-1)
    test_images = np.array(test_images).reshape(10000,-1)

    acc, train_hvs, enc_base = train(
        X_train=train_images, y_train=train_labels,
        X_test=test_images, y_test=test_labels,
        D=1024, alg = 'rp-Kron', Kron_shape=[32, 28], epoch=5, lr=1.0)
    
    # acc, class_hvs, enc_base_matrix = train_binary(
    #     X_train=train_images, y_train=train_labels,
    #     X_test=test_images, y_test=test_labels,
    #     D=1024, alg = 'rp-Kron', Kron_shape=[32, 28], epoch=5, lr=1.0)

    print("\nTest accuracy: {:.4f}".format(acc))
    