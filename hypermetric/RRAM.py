import numpy as np
from tqdm import tqdm as tqdm
from scipy import stats
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


class RRAM:
    def __init__(self, R=[2500, 16000], R_deviation=[0.18, 0.45], Vread=0.9, WL_resolution=1, bits_per_cell=1, S_ou=4, pdf_type='lognorm', device='cpu'):
        """
            R: Resistance for ON/OFF states
            R_deviation: Resistance deviation for ON/OFF states
            Vread: Voltage for read opeartion
            WL_resolution: Number of bits to encode input voltage levels, only support 1 for current version
            S_ou: Number of operation unit (activated WL for computing)
            pdf_type: Probability distribution to model resistance distribution - lognorm has not been verified
        """
        self.Ron, self.Roff = R
        self.R_ratio = self.Roff/self.Ron
        self.R, self.R_sigma = R, R_deviation

        self.Vread = Vread
        self.WL_resolution = WL_resolution
        self.bits_per_cell = bits_per_cell
        self.S_ou = S_ou

        self.pdf_type = pdf_type
        self.device = device

        self.Ron_dist = self.init_distribution(
            mu=self.Ron, sigma=self.R_sigma[0])
        self.Roff_dist = self.init_distribution(
            mu=self.Roff, sigma=self.R_sigma[1])
        self.init_ref_current_binary()
        self.init_ref_current_cim()

    def init_distribution(self, mu, sigma, pdf_type='lognorm'):
        if pdf_type == 'norm':
            loc, scale = mu, sigma*mu
            return stats.norm(loc=loc, scale=scale)
        elif pdf_type == 'lognorm':
            shape, scale = sigma, mu
            return stats.lognorm(s=shape, scale=scale)
        else:
            raise Exception("Wrong pdf type!")

    def init_ref_current_binary(self):
        # self.ref_current_binary = np.zeros(2)
        # self.ref_current_binary[0] = self.Vread/self.Roff
        # self.ref_current_binary[1] = self.Vread/self.Ron

        self.ref_current_binary = (
            self.Vread/self.Roff + self.Vread/self.Ron) / 2

    def init_ref_current_cim(self, scheme='dual_ref'):
        if scheme == 'dual_ref':
            self.ref_current_cim = np.zeros(self.S_ou, dtype=np.float32)
            for i in range(self.S_ou):
                n_lrs, n_hrs = i, self.S_ou-i
                self.ref_current_cim[i] = (
                    (self.Vread/self.Ron*n_lrs + self.Vread/self.Roff*n_hrs) + (self.Vread/self.Ron*(n_lrs+1))) / 2
                # self.ref_current_cim[i,0] = self.Vread/self.Ron*n_lrs + self.Vread/self.Roff*n_hrs
                # self.ref_current_cim[i,1] = self.Vread/self.Ron*(n_lrs+1)
        else:
            raise NotImplementedError("Wrong sensing scheme: ", scheme)

    def rram_sense_current_binary(self, read_current, scheme='dual_ref'):
        if scheme == 'dual_ref':
            sense_diff = self.ref_current_binary - read_current
            sense_data = np.where(sense_diff <= 0, 1, 0).astype(np.float32)
            return sense_data
        else:
            raise NotImplementedError("Wrong sensing scheme: ", scheme)

    def rram_sense_current_cim(self, read_current: np.ndarray, scheme='dual_ref'):
        if scheme == 'dual_ref':
            sense_diff = read_current[..., np.newaxis]-self.ref_current_cim
            min_idx = np.argmin(np.abs(sense_diff), axis=-1)
            min_diff = sense_diff[np.arange(len(sense_diff)), min_idx]
            sense_data = np.minimum(min_idx + (min_diff > 0), self.S_ou)
            return sense_data.astype(np.float32)
        else:
            raise NotImplementedError("Wrong sensing scheme: ", scheme)

    def rram_write_binary(self, array_data: np.ndarray):
        """
            Create a ReRAM array with single-level cell (SLC) data
        """
        shape = array_data.shape

        rand_Ron = self.Ron_dist.rvs(array_data.size).reshape(shape)
        rand_Roff = self.Roff_dist.rvs(array_data.size).reshape(shape)

        self.rram_array = rand_Ron*array_data + rand_Roff*(1-array_data)

        if self.device == 'cpu':
            self.rram_array = np.array(self.rram_array, dtype=np.float32)

    def rram_read_binary(self, idx=None):
        """
            Perform ReRAM read operation for binary data
        """
        if idx:
            return self.rram_sense_current_binary(self.Vread/self.rram_array[idx])
        else:
            return self.rram_sense_current_binary(self.Vread/self.rram_array)

    def rram_imc(self, idx):
        """
            Perform ReRAM in-memory computing by idx addressing
        """
        assert len(idx) == self.S_ou, "Act. WL num should equal to S_ou"

        Res = self.rram_array[idx]
        BL_current = np.sum(self.Vread/Res, axis=0).flatten()
        print('BL current:\n', BL_current)
        print('Ref current:\n', self.ref_current_cim)
        cim_out = self.rram_sense_current_cim(BL_current)
        return cim_out

    def rram_hd_am(
        self, 
        query_hvs:np.ndarray, 
        collect_stats:bool=False):
        """
            HD associative memory (AM) search using ReRAM
            
            query_hvs: query HVS to search - shape [N, D]
        """
        def padding(x):
            # TODO: Add vector padding
            pass

        BL_current = self.Vread * query_hvs[:, np.newaxis, :] / self.rram_array # N, C, D
        acc_current = BL_current.reshape(
            BL_current.shape[0], BL_current.shape[1], -1, self.S_ou).sum(axis=-1)
        original_shape = acc_current.shape
        
        num_match_1 = self.rram_sense_current_cim(
            acc_current.flatten()).reshape(original_shape) # N, C, D//S_ou

        BL_current_inv = self.Vread * (1.0-query_hvs[:, np.newaxis, :]) / self.rram_array
        acc_current = BL_current_inv.reshape(
            BL_current_inv.shape[0], BL_current_inv.shape[1], -1, self.S_ou).sum(axis=-1)
        num_unmatch_01 = self.rram_sense_current_cim(
            acc_current.flatten()).reshape(original_shape)

        zero_cnt_query = (1-query_hvs).reshape(query_hvs.shape[0],-1, self.S_ou).sum(axis=-1) # N, D//S_ou
        zero_cnt_query = zero_cnt_query[:, np.newaxis, :]
        Hamming_sim_seg = zero_cnt_query - num_unmatch_01 + num_match_1
        Hamming_sim = (Hamming_sim_seg).sum(axis=-1)
        preds = np.argmax(Hamming_sim, axis=-1).astype(int)
        
        if collect_stats:
            return preds, Hamming_sim_seg
        else:
            return preds


    def build_BL_accumulated_current_distribution(self):
        # plt.figure()
        self.BL_acc_current_chart = []
        for i in range(self.S_ou+1):
            n_lrs, n_hrs = i, self.S_ou-i

            current_Ron = self.init_distribution(mu=self.Vread/self.Ron*n_lrs, sigma=self.R_sigma[0])
            current_Roff = self.init_distribution(mu=self.Vread/self.Roff*n_hrs, sigma=self.R_sigma[1])

            acc_current = current_Ron.rvs(100000) + current_Roff.rvs(100000)
            fit_parms = stats.lognorm.fit(acc_current, floc=0)
            lognorm_dist = stats.lognorm(*fit_parms)
            self.BL_acc_current_chart.append(lognorm_dist)
            # sns.distplot(lognorm_dist.rvs(100000), label='WL={}'.format(i))
        
        # plt.legend()
        # plt.title('BL Accumulated Current Distribution')


    def build_error_table(self):
        file_path = './error_tables/error_table_Ron_{}_Roff_{}_Rsigma_{}_Sou_{}_Vr_{}_WLr_{}_cellbits_{}_pdftype_{}.npy'.format(
            self.Ron, self.Roff, self.R_sigma, self.S_ou, self.Vread, self.WL_resolution, self.bits_per_cell, self.pdf_type)

        if os.path.exists(file_path):
            print('Found existing error tables.')
            error_table = np.load(file_path)
        else:            
            error_table = np.identity(self.S_ou+1)

            # self.build_BL_accumulated_current_distribution() # Method 1
            # for i in tqdm(range(self.S_ou+1), desc='Building error table'):
                # Method 1:
                # BL_current_samples = self.BL_acc_current_chart[i].rvs(int(1e6))
                # for j in range(self.S_ou+1):
                    # error_table[i, j] = np.count_nonzero(sense_output==j)/sense_output.size
                    
                # Method 2:
                # n_lrs, n_hrs = i, self.S_ou-i
                # current_Ron = self.init_distribution(mu=self.Vread/self.Ron*n_lrs, sigma=self.R_sigma[0])
            
                # BL_current_samples = np.array([])
                # for j in range(n_hrs+1):
                #     current_Roff = self.init_distribution(
                #         mu=self.Vread/self.Roff*j, sigma=self.R_sigma[1])

                #     Ion_sample, Ioff_sample = current_Ron.rvs(100000), current_Roff.rvs(100000)

                #     BL_current_samples = np.append(BL_current_samples, Ion_sample + Ioff_sample)
                # for j in range(self.S_ou+1):
                    # error_table[i, j] = np.count_nonzero(sense_output==j)/sense_output.size
            
            # Method 3:
            virtual_rram = RRAM(
                Vread=self.Vread, 
                R=self.R, R_deviation=self.R_sigma, S_ou=self.S_ou)
            inp = np.random.randint(2, size=10000000)
            weight = np.random.randint(2, size=10000000)
            virtual_rram.rram_write_binary(weight)

            BL_current = virtual_rram.Vread*inp/virtual_rram.rram_array
            BL_current_samples = BL_current.reshape(-1, self.S_ou).sum(axis=-1)
                
            sense_output = self.rram_sense_current_cim(BL_current_samples)
            correct_output = (inp*weight).reshape(-1, self.S_ou).sum(axis=-1)
            error_table = confusion_matrix(correct_output, sense_output, normalize='true')

            # np.save(file_path, error_table)
        
        self.error_table = error_table
        return error_table


    def plot_rram_cell_stats(self):
        current_Ron = self.init_distribution(
            mu=self.Vread/self.Ron, sigma=self.R_sigma[0])
        current_Roff = self.init_distribution(
            mu=self.Vread/self.Roff, sigma=self.R_sigma[1])
        Ion_sample, Ioff_sample = current_Ron.rvs(
            100000), current_Roff.rvs(100000)

        plt.figure()
        sns.distplot(Ion_sample, label='On')
        sns.distplot(Ioff_sample, label='Off')
        plt.legend()
        plt.title("RRAM read current")
        plt.xlabel('BL Current (A)')
        plt.show()

        Ron = self.init_distribution(mu=self.Ron, sigma=self.R_sigma[0])
        Roff = self.init_distribution(mu=self.Roff, sigma=self.R_sigma[1])
        Ron_sample, Roff_sample = Ron.rvs(100000), Roff.rvs(100000)

        plt.figure()
        sns.ecdfplot(Ron_sample, label='On')
        sns.ecdfplot(Roff_sample, label='Off')
        plt.legend()
        plt.title("RRAM resistance CDF")
        plt.xlabel('Resistance')
        plt.show()
        


    def plot_rram_cim_stats(self):
        plt.figure()
        for i in range(self.S_ou+1):
            n_lrs, n_hrs = i, self.S_ou-i

            current_Ron = self.init_distribution(
                mu=self.Vread/self.Ron*n_lrs, sigma=self.R_sigma[0])
            
            sample_i = np.array([])
            for j in range(n_hrs+1):
                current_Roff = self.init_distribution(
                    mu=self.Vread/self.Roff*j, sigma=self.R_sigma[1])

                Ion_sample, Ioff_sample = current_Ron.rvs(10000), current_Roff.rvs(10000)

                sample_i = np.append(sample_i, Ion_sample + Ioff_sample)

            sns.distplot(sample_i, label='WL={}'.format(i))

        for i in range(self.S_ou):
            plt.axvline(x=self.ref_current_cim[i], color='r', ls='--')

        plt.legend()
        plt.xlabel('BL Current (A)')
        plt.title("RRAM CIM Current Distribution")


    # def sim_binary_GEMM(self, A, B, map_func=None):
    #     '''
    #         Simulate the binary GEMM on RRAM
    #         A: axb, B:bxc, C:axc
    #     '''
    #     if A.shape[1] != B.shape[0]:
    #         raise Exception('Invalid input matrix dimension {} and {}!'.format(A.shape, B.shape))

    #     dim_b_segs = A.shape[1]//self.S_ou
    #     C = np.zeros([A.shape[0], dim_b_segs, B.shape[1]], dtype=float)
    #     for i in tqdm(range(A.shape[0]), desc='Simulating CIM:'):
    #         for j in range(B.shape[1]):
    #             correct_results = A[i,:] * B[:,j].T
    #             correct_results= correct_results.reshape([self.S_ou, -1]).sum(axis=0)
    #             C[i,:,j] = correct_results

    #     for i in tqdm(range(self.S_ou+1), desc='Injecting error'):
    #         idx = C==i
    #         N_sample = np.count_nonzero(idx)
    #         results_with_error = np.random.choice(self.S_ou+1, size=N_sample, p=self.bit_error_table[i,:])
    #         if map_func is not None:
    #             C[idx] = map_func(results_with_error)
    #         else:
    #             C[idx] = results_with_error
    #     return C.sum(axis=1)
