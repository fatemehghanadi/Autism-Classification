# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#%%
from nilearn import datasets
from imports import preprocess_data as reader
import os
import shutil
from imports.utils import arg_parse
from config import get_cfg_defaults
import numpy as np
import pywt
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# from vmdpy import VMD


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    pipeline = cfg.DATASET.PIPELINE
    atlas =cfg.DATASET.ATLAS
    connectivity = cfg.METHOD.CONNECTIVITY
    download = cfg.DATASET.DOWNLOAD

    root_dir = cfg.DATASET.ROOT
    data_folder = os.path.join(root_dir, cfg.DATASET.BASE_DIR)
    waveletFlage=cfg.WAV
    

    # pipeline = "cpac"#cfg.DATASET.PIPELINE
    # atlas ="cc200" #cfg.DATASET.ATLAS
    # connectivity = "TPE"#cfg.METHOD.CONNECTIVITY
    # download = True#cfg.DATASET.DOWNLOAD
    # root_dir = "./data"#cfg.DATASET.ROOT
    # data_folder = os.path.join(root_dir, 'ABIDE_pcp/cpac/filt_noglobal/')

    # Files to fetch
    files = ['rois_' + atlas]

    # # Download database files
    phenotype_file = os.path.join(root_dir, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    # if download or not os.path.exists(phenotype_file):
    #     datasets.fetch_abide_pcp(data_dir=root_dir, pipeline=pipeline, band_pass_filtering=True,
    #                              global_signal_regression=False, derivatives=files, quality_checked=cfg.DATASET.QC)

    phenotype_df = reader.get_phenotype(phenotype_file)

    subject_ids = []
    # # Create a folder for each subject
    for i in phenotype_df.index:
        sub_id = phenotype_df.loc[i, "SUB_ID"]

        subject_folder = os.path.join(data_folder, "%s" % sub_id)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)
        for fl in files:
            fname = "%s_%s.1D" % (phenotype_df.loc[i, "FILE_ID"], fl)
            data_file = os.path.join(data_folder, fname)
            if os.path.exists(data_file) or os.path.exists(os.path.join(subject_folder, fname)):
                subject_ids.append(sub_id)
                if not os.path.exists(os.path.join(subject_folder, fname)):
                    shutil.move(data_file, subject_folder)

    sub_id_fpath = os.path.join(data_folder, "subject_ids.txt")
    if not os.path.exists(sub_id_fpath):
        f = open(sub_id_fpath, "w")
        for sub_id_ in subject_ids:
            f.write("%s\n" % sub_id_)
        f.close()
    else:
        subject_ids = reader.get_ids(data_folder)
        subject_ids = subject_ids.tolist()

    time_series = reader.get_timeseries(subject_ids, atlas, data_folder)
    # print(len(time_series))#1035
    # print(len(time_series[0]))#196
    # print(len(time_series[0][0]))#200
    
    # Save the array of arrays
    # print(time_series[7].shape)
    # X_padded = pad_sequences(time_series, dtype='float32', padding='post', truncating='post', value=0.0)
    # print(X_padded[7].shape)

    
    
    #### 
    # waveletFlage=True
    if waveletFlage:
        wavelet = 'db20'  # Example wavelet
        level = 3     # Example decomposition level
        waveltsDatas=[]
        for i in range(0,len(time_series)):
            waveletData=[]
            data=time_series[i].T
            # print(data.shape)
            for r in range(0,200):
                coeffs = pywt.wavedec(data[r], wavelet, level=level)
                # print(len(coeffs))
                concatenated_coeffs = np.concatenate(coeffs)
                waveletData.append(concatenated_coeffs)
                # print(len(waveletData))
                # print(len(waveletData[0])) 
                # print(len(waveletData[0][0])) 
                # break
            waveletData=np.array(waveletData)
            waveltsDatas.append(np.array(waveletData.T))
            # print('------------------wav---------------------')
            # print(len(waveltsDatas))#1035
            # print(len(waveltsDatas[0])) #311
            # print(len(waveltsDatas[0][0])) #200

        # pcc_matrixes=[]
        # new_datas=[]
        # for d in waveltsDatas:
        #     pcc_matrix = np.corrcoef(d, rowvar=True)
        #     pcc_matrixes.append(pcc_matrix)
        #     new_data = np.dot(pcc_matrix, d)
        #     new_datas.append(new_data)
        # print(len(new_datas))#1035
        # print(len(new_datas[0]))#311
        # print(len(new_datas[0][0])) #200
    
    
    # VMD --------------------------------------------------------------------        
    
    # vmdflag=True
    # if vmdflag:
    #     print('VMD processing')
    #     alpha = 10     # moderate bandwidth constraint  
    #     tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    #     K = 3             # 3 modes  
    #     DC = 0            # no DC part imposed  
    #     init = 1           # initialize omegas uniformly  
    #     tol = 1e-6 

    #     vmddatas=[]
    #     for d in time_series:
    #         dt=d.T
    #         vmddata=[]
    #         for r in range(0,200):
    #             signal_data = np.array(dt[r]) 
    #             u, u_hat, omega = VMD(signal_data, alpha, tau, K, DC, init, tol)
    #             u_hat=u_hat.T
    #             # print(u_hat.shape)
    #             t=int(u_hat.shape[1]/2)
                
    #             a=u_hat[0][0:t]
                
    #             for i in range(1,K):
    #                 b=u_hat[i][0:t]
    #                 a=np.concatenate((a, b), axis=None)
    #             # print(a.shape)
    #             vmddata.append(a)
    #         vmddata=np.array(vmddata)
    #         vmddatas.append(vmddata.T)
            
            # print('------------------VMD---------------------')
            # print(a.shape)
            
            # print(len(vmddatas))
            # print(len(vmddatas[0])) #time
            # print(len(vmddatas[0][0])) #200
            





    
    # Fixing random state for reproducibility
    # np.random.seed(19680801)

    # dt = 0.01
    # t = np.arange(0, 30, dt)
    # nse1 = np.random.randn(len(t))                 # white noise 1
    # nse2 = np.random.randn(len(t))                 # white noise 2

    # Two signals with a coherent part at 10 Hz and a random part
    # s1 = np.sin(2 * np.pi * 10 * t) + nse1
    # s2 = np.sin(2 * np.pi * 10 * t) + nse2
    # print(waveltsDatas[90].shape)
    # s=waveltsDatas[90]
    # plt.plot(s)
    # plt.xlabel('')
    # plt.ylabel('Wavelet Transformed')
    # plt.grid(True)
    # plt.savefig('./Wavletsignal.pdf')
    # plt.show()
        
        
    #############################
            
    # Compute and save connectivity matrices
    if connectivity in ["correlation", 'partial correlation', 'covariance', 'tangent', "TPE"] and waveletFlage==False: #### 
        reader.subject_connectivity(time_series, atlas, connectivity, save=True, out_path=cfg.OUTPUT.OUT_PATH)
    if connectivity in ["correlation", 'partial correlation', 'covariance', 'tangent', "TPE"] and waveletFlage==True: #### 
        reader.subject_connectivity(waveltsDatas, atlas, connectivity, save=True, out_path='./wavdotpcc')
    # if connectivity in ["correlation", 'partial correlation', 'covariance', 'tangent', "TPE"] and vmdflag==True: ####
    #     reader.subject_connectivity(vmddatas, atlas, connectivity, save=True, out_path='./vmdpcc')



if __name__ == '__main__':
    main()




# %%
