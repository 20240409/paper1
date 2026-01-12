import os
import glob
import numpy as np
import scipy.io as sio


if __name__ == "__main__":
    save_path = './data/'

    data_path = "Your SEEDVIG dataset folder path\EEG_Feature_5Bands"# path to the raw SEED dataset
    data_names = glob.glob(os.path.join(data_path, '*.mat'))
    data_names.sort()


    label_path = "Your SEEDVIG dataset folder path\perclos_labels"
    label_names=glob.glob(os.path.join(label_path,"*.mat"))
    label_names.sort()

    os.makedirs(save_path, exist_ok=True)

    for sub in range(23):
        data_path = data_names[sub]
        T_D = sio.loadmat(data_path)
        temp_data=T_D['de_LDS']

        label_path=label_names[sub]
        T_L=sio.loadmat(label_path)
        temp_label=T_L['perclos']

        sio.savemat(os.path.join(save_path, 'DE_' + str(sub) + '.mat'), {'DE_feature': np.array(temp_data)}) # save the features
        sio.savemat(os.path.join(save_path, 'label_'+str(sub)+'.mat'), {'de_labels': np.array(temp_label)}) # save the labels
