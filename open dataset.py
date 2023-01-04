import glob
import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_list = glob.glob("data/cifar-10-batches-py/data_batch_1")


for l in train_list:
    l_dict = unpickle(l)
    print(l_dict.keys())

for img_idx, img_data in enumerate(l_dict[b'data']):
    # print(img_idx)
    # print(img_data)
    img_label = l_dict[b'labels'][img_idx]
    img_name = l_dict[b'filenames'][img_idx]

    # img_label_name = label_name[img_label]
    img_data = np.reshape(img_data, [3, 32, 32])
    img_data = np.transpose(img_data, (1, 2, 0))

    cv2.imshow("im_data", cv2.resize(img_data, (200, 200)))
    cv2.waitKey(0)


