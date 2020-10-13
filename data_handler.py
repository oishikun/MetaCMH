import h5py
import scipy.io as scio


def load_data(path, dataset):
    file = h5py.File(path)
    if dataset == 'Flickr-25K':

        query_size = 2000
        training_size = 10000
        database_size = 18015
        images = file['images'][:].astype('float')
        labels = file['LAll'][:]
        tags = file['YAll'][:]
        query_x = images[0: query_size]
        train_x = images[query_size: training_size + query_size]
        retrieval_x = images[query_size: query_size + database_size]

        query_y = tags[0: query_size]
        train_y = tags[query_size: training_size + query_size]
        retrieval_y = tags[query_size: query_size + database_size]

        query_L = labels[0: query_size]
        train_L = labels[query_size: training_size + query_size]
        retrieval_L = labels[query_size: query_size + database_size]

    else:
        train_L = file['train_L'][:]
        train_L = train_L.transpose(1, 0)
        print('train_L',train_L.shape)
        query_L = file['test_L'][:]
        query_L = query_L.transpose(1, 0)
        # print(query_L.shape)
        retrieval_L = file['retrieval_L'][:]
        retrieval_L = retrieval_L.transpose(1, 0)
        # print(retrieval_L.shape)
        train_x = file['train_x'][:].astype('float')
        train_x = train_x.transpose(3, 0, 1, 2)
        # print('train_x', train_x.shape)
        query_x = file['test_x'][:].astype('float')
        query_x = query_x.transpose(3, 0, 1, 2)
        # print(query_x.shape)
        retrieval_x = file['retrieval_x'][:].astype('float')
        retrieval_x = retrieval_x.transpose(3, 0, 1, 2)
        # print(retrieval_x.shape)
        train_y = file['train_y'][:]
        train_y = train_y.transpose(1, 0)
        # print('train_y',train_y.shape)
        query_y = file['test_y'][:]
        query_y = query_y.transpose(1, 0)
        # print(query_y.shape)
        retrieval_y = file['retrieval_y'][:]
        retrieval_y = retrieval_y.transpose(1, 0)
        # print(retrieval_y.shape)

    file.close()
    return train_L, query_L, retrieval_L, train_x, query_x, retrieval_x, train_y, query_y, retrieval_y


def load_pretrain_model(path):
    return scio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)