import warnings


class DefaultConfig(object):

    # data parameters
    # data_path = './data/FLICKR-25K.mat'
    # data_set = 'Flickr-25K_LT_1'
    data_path = '../data/Flickr_LT_splited_5.mat'
    data_set = 'Flickr-LT_5'
    pretrain_model_path = './data/imagenet-vgg-f.mat'
    if data_set == 'Flickr-25K' or data_set == 'Flickr-LT' or data_set == 'Flickr-LT_5':
        load_txt_path = './data/text_model_flickr.pth'
    else:
        load_txt_path = './data/text_model.pth'
    batch_size = 128

    # hyper-parameters
    max_epoch = 100
    gamma = 1
    eta = 1
    bit = 64  # final binary code length
    base_lr = 10 ** (-1.5)
    memory_lr = 10 ** (-1.5)  # initial learning rate

    use_gpu = True

    valid = False

    print_freq = 2  # print info every N epoch

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
