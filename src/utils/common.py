import torchvision.transforms as transforms

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def split_model_name(model_name):
    if '--' in model_name:
        classifier, n_iter = model_name.split('--')
    else:
        classifier, n_iter = model_name, None
    return classifier, n_iter

def get_transformation():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
    return trans

def get_aug_transformation():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    trans = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
    return trans
