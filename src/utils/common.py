
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