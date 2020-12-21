import requests, zipfile, os
from tqdm import tqdm
import shutil
import glob

def main():
    url = "https://rutgers.box.com/shared/static/y9wi8ic7bshe2nn63prj9vsea7wibd4x.zip"

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get('content-length', 0))
    block_size = 2**20 # Mebibyte
    t=tqdm(total=total_size, unit='MiB', unit_scale=True)

    with open('state_dicts.zip', 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception('Error, something went wrong')
        
    print('Download successful. Unzipping file.')
    path_to_zip_file = os.path.join(os.getcwd(), 'state_dicts.zip')
    directory_to_extract_to = 'cifar10/models/'
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print('Unzip file successful!')

    dst = 'res/cifar10/models'
    os.makedirs(dst, exist_ok=True)
    os.remove(path_to_zip_file)
    src_paths = glob.glob('temp/state_dicts/*.pt')
    for src in src_paths:
        shutil.move(src, dst)
    shutil.rmtree('temp/')
        
if __name__ == '__main__':
    main()