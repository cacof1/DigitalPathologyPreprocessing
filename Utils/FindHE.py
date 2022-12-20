import sys

sys.path.insert(0, '/home/cacof1/Software/DigitalPathologyPreprocessing/')

from Dataloader.Dataloader import *
import toml
import copy
from QA.StainNormalization import ColourNorm
from concurrent.futures import ThreadPoolExecutor
import concurrent


# pd.set_option('display.max_rows', None)

def convert_dict_to_list_of_lists(updated_kv):
    # converts an ordered dict to a list of lists, so updated_kv is compatible with omero functions... bit annoying
    kv_list = []
    for k, vset in updated_kv.items():
        for v in vset:
            kv_list.append([k, v])
    return kv_list


def get_subject_info(config, row):
    row = pd.DataFrame(
        {'id_omero': [row.id_omero], 'id_external': [row.id_external], 'Size': [row.Size], 'SVS_PATH': [row.SVS_PATH],
         'OwnerID': [row.OwnerID], 'GroupID': [row.GroupID], 'index': [row.index]})
    print('pre synchro')
    SynchronizeSVS(config, row)
    wsi = openslide.open_slide(row['SVS_PATH'].iloc[0])
    print(row)

    # Extract HE stain vectors from 10k randomly selected 256x256 tiles of the WSI.
    vis = 0
    tilesize = (256, 256)
    n_tiles_for_HE_test = 100
    max_no = np.floor(np.array(wsi.dimensions) / np.array(tilesize)).astype(int)
    RX = tuple(np.random.randint(0, high=max_no[0] - 1, size=n_tiles_for_HE_test) * tilesize[0])
    RY = tuple(np.random.randint(0, high=max_no[1] - 1, size=n_tiles_for_HE_test) * tilesize[1])
    img = np.transpose(np.concatenate([np.array(wsi.read_region(start, vis, tilesize).convert("RGB"), dtype=np.float32) / 255. for start in zip(RX, RY)], axis=0), (2, 0, 1))
                                       
    HE, maxC = ColourNorm.Macenko().find_HE(img, get_maxC=True)
    HE, maxC = np.array(HE), np.array(maxC)
    del img
    H_str = np.array2string(HE[:, 0], separator=',')
    E_str = np.array2string(HE[:, 1], separator=',')
    maxC_H = np.array2string(maxC[0])
    maxC_E = np.array2string(maxC[1])

    os.remove(row['SVS_PATH'].iloc[0])
    
    return row['id_omero'].iloc[0],HE,maxC


config = toml.load(sys.argv[1])
# config = toml.load('../ConfigDefault.ini')

# 1. Download all relevant files
SVS_dataset = QueryImage(config)

## Filter to keep just the files that are svs and H&E
SVS_dataset = SVS_dataset[SVS_dataset['id_external'].str.contains("\[0\]")]
SVS_dataset = SVS_dataset[~SVS_dataset['id_external'].str.contains("pHH3")]
SVS_dataset = SVS_dataset[SVS_dataset['GroupID'] == 55]
SVS_dataset = SVS_dataset.drop_duplicates()
SVS_dataset = SVS_dataset.reset_index(level=0)
print(SVS_dataset)

SVS_dataset = SVS_dataset.head(n=5)
# For testing
# for idx, row in SVS_dataset.iterrows():
#     HE = get_subject_info(config, row)
#     print(HE)

# To do In Parallel
data = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    jobs = {executor.submit(get_subject_info, config, row) for idx, row in SVS_dataset.iterrows()}
    executor.shutdown(wait=True)
    for job in concurrent.futures.as_completed(jobs):
        print(job,dir(job))
        id_omero, HE, max_C = job.result()
        data[id_omero]= {"HE":HE, "max_C":max_C}

        del jobs[job] 
np.save("d1.npy", data)
