import sys
sys.path.insert(0, '/home/cacof1/Software/DigitalPathologyPreprocessing/')
from Dataloader.Dataloader import *
import toml
import copy
import torch
from QA.StainNormalization import ColourNorm
from concurrent.futures import ThreadPoolExecutor
import concurrent
from torchvision import transforms
from torchvision import transforms
# pd.set_option('display.max_rows', None)

def convert_dict_to_list_of_lists(updated_kv):
    # converts an ordered dict to a list of lists, so updated_kv is compatible with omero functions... bit annoying
    kv_list = []
    for k, vset in updated_kv.items():
        for v in vset:
            kv_list.append([k, v])
    return kv_list

def get_subject_info(config, row, idx):
    row = pd.DataFrame(
        {'id_omero': [row.id_omero], 'id_external': [row.id_external], 'Size': [row.Size], 'SVS_PATH': [row.SVS_PATH],
         'OwnerID': [row.OwnerID], 'GroupID': [row.GroupID], 'index': [row.index]})
    print('pre synchro', idx)
    print(row)
    SynchronizeSVS(config, row)
    wsi = openslide.open_slide(row['SVS_PATH'].iloc[0])
    
    # Extract HE stain vectors from 1k randomly selected 256x256 tiles of the WSI.
    vis = 0
    tilesize = (256, 256)
    n_tiles_for_HE_test = 1000
    max_no = np.floor(np.array(wsi.dimensions) / np.array(tilesize)).astype(int)
    RX = tuple(np.random.randint(0, high=max_no[0] - 1, size=n_tiles_for_HE_test) * tilesize[0])
    RY = tuple(np.random.randint(0, high=max_no[1] - 1, size=n_tiles_for_HE_test) * tilesize[1])
    img = np.transpose(np.concatenate([np.array(wsi.read_region(start, vis, tilesize).convert("RGB"), dtype=np.float32) / 255. for start in zip(RX, RY)], axis=0), (2, 0, 1))
    img = torch.from_numpy(img)

    HE, maxC = ColourNorm.Macenko().find_HE(img, get_maxC=True)
    HE, maxC = np.array(HE), np.array(maxC)
    print(row,HE, maxC)
    H_str = np.array2string(HE[:, 0], separator=',')
    E_str = np.array2string(HE[:, 1], separator=',')
    maxC_H = np.array2string(maxC[0])
    maxC_E = np.array2string(maxC[1])
    # Upload your KV Pair here
    with connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw']) as conn:
        conn.SERVICE_OPTS.setOmeroGroup(str(row['GroupID'].iloc[0]))
        # conn.SERVICE_OPTS.setOmeroUser(str(row['GroupID'].iloc[0]))
    
        image = conn.getObject("Image", row['id_omero'].iloc[0])
        existing_kv = get_existing_map_annotations(image)
        updated_kv = copy.deepcopy(existing_kv)
        
        # update the k/v structure with new fields that you have requested.
        updated_kv['H_stainvec'] = set()
        updated_kv['E_stainvec'] = set()
        updated_kv['H_maxC'] = set()
        updated_kv['E_maxC'] = set()
        updated_kv['H_stainvec'].add(H_str)
        updated_kv['E_stainvec'].add(E_str)
        updated_kv['H_maxC'].add(maxC_H)
        updated_kv['E_maxC'].add(maxC_E)

        # Update on omero if there is a change in the kv pairs, otherwise skip
        if existing_kv != updated_kv:
            remove_map_annotations(conn, image)
            map_ann = omero.gateway.MapAnnotationWrapper(conn)
            namespace = omero.constants.metadata.NSCLIENTMAPANNOTATION
            map_ann.setNs(namespace)
            map_ann.setValue(convert_dict_to_list_of_lists(updated_kv))
            map_ann.save()
            print("Annotation {} created for ID {}.".format(map_ann.id, row['id_omero'].iloc[0]))
            image.linkAnnotation(map_ann)
            
    
    os.remove(row['SVS_PATH'].iloc[0])
    
    return row['id_omero'].iloc[0],HE,maxC


config = toml.load(sys.argv[1])

# 1. Download all relevant files
SVS_dataset = QueryImage(config)

## Filter to keep just the files that are svs and H&E
SVS_dataset = SVS_dataset[SVS_dataset['id_external'].str.contains("\[0\]")]
SVS_dataset = SVS_dataset[~SVS_dataset['id_external'].str.contains("pHH3")]
SVS_dataset = SVS_dataset[SVS_dataset['GroupID'] == 55]
SVS_dataset = SVS_dataset.drop_duplicates()
SVS_dataset = SVS_dataset.reset_index(level=0)
print(SVS_dataset)

#SVS_dataset = SVS_dataset.head(n=5)
## For testing
#for idx, row in SVS_dataset.iterrows():
#    print(idx, row)
#    HE = get_subject_info(config, row, idx)

#To do In Parallel
data = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    jobs = {executor.submit(get_subject_info, config, row, idx) for idx, row in SVS_dataset.iterrows()}
    executor.shutdown(wait=True)
    #for job in concurrent.futures.as_completed(jobs):
    #    print(job,dir(job))
    #    id_omero, HE, max_C = job.result()
    #    data[id_omero]= {"HE":HE, "max_C":max_C}
        
#        del jobs[job] 
#np.save("d1.npy", data)
