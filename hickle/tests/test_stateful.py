import glob
import warnings
import hickle as hkl
import h5py

import numpy as np
import os.path
import sys
import hickle.tests.classes.Beat 
hickle.tests.classes.Beat.__name__ = 'Beat'
hickle.tests.classes.Beat.Beat.__module__ = 'Beat'
hickle.tests.classes.Beat.Wave.__module__ = 'Beat'
sys.modules['Beat'] = hickle.tests.classes.Beat
import hickle.tests.classes.MorphometricPoint
hickle.tests.classes.MorphometricPoint.__name__ = 'MorphometricPoint'
hickle.tests.classes.MorphometricPoint.MorphPoint.__module__ = 'MorphometricPoint'
hickle.tests.classes.MorphometricPoint.MorphPointList.__module__ = 'MorphometricPoint'
sys.modules['MorphometricPoint'] = hickle.tests.classes.MorphometricPoint
import hickle.tests.classes.AnnotationManager
hickle.tests.classes.AnnotationManager.__name__ = 'AnnotationManager'
hickle.tests.classes.AnnotationManager.AnnotationManager.__module__ = 'AnnotationManager'
sys.modules['AnnotationManager'] = hickle.tests.classes.AnnotationManager
hickle.tests.classes.Beat.Beat.set_annotation_manager(hickle.tests.classes.AnnotationManager.AnnotationManager)

import time
import pickle
filename = r"PanTomkins-incartdbI1120200819154408.537457.pkl"
datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev_check')

# the following is required as package name of with_state is hickle
# and load_loader refuses load any loader module for classes defined inside
# hickle package exempt when defined within load_*.py loaders modules.
# That has to be done by hickle sub modules directly using register_class function
pickle_dumps = pickle.dumps
pickle_loads = pickle.loads

#datapath = "dev_check"
#dataset = np.load(os.path.join(datapath,filename),allow_pickle=True)
def test_write_complex_hickle():
    import Beat, MorphometricPoint,AnnotationManager
    testfile = glob.glob(os.path.join(datapath,filename))[0]
    print(testfile)
    fid = open(testfile,"rb")
    starttime = time.perf_counter()
    dataset = pickle.load(fid)
    print("pickle.load lasted {} s".format(time.perf_counter() - starttime))
    fid.close()
    print(*dataset.keys())


    starttime = time.perf_counter()
    beats = dataset['beats']
    print("load lasted {} s".format(time.perf_counter() - starttime))
    dumpfile = os.path.join(datapath,filename.replace(".pkl","-hk.h5"))
    print(dumpfile)
    #beats = dataset['beats'] = beats[:50]
    hkl.enable_compact_expand(Beat.Beat,MorphometricPoint.MorphPointList,AnnotationManager.AnnotationManager)
    starttime = time.perf_counter()
    hkl.dump(dataset,dumpfile,compression="gzip",shuffle=True)
    print("hickle.dump (compact expand enabled) lasted {} s".format(time.perf_counter() - starttime))
    starttime = time.perf_counter()
    dataset2 = hkl.load(dumpfile)
    print("hickle.load (compact expand enabled) lasted {} s".format(time.perf_counter() - starttime))
    #print(*dataset2.keys())
    starttime = time.perf_counter()
    beats2 = dataset2['beats']
    print("load lasted {} s".format(time.perf_counter() - starttime))
    print(len(beats2),len(beats2) == len(beats))
    #assert beats2 == dataset["beats"][0]
    hkl.disable_compact_expand(Beat.Beat,MorphometricPoint.MorphPointList,AnnotationManager.AnnotationManager)
    dumpfile = os.path.join(datapath,filename + ".h5")
    starttime = time.perf_counter()
    hkl.dump(dataset,dumpfile,compression="gzip",shuffle=True)
    print("hickle.dump lasted {} s".format(time.perf_counter() - starttime))
    starttime = time.perf_counter()
    dataset2 = hkl.load(dumpfile)
    print("hickle.load lasted {} s".format(time.perf_counter() - starttime))
    #print(*dataset2.keys())
    starttime = time.perf_counter()
    beats2 = dataset2['beats']
    print("load lasted {} s".format(time.perf_counter() - starttime))
    print(len(beats2),len(beats2) == len(beats))
    #assert beats2 == dataset["beats"][0]
    

    #dataset.close()

if __name__ == '__main__':
    test_write_complex_hickle()

