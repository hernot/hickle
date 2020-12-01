import numpy as np
import os.path
import Beat
import MorphometricPoint
import AnnotationManager
import time
import hickle
import pickle

Beat.Beat.set_annotation_manager(AnnotationManager.AnnotationManager)

#filename = "PanTomkins-incartdbI1120200819154408.537457.h5"
#filename = "PanTomkins-incartdbI0120201104170545.097297.h5"
#filename = "PanTomkins-incartdbI1720201104213032.408468.h5"
filename = "PanTomkins-incartdbI7520201106160606.618832.h5"
datapath = "../dev_check"
#dataset = np.load(os.path.join(datapath,filename),allow_pickle=True)
starttime = time.perf_counter()
filepath = os.path.join(datapath,filename)
dataset = hickle.load(filepath)
print("kickle.load lasted {} s".format(time.perf_counter() - starttime))
print(*dataset.keys())

starttime = time.perf_counter()
beats = dataset['beats']
beat = beats[2].__compact__()
annotation = beat['Annotation'].__compact__()
morphometry = annotation['QRS_Morphometry'].__compact__()
morphometry2 = MorphometricPoint.MorphPointList()
morphometry2.__expand__(morphometry)
annotation2 = AnnotationManager.AnnotationManager()
annotation2.__expand__(annotation)
annotation2.QRS_Morphometry = morphometry
compactbeat = dict(beat)
compactbeat['Annotation'] = annotation2
beat2 = Beat.Beat(None,None)
beat2.__expand__(compactbeat)
annotation3 = annotation2.__compact__()

print("load lasted {} s".format(time.perf_counter() - starttime))
fid = open(filepath.replace(".h5",".pkl"),"wb")
pickle.dump(dataset,fid)
fid.close()
#dataset.close()
#fid = open(filepath.replace(".h5",".pkl"),"rb")
#dataset2 = pickle.load(fid)
#assert dataset2[beats]== dataset.beats
#fid.close()



#hickle.dump(newbeat,"hickle_beat_test.h5")
#hicklednewbeat = hickle.load("hickle_beat_test.h5")
a=1
