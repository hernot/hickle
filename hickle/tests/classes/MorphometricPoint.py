import numpy as np
import pickle
import Beat as beatandwave

Beat = beatandwave.Beat
Wave = beatandwave.Wave


_Kind = dict(
    Unknown = 0,
    SigMax = 1,
    SigMin = 2,
    SigPeak = 3,
    WaveMax = 4,
    WaveMin = 8,
    WavePeak = 12,
    GradMax = 16,
    GradMin = 32,
    GradPeak = 48,
    CurvMax = 64,
    CurvMin = 128,
    CurvPeak = 192
)

class MorphPointList(list):
            
    def __compact__(self,samples=None,kind = None,values = None):
        if len(self) == 5 and all( isinstance(item,MorphPointList) for item in self):
            scalestart = [0] * 5
            scalestart[1] = scalestart[0] + len(self[0])
            scalestart[2] = scalestart[1] + len(self[1])
            scalestart[3] = scalestart[2] + len(self[2])
            scalestart[4] = scalestart[3] + len(self[3])
            totallen = scalestart[4] + len(self[4])
            samples = np.zeros(totallen,dtype = np.int64)
            kind = np.zeros(totallen,dtype = np.uint8)
            values = np.zeros([totallen,4],dtype = np.float32)
            # ignore return values as recursive call stores points to provided numpy.ndarrays
            self[0].__compact__(samples[:scalestart[1]],kind[:scalestart[1]],values[:scalestart[1],:])
            self[1].__compact__(samples[scalestart[1]:scalestart[2]],kind[scalestart[1]:scalestart[2]],values[scalestart[1]:scalestart[2],:])
            self[2].__compact__(samples[scalestart[2]:scalestart[3]],kind[scalestart[2]:scalestart[3]],values[scalestart[2]:scalestart[3],:])
            self[3].__compact__(samples[scalestart[3]:scalestart[4]],kind[scalestart[3]:scalestart[4]],values[scalestart[3]:scalestart[4],:])
            self[4].__compact__(samples[scalestart[4]:],kind[scalestart[4]:],values[scalestart[4]:,:])
            return dict(
                samples = samples,
                kind = kind,
                values = values,
                start = scalestart
            )
        bigpack = (samples is not None) + (kind is not None) + (values is not None)
        if bigpack == 0:
            samples = np.zeros(len(self),dtype=np.int64)
            kind = np.zeros(len(self),dtype=np.uint8),
            values = np.zeros([len(self),4],dtype=np.float64)
        elif bigpack != 3:
            raise ValueError('samples kind and values must either be None or appropriate numpy.ndarrays')
        for index,item in enumerate(self):
            samples[index] = item.sample
            kind[index] = item.kind
            values[index,:] = (item.value,item.wavelet,item.gradiend,item.curvature)
        return dict(
            samples = samples,
            kind = kind,
            values = values
        )

    def __expand__(self,compact):
        if isinstance(compact,dict):
            samples = compact['samples']
            kind = compact['kind']
            values = compact['values']
            start = compact.get('start')
        else:
            samples,kind,values = compact
            start = None
        if samples.ndim != 1 or kind.ndim != 1 or values.ndim != 2:
            raise ValueError("'compact' does not represent compacted MorphPoint list")
        if samples.size != kind.size or (samples.size,4) != values.shape:
            raise ValueError("'compact' does not represent compacted MorphPoint list")
        if start is None:
            for index in range(samples.size):
                self.append(MorphPoint(samples[index],*values[index,:],kind[index]))
            return
        if len(start) != 5 or np.any(np.diff(start) < 0) or start[-1] > samples.size:
            raise ValueError("'compact' does not represent compacted MorphPoint lists for all five scales")
        self.extend(MorphPointList() for _ in range(5))
        self[0].__expand__((samples[:start[1]],kind[:start[1]],values[:start[1],:]))
        self[1].__expand__((samples[start[1]:start[2]],kind[start[1]:start[2]],values[start[1]:start[2],:]))
        self[2].__expand__((samples[start[2]:start[3]],kind[start[2]:start[3]],values[start[2]:start[3],:]))
        self[3].__expand__((samples[start[3]:start[4]],kind[start[3]:start[4]],values[start[3]:start[4],:]))
        self[4].__expand__((samples[start[4]:],kind[start[4]:],values[start[4]:,:]))
        
class MorphPoint(object):
    def __getstate__(self):
        unknown = _Kind['Unknown']
        return {
            'Sample':self.sample.tolist() if type(self.sample).__module__ == np.__name__ else self.sample,
            'Kind':[ _key for _key in ["SigMax","SigMin","WaveMin","WaveMax","GradMax","GradMin","CurvMin","CurvMax","Unknown"] if self.kind & _Kind[_key] != 0 or ( self.kind == unknown and _key == "Unknown" )],
            'Value':self.value.tolist() if type(self.value).__module__ == np.__name__ else self.value,
            'Wavelet':self.wavelet.tolist() if type(self.wavelet).__module__ == np.__name__ else self.wavelet,
            'Gradiend':self.gradiend.tolist() if type(self.gradiend).__module__ == np.__name__ else self.gradiend,
            'Curvature':self.curvature.tolist() if type(self.curvature).__module__ == np.__name__ else self.curvature,
        }
    def __setstate__(self,state):
        self.__init__(None,None,None,None,None,None)
        self.__dict__.update({
            _key.lower() : _val if _key != "Kind" or not isinstance(_val,(list,tuple))else sum(
                _Kind[_flag] for _flag in _val 
            )
            for _key,_val in state.items() if _key in ["Sample","Kind","Value","Wavelet","Gradiend","Curvature"] and (
                 ( _key == "Kind" and type(_val) in [list,tuple] and len(_val) > 0 ) or
                np.isscalar(_val)
            )
        })
        _missing = tuple((_key for _key in ["Sample","Kind","Value","Wavelet","Gradiend","Curvature"] if self.__dict__[_key.lower()] is None))
        if len(_missing) > 1:
            raise pickle.UnpicklingError("Key(s): {0:s} missing in Morphometric point description or value not recognized".format(",".join(_missing)))

    def __repr__(self):
        return "{0}({1},{2},{3},{4},{5},{6})".format(
            self.__class__.__name__,
            self.sample,
            self.value,
            self.wavelet,
            self.gradiend,
            self.curvature,
            self.kind
        )
            
    def __init__(self,sample,value = None,wavelet = None,gradiend = None ,curvature = None,kind = 0):
        self.sample = sample
        self.value = value
        self.wavelet = wavelet
        self.gradiend = gradiend
        self.curvature = curvature
        self.kind = kind

    def __eq__(self,other):
        return self.sample == ( other.sample if isinstance(other,self.__class__) else other )

    def __ne__(self,other):
        return self.sample != ( other.sample if isinstance(other,self.__class__) else other )

    def __lt__(self, other):
        return self.sample < ( other.sample if isinstance(other,self.__class__) else other )

    def __le__(self, other):
        return self.sample <= ( other.sample if isinstance(other,self.__class__) else other )

    def __gt__(self, other):
        return self.sample > ( other.sample if isinstance(other,self.__class__) else other )

    def __ge__(self, other):
        return self.sample >= ( other.sample if isinstance(other,self.__class__) else other )

for kind,val in _Kind.items():
    setattr(MorphPoint,kind,val)

