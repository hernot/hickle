import numpy as np
import pickle

class Wave(object):
    def __init__(self, start, peak, stop):
        self.__start = int(start)
        self.__peak = int(peak)
        self.__stop = int(stop)

    __slots__ = ("__start","__peak","__stop" )

    def __getstate__(self):
        return dict(
            Start = self.__start,
            Peak = self.__peak,
            Stop = self.__stop
        )

    def __setstate__(self,state):
        self.__init__(0,0,0)
        _npint64 = np.int64
        for key in {"Start","Peak","Stop"}:
            if key not in state:
                raise pickle.UnpicklingError("Key: {0:s} missing in Wave description".format(key))
            try:
                # _Wave is necessary infront of private attributes as in compiled version all attributes
                #  are private
                setattr(self,"_Wave__" + key.lower(),state[key][0] if isinstance(state[key],(list,tuple)) and len(state[key]) == 4 else state[key] )
            except Exception as msg:
                raise pickle.UnpicklingError("Key: {0:s} is not a valid int64 ({1})".format(key,msg))

    @property
    def Start(self):
        return self.__start

    @property
    def Peak(self):
        return self.__peak

    @property
    def Stop(self):
        return self.__stop

    def scale(self,factor):
        #print(self.__start,self.__peak,self.__stop,factor)
        self.__start = int( self.__start * factor + 0.5 )
        self.__peak = int( self.__peak * factor +0.5 )
        self.__stop = int( self.__stop * factor +0.5 )
        #print(self.__start,self.__peak,self.__stop,factor)

    def __repr__(self):
        return "<{3} {{'start': {0}, 'peak': {1}, 'stop': {2}, }}>".format(self.__start,self.__peak,self.__stop,self.__class__.__name__)

class Beat(object):
    Incomplete = np.int64(0) # not all parts of beat yet recognized
    Normal = np.int64(1) # normal beat with P, QRS and T wave
    Narrow = np.int64(2) # Narrow QRS
    Extra = np.int64(4) # extra systolic beat beat without p
    LargeQ = np.int64(8) # Large Q Peak inside search window
    LargeR = np.int64(16) # Large S Peak but smaller than R, peakentry + 2
    HugeS = np.int64(32) # Huge S Peak outside search window but within integration peak / 2 window larger than R
    SteepRS = np.int64(64) # Not sure if ectopic or fusion beat QRS has two sharp peaks the slope between them is more than 2.5 * the average slope of the last 6 beats
    Ventricular = np.int64(128) # beat comes from ventricle, currently only used in combination with Extra
    SignalOn = np.int64(2**57)
    NoSignal = np.int64(2**58)
    LimitsExceeded = np.int64(2**59)
    StartBeat = np.int64(2**60)
    Artefact = np.int64(2**61)
    NotDecided = np.int64(2**62)

    _annotation_manager = dict

    @classmethod
    def set_annotation_manager(cls,manager):
        cls._annotation_manager = manager

    @classmethod
    def get_annotatoin_manager(cls):
        return cls._annotation_manager

    @staticmethod
    def EncodeClass(beattype):
        if beattype == Beat.Incomplete:
            return ["Incomplete"]
        
        bitmap = np.int64(0)
        flags = []
        for key,value in Beat.__dict__.items():
            if key == "Incomplete" or type(value) != np.int64:
                continue
            if beattype & value:
                flags.append(key)
                bitmap = bitmap | value
        unnamed = beattype & np.invert(bitmap)
        if unnamed > 0:
            flags.append(unnamed.tolist())
        return flags

    @staticmethod
    def DecodeClass(flags):
        if len(flags) < 1 or flags[0] == 'Incomplete':
            return Beat.Incomplete
        beattype = 0
        bitmask = np.invert(np.int64(2**63-1) & np.invert(np.int64(255 | np.int64(31<< 57))))
        for fl in flags:
            if type(fl) == int:
                beattype = beattype | np.int64(fl & bitmask)
                continue
            if type(fl) != type('') or fl not in Beat.__dict__ or type(Beat.__dict__[fl]) != np.int64:
                raise pickle.UnpicklingError("Named flag '{0:s}' not a valid beat class or unknown encode in integer".format(fl))
            beattype = beattype | Beat.__dict__[fl]
        return beattype
    
            

    def __init__(self, pwave, qrs,minsig = np.inf,maxsig = -np.inf,beattype = Incomplete):
        self.__pwave = pwave
        self.__qrs = qrs
        self.__twave = None
        self.__tail = []
        self.__beattype = beattype
        self.__minsig = np.float64(minsig)
        self.__maxsig = np.float64(maxsig)
        self.__annotation = self.__class__._annotation_manager()

    __slots__ = ( "__pwave","__qrs","__twave","__tail","__beattype","__minsig","__maxsig","__annotation")

    def __repr__(self):
        return "<{0} {{'PWave': {1}, 'QRS': {2}, 'TWave' {3}, 'Type':{4}, 'Min': {5} 'Max': {6}, 'Annotation': {7}}}>".format(
            self.__class__.__name__,
            self.__pwave,
            self.__qrs,
            self.__twave,
            self.__beattype,
            self.__minsig,
            self.__maxsig,
            self.__annotation
        )
    def __getstate__(self):
        _np_name = np.__name__
        return {
            "PWave":self.__pwave if self.__pwave is not None else 'None',
            "QRS":self.__qrs if self.__qrs is not None else 'None',
            "TWave":self.__twave if self.__twave is not None else 'None',
            "Tail":self.__tail[1:] if len(self.__tail) > 1 else [],
            "MinAmp": self.__minsig.tolist(),
            "MaxAmp": self.__maxsig.tolist(),
            "Annotation": self.__annotation
        }
                
    def __setstate__(self,state):
        self.__init__(None,None)
        membermap = {
            "MinAmp":["__minsig",np.float64],
            "MaxAmp":["__maxsig",np.float64],
        }
        for key in {"PWave","QRS","TWave","Tail","MinAmp","MaxAmp","Annotation"}:
            if key not in state:
                raise pickle.UnpicklingError("Key: {0:s} missing in Wave description".format(key))
            try:
                if key in membermap:
                    # _Beat is necessary infront of private attributes as in compiled version all attributes
                    #  are private
                    setattr(self,"_Beat" + membermap[key][0],membermap[key][1](state[key] if state[key] != 'None' else None ))
                    continue
                # _Beat is necessary infront of private attributes as in compiled version all attributes
                #  are private
                setattr(self,"_Beat__" + key.lower(),state[key] if state[key] != "None" else None )
            except Exception as msg:
                raise pickle.UnpicklingError("Key: {0:s} is not a valid int64 ({1},{2})".format(key,str(msg),state[key]))
        self.__annotation = state['Annotation']
        if not isinstance(self.__annotation,self.__class__._annotation_manager):
            try:
                self.__annotation = self.__class__._annotation_manager(self.__annotation)
            except:
                raise pickle.UnpicklingError("Annotation must be a valid dict")
            if not isinstance(self.__annotation,self.__class__._annotation_manager):
                raise pickle.UnpicklingError("Annotation must be a valid dict")
        if not isinstance(self.__tail,list):
            if self.__tail is not None:
                raise pickle.UnpicklingError("Tail must be a list of additional beat waves")
            for tw in self.__tail:
                if not isinstance(tw, Wave):
                    raise pickle.UnpicklingError("Tail must be a list of additional beat waves non Wave object found")
        if self.__twave is not None and not isinstance(self.__twave, Wave):
            raise pickle.UnpicklingError("TWave must be None or a vaild Wave object")
        if self.__pwave is not None and not isinstance(self.__pwave, Wave):
            raise pickle.UnpicklingError("PWave must be None or a vaild Wave object")
        if self.__qrs is None or not isinstance(self.__qrs, Wave):
            raise pickle.UnpicklingError("QRS must be a vaild Wave object")
        if self.__twave is not None:
            self.__tail = [self.__twave] + self.__tail

    def __compact__(self):
        """ return None or return immediately to trun off Compacting """
        waves = np.zeros([3 + len(self.__tail[1:]),3],dtype = np.uint64)
        if self.__pwave is not None:
            waves[0,:] = self.__pwave.Start,self.__pwave.Peak,self.__pwave.Stop
        if self.__qrs is not None:
            waves[1,:] = self.__qrs.Start,self.__qrs.Peak,self.__qrs.Stop
        if self.__twave is not None:
            waves[2,:] = self.__twave.Start,self.__twave.Peak,self.__twave.Stop
        for extra_id,extra in enumerate(self.__tail[1:],3):
            waves[extra_id,:] = extra.Start,extra.Peak,extra.Stop
        minmaxamp = np.array([self.__minsig.tolist(),self.__maxsig.tolist()])
        
        return dict(
            Waves = waves,
            MinMaxSig = minmaxamp,
            Annotation = self.__annotation
        ) 

    def __expand__(self,compact):
        self.__init__(None,None)
        waves = compact['Waves']
        if any(waves[0,:] > 0):
            self.__pwave = Wave(*waves[0,:])
        if any(waves[1,:] > 0):
            self.__qrs = Wave(*waves[1,:])
        if any(waves[2,:] > 0):
            Beat.AddTWave(Wave(*waves[2,:]),self)
        for extra in waves[3:,:]:
            Beat.AddTWave(Wave(*extra),self)
        self.__minsig,self.__maxsig = compact['MinMaxSig']
        self.__annotation = compact['Annotation']
        if not isinstance(self.__annotation,self.__class__._annotation_manager):
            try:
                self.__annotation = self.__class__._annotation_manager(self.__annotation)
            except:
                raise pickle.UnpicklingError("Tail must be a list of additional beat waves non Wave object found")
            if not isinstance(self.__annotation,self.__class__._annotation_manager):
                raise pickle.UnpicklingError("Tail must be a list of additional beat waves non Wave object found")
            
    @property
    def Minval(self):
        return self.__minsig

    @Minval.setter
    def Minval(self,value):
        value = np.float64(value)
        if value < self.__minsig:
            self.__minsig = value

    @property
    def Maxval(self):
        return self.__maxsig

    @Maxval.setter
    def Maxval(self,value):
        value = value
        if value > self.__maxsig:
            self.__maxsig = value

    @property
    def Class(self):
        return self.__beattype

    @Class.setter
    def Class(self,value):
        self.__beattype = value

    @property
    def PWave(self):
        return self.__pwave

    @property
    def QRS(self):
        return self.__qrs

    @property
    def TWave(self):
        return self.__twave
    @property
    def Tail(self):
        return self.__tail[1:]

    @property
    def Start(self):
        if self.__pwave is not None:
            return self.__pwave.Start
        return self.__qrs.Start

    @property
    def Stop(self):
        if len(self.__tail) > 0:
            return self.__tail[len(self.__tail)-1].Stop
        return self.__qrs.Stop

    @property
    def Count(self):
        if self.__pwave is None:
            if self.__twave is None:
                return 1
            return len(self.__tail) + 1
        return len(self.__tail) + 2

    @property
    def Annotation(self):
        return self.__annotation

    def scale(self,factor):
        if self.__pwave is not None:
            self.__pwave.scale(factor)
        self.__qrs.scale(factor)
        for _wave in self.__tail:
            _wave.scale(factor)

    @classmethod
    def AddTWave(cls, twave, to):
        if twave is not None:
            if to.__twave is None:
                to.__twave = twave
                to.__beattype = to.__beattype | Beat.Normal
            to.__tail.append(twave)

    @classmethod
    def ResetTWaves(cls,ofbeat):
        ofbeat.__tail.clear()
        ofbeat.__twave = None

    @classmethod
    def RemoveTWave(cls,twave, frombeat):
        """ returns None if no twave is left after operation or it had never a twave"""
        if frombeat.__twave is None:
            return None
        for tw in np.arange(0,len(frombeat.__tail)):
            if frombeat.__tail[tw] == twave:
                del frombeat.__tail[tw]
                if twave == frombeat.__twave:
                    if len(frombeat.__tail) < 1:
                         frombeat.__twave = None
                    else:
                         frombeat.__twave = frombeat.__tail[0]
                return frombeat.__twave
        return None
        

    @classmethod
    def SwitchP2TWave(cls, frombeat, tobeat,relocatewave = 0):
        if ( tobeat.__twave is not None ) or ( frombeat.__pwave is None ):
            return
        relocatewave = tobeat.QRS.Peak
        newtwave = Wave(
            frombeat.__pwave.Start + relocatewave,
            frombeat.__pwave.Peak + relocatewave,
            frombeat.__pwave.Stop + relocatewave
        )
        tobeat.__twave = newtwave
        tobeat.__tail.append(newtwave)
        frombeat.__pwave = None
        frombeat.__beattype = Beat.Ventricular

    @classmethod
    def CutTailFit(cls,ofbeat,towave,relocatetail = 0):
        if ofbeat.__twave is None:
            return
        tailwave = len(ofbeat.__tail)-1
        if ofbeat.__tail[tailwave].Stop - relocatetail < towave.Start:
            return
        removetail = None
        while tailwave >= 0 and towave.Start <= ofbeat.__tail[tailwave].Stop - relocatetail:
            removetail = ofbeat.__tail[tailwave]
            del ofbeat.__tail[tailwave]
            if removetail == ofbeat.__twave:
                ofbeat.__twave = None
            tailwave -= 1
        if removetail is None:
            return
        if removetail.Peak - relocatetail >= towave.Start:
            return
        newtailwave = Wave(removetail.Start,removetail.Peak,towave.Start + relocatetail )
        if ofbeat.__twave is None:
            ofbeat.__twave = newtailwave
        ofbeat.__tail.append(newtailwave)
