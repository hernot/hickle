import numpy as np
import Beat
import MorphometricPoint as mp

class AnnotationManager():
    def __init__(self,sequence={}):
        if isinstance(sequence,dict) and sequence:
            swapmeasure = sequence['R_pre_mml'].shape[0] > 5 + sequence['R_peak_mml'].shape[0] > 5 + sequence['R_post_mml'].shape[0] > 5
            if swapmeasure > 0:
                if swapmeasure != 3:
                    raise ValueError('corrupt Beat annotation')
                sequence['R_pre_mml'] = sequence['R_pre_mml'][:5,[0,1,3,2,4]]
                sequence['R_peak_mml'] = sequence['R_peak_mml'][:5,[0,1,3,2,4]]
                sequence['R_post_mml'] = sequence['R_post_mml'][:5,[0,1,3,2,4]]
                sequence['extra_r_mml'] = (
                    [ extra[:,[0,1,3,2,4]] for extra in sequence['extra_r_mml'][0] ],
                    [ extra[:,[0,1,3,2,4]] for extra in sequence['extra_r_mml'][1] ]
                )
        self.__dict__.update(sequence)
    
    def __setitem__(self,key,value):
        self.__dict__[key] = value

    def __getitem__(self,key):
        return self.__dict__[key]

    def __delitems__(self,key):
        del self.__dict__[key]

    def pop(self,key,default = None):
        return self.__dict__.pop()

    def __iter__(self):
        yield from self.__dict__

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self,item):
        return item in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def get(self,key,default = None):
        return self.__dict__.get(key,default)

    def __eq__(self,other):
        return self.__dict__ == other

    def __ne__(self,other):
        return self.__dict__ != other

    def popitem(self,item):
        return self.__dict__.popitem(item)
    
    def clear(self):
        self.__dict__.clear()

    def update(self,sequence):
        self.__dict__.update(sequence)

    def setdefault(self,key,default = None):
        self.__dict__.setdefault(key,default)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __compact__(self):
        try:
            if (
                self.R_peak_mml.base is not None and 
                all(
                    extra.base is self.R_peak_mml.base
                    for extra in (
                        self.Q_peak_mml,
                        *self.extra_r_mml[0],self.R_pre_mml,self.R_peak_mml,self.R_post_mml,*self.extra_r_mml[1],
                        self.S_peak_mml
                    )
                )
            ):
                delineation = self.R_peak_mml.base
            else:
                padding = -np.ones([5,2])
                delineation = np.stack([
                    np.block([self.Q_peak_mml,padding]),
                    self.R_pre_mml[:5,:],self.R_peak_mml[:5,:],self.R_post_mml[:5,:],
                    np.block([self.S_peak_mml,padding]),
                    *self.extra_r_mml[0],*self.extra_r_mml[1]
                ])
            if self.extra_r_mml[1]:
                remark = delineation[-len(self.extra_r_mml[1]):,:,:]
                if any(remark[:,0,0] != -42):
                    # mark all added post mml as post mml
                    remark[:,0,0] = -42
            if self.Q_onset_mml.base is not None and self.Q_onset_mml.base is self.S_offset_mml.base:
                onoffset = self.Q_onset_mml.base
            else:
                onoffset = np.stack([self.Q_onset_mml,self.S_offset_mml])
            if (
                self.Wave_max.base is not None and 
                all( 
                    paramset.base is self.Wave_max.base
                    for paramset in ( self.Wave_min,self.Means,self.CurvMax,self.Wave_max )
                )
            ):
                waveletinfo = self.Wave_max.base
            else:
                padd = np.zeros([1,6])
                waveletinfo = np.stack([
                    np.vstack([
                        np.block([self.Wave_min,self.Wave_max]),
                        padd
                    ]),
                    np.vstack([
                        np.block([self.CurvMax,np.zeros([5,3])]),
                        padd
                    ]),
                    self.Means
                ])
            morphometry = self.__dict__.get('QRS_Morphometry',None)
            if isinstance(morphometry,(list,tuple)):
                if len(morphometry) == 5:
                    morphometry = mp.MorphPointList( mp.MorphPointList(item) for item in morphometry )
                else:
                    morphometry = mp.MorphPointList( mp.MorphPointList() for _ in range(5))
            elif not isinstance(morphometry,mp.MorphPointList):
                morphometry = mp.MorphPointList( mp.MorphPointList() for _ in range(5))
            if isinstance(self.QR_Tail,np.ndarray) and self.QR_Tail.base is not None and isinstance(self.RS_Tail,np.ndarray) and self.RS_Tail.base is self.QR_Tail.base:
                r_tails = self.QR_Tail.base
            else:
                max_tail = max(len(self.QR_Tail),len(self.RS_Tail))
                r_tails = np.stack([
                    (np.vstack([self.QR_Tail,np.zeros([max_tail - len(self.QR_Tail),3])]) if self.QR_Tail else np.zeros([max_tail,3])),
                    (np.vstack([self.RS_Tail,np.zeros([max_tail - len(self.RS_Tail),3])]) if self.RS_Tail else np.zeros([max_tail,3]))
                ]) if max_tail > 0 else None
            if self.get('T_search_peaks',None):
                totalpeaks = len(self.T_search_peaks['s4']) + len(self.T_search_peaks['s5'])
                tpeaks = self.T_search_peaks
                starts = [0,len(self.T_search_peaks['s4'])]
                if self.get('P_search_peaks',None):
                    starts.extend((totalpeaks,totalpeaks + len(self.P_search_peaks['s5'])))
                    totalppeaks = len(self.P_search_peaks['s4']) + len(self.P_search_peaks['s5'])
                    totalpeaks += totalppeaks
                    ppeaks = self.P_search_peaks
                    kind = np.zeros(totalpeaks,dtype = np.int8)
                    points = np.zeros([totalpeaks,2],dtype=np.int16)
                    samples = np.zeros([totalpeaks,3],dtype=np.int64)
                    area = np.zeros(totalpeaks,dtype=np.float32)
                    for peakid,peak in enumerate((p for plist in (self.P_search_peaks['s4'],self.P_search_peaks['s5']) for p in plist) ,starts[2]):
                        kind[peakid] = peak[0]
                        points[peakid,:] = peak[2],peak[6]
                        samples[peakid,:] = peak[1],peak[3],peak[5]
                        area[peakid] = peak[4]
                    maxareas = np.array([self.T_search_peaks['s4_max_area'],self.T_search_peaks['s5_max_area'],self.P_search_peaks['QRS_av_peak_area']],dtype = np.float32)
                else:
                    totalppeaks = 0
                    ppeaks = ()
                    kind = np.zeros(totalppeaks,dtype = np.int8)
                    points = np.zeros([totalpeaks,2],dtype=np.int16)
                    samples = np.zeros([totalpeaks,3],dtype=np.int64)
                    area = np.zeros([totalpeaks,1],dtype=np.float32)
                    maxareas = np.array([self.T_search_peaks['s4_max_area'],self.T_search_peaks['s5_max_area'],0],dtype = np.float32)
                for peakid,peak in enumerate((p for plist in (self.T_search_peaks['s4'],self.T_search_peaks['s5']) for p in plist),0):
                    kind[peakid] = peak[0]
                    points[peakid,:] = peak[2],peak[6]
                    samples[peakid,:] = peak[1],peak[3],peak[5]
                    area[peakid] = peak[4]
            elif self.get('P_search_peaks',None):
                tpeaks = ()
                totalpeaks = totalppeaks = len(self.P_search_peaks['s4']) + len(self.P_search_peaks['s5'])
                ppeaks = self.P_search_peaks
                starts = [0,0,0,len(self.P_search_peaks['s4'])]
                kind = np.zeros(totalppeaks,dtype = np.int8)
                points = np.zeros([totalpeaks,2],dtype=np.int16)
                samples = np.zeros([totalpeaks,3],dtype=np.int64)
                area = np.zeros([totalpeaks+3,1],dtype=np.float32)
                maxareas = np.array([0,0,self.P_search_peaks['QRS_av_peak_area']],dtype = np.float32)
                for peakid,peak in enumerate((p for plist in (self.P_search_peaks['s4'],self.P_search_peaks['s5']) for p in plist) ,starts[2]):
                    kind[peakid] = peak[0]
                    points[peakid,:] = peak[2],peak[6]
                    samples[peakid,:] = peak[1],peak[3],peak[5]
                    area[peakid] = peak[4]
            else:
                t_search_peaks = {}
                totalppeaks = totalpeaks = 0
                tpeaks = ppeaks = ()
            
            ratesbase = rates = None
            for item in (self.PeakRatio,self.R_low_rates,self.R_rate_ratio):
                if not isinstance(item,(list,tuple)):
                    if ratesbase is None:
                        if item.base is not None:
                            ratesbase = item.base
                            continue
                    elif ratesbase is item.base:
                        continue
                rates = np.array(
                    [
                        [*self.PeakRatio,0],
                        [self.R_pre_rate,self.R_peak_rate,self.R_post_rate],
                        self.R_low_rates,
                        self.R_rate_ratio
                    ],
                    dtype = np.float32
                )
                break
            if rates is None:
                rates = ratesbase
                rates[1,:] = self.R_pre_rate,self.R_peak_rate,self.R_post_rate
            t_area = self.get('T-Area',None)
            t_baseline = self.get('T-Baseline',None)
            if t_area is not None and t_baseline is not None:
                totalparams = 0
                if self.Atrial_flutter is not None: 
                    totalparams = 3 + len(self.Atrial_flutter['parameters'])
                    flutterpeaks = np.array([
                        [peak.Start,peak.Peak,peak.Stop]
                        for peak in self.Atrial_flutter['peaks']
                    ])
                    wave_flags = np.zeros(totalparams,dtype = np.uint8) # 0,1,2,3 kind 4 ==Artefact 8==Spit-T
                    wave_area_and_base = np.zeros([totalparams,2],dtype = np.float32) # 0 area 1 baseline
                    for entryid,parameters in enumerate(self.Atrial_flutter['parameters'],3):
                        wave_flags[entryid] = parameters['Kind'] + ( parameters['Artefact'] >> 2 )
                        wave_area_and_base[entryid,:] = parameters['Area'],parameters['Baseline']
                else:
                    flutterpeaks = None
                if self.get('U-Wave',None) is not None:
                    if totalparams < 1:
                        totalparams = 3
                        flutterpeaks = None
                        wave_flags = np.zeros([3],dtype = np.uint8) # 0 is kind 1 is flag with 1==Artefact 2==Spit-T
                        wave_area_and_base = np.zeros([3,2],dtype = np.float32) # 0 area 1 baseline
                    parameters = self['U-Wave']
                    wave_flags[2] = parameters['Kind'] + ( parameters['Artefact'] >> 2 ) + ( parameters['Split-T'] >> 3 )
                    wave_area_and_base[2,:] = parameters['Area'],parameters['Baseline']
                else:
                    totalparams = 2
                    flutterpeaks = None
                    wave_flags = np.zeros([2],dtype = np.uint8) # 0 is kind 1 is flag with 1==Artefact 2==Spit-T
                    wave_area_and_base = np.zeros([2,2],dtype = np.float32) # 0 area 1 baseline
                if self.P_Wave_parameters is not None:
                    parameters = self.P_Wave_parameters
                    wave_flags[0] = parameters['Kind'] + ( parameters['Artefact'] >> 2 )
                    wave_area_and_base[0,:] = parameters['Area'],parameters['Baseline']
                wave_flags[1] = self['T-Kind']
                wave_area_and_base[1,:] = self['T-Area'],self['T-Baseline']
            else:
                wave_flags = None
                flutterpeaks = None
                wave_area_and_base = None
            return dict(
                Delineation = delineation,
                OnOffSet = onoffset,
                WaveletPrameters = waveletinfo,
                QRS_Morphometry = morphometry,
                R_peak_tails = r_tails,
                SearchPeaks = dict(
                    starts = starts,
                    kind = kind,
                    points = points,
                    samples = samples,
                    area = area,
                    maxareas = maxareas
                ) if totalpeaks > 0 else None,
                PeakRates = rates,
                WaveFlags = wave_flags,
                WaveAreasAndBases = wave_area_and_base,
                AtrialFlutter = flutterpeaks,
                ST_Segments = self.get('ST-Segments',None),
                QRS_pt_peak = self.QRS_pt_peak,
                Traces = np.array(self.traces,dtype = np.int8 )
            )
        except:
            raise

    def __expand__(self,compact):
        delineation = compact['Delineation']
        self.Q_peak_mml = delineation[0,:,:3]
        self.R_pre_mml = delineation[1,:,:]
        self.R_peak_mml = delineation[2,:,:]
        self.R_post_mml = delineation[3,:,:]
        self.S_peak_mml = delineation[4,:,:3]
        self.extra_r_mml = ([],[])
        for index,mml in enumerate(delineation[5:,:,:],5):
            if mml[0,0] == -42:
                self.extra_r_mml[0].extend([*delineation[5:index,:,:]])
                self.extra_r_mml[1].extend([*delineation[index:,:,:]])
                break
        if delineation.shape[0] > 5 and not self.extra_r_mml[0] and not self.extra_r_mml[1]:
            self.extra_r_mml[0].extend([*delineation[5:]])
        onoffset = compact['OnOffSet']
        self.Q_onset_mml = onoffset[0,:,:]
        self.S_offset_mml = onoffset[1,:,:]
        waveletinfo = compact['WaveletPrameters']
        self.Wave_min = waveletinfo[0,:5,:3]
        self.Wave_max = waveletinfo[0,:5,3:]
        self.CurvMax = waveletinfo[1,:5,:3]
        self.Means = waveletinfo[2,:,:]
        self.QRS_Morphometry = compact['QRS_Morphometry']
        r_tails = compact['R_peak_tails']
        if r_tails is not None:
            self.QR_Tail = r_tails[0,:r_tails.shape[1]-sum(r_tails[0,-1::-1,0]== 0),:]
            self.RS_Tail = r_tails[1,:r_tails.shape[1]-sum(r_tails[1,-1::-1,0]== 0),:]
        else:
            self.QR_Tail = []
            self.RS_Tail = []
        searchpeaks = compact['SearchPeaks']
        if searchpeaks is not None:
            starts = searchpeaks['starts']
            kind = searchpeaks['kind']
            points = searchpeaks['points']
            samples = searchpeaks['samples']
            area = searchpeaks['area']
            maxareas = searchpeaks['maxareas']
            starts.append(len(kind))
            if len(starts) > 4:
                s4 = slice(*starts[2:4])
                s5 = slice(*starts[3:5])
                self.P_search_peaks = dict(
                    s4 = tuple( tuple(item) for item in zip(
                        kind[s4],
                        samples[s4,0],points[s4,0],
                        samples[s4,1],area[s4],
                        samples[s4,2],points[s4,1]
                    ) ),
                    s5 = tuple( tuple(item) for item in zip(
                        kind[s5],
                        samples[s5,0],points[s5,0],
                        samples[s5,1],area[s5],
                        samples[s5,2],points[s5,1]
                    ) ),
                    QRS_av_peak_area = maxareas[2]
                )
            s4 = slice(*starts[:2])
            s5 = slice(*starts[1:3])
            self.T_search_peaks = dict(
                s4 = tuple( tuple(item) for item in zip(
                    kind[s4],
                    samples[s4,0],points[s4,0],
                    samples[s4,1],area[s4],
                    samples[s4,2],points[s4,1]
                ) ),
                s5 = tuple( tuple(item) for item in zip(
                    kind[s5],
                    samples[s5,0],points[s5,0],
                    samples[s5,1],area[s5],
                    samples[s5,2],points[s5,1]
                ) ),
                s4_max_area = maxareas[0],
                s5_max_area = maxareas[1]
            )
        rates = compact['PeakRates']
        self.PeakRatio = rates[0,:2]
        self.R_pre_rate,self.R_peak_rate,self.R_post_rate = rates[1,:]
        self.R_low_rates = rates[2,:]
        self.R_rate_ratio = rates[3,:]
        wave_flags = compact['WaveFlags']
        wave_area_and_base = compact['WaveAreasAndBases']
        flutterpeaks = compact['AtrialFlutter']
        if wave_flags is not None and wave_area_and_base is not None:
            if len(wave_flags) > 2 and len(wave_area_and_base) > 2:
                if flutterpeaks is not None and len(wave_flags) > 3 and len(wave_area_and_base) > 3:
                    atrial_flutter = self.Atrial_flutter = dict()
                    atrial_flutter['peaks'] = [ Beat.Wave(*item) for item in flutterpeaks ]
                    atrial_flutter['parameters'] = [
                        dict(Area=item[0],Baseline=item[1],Kind=item[2]&0x3,Artefact=bool(item[2]&0x4))
                        for item in zip(wave_area_and_base[3:,0],wave_area_and_base[3:,1],wave_flags[3:])
                    ]
                else:
                    self.Atrial_flutter = None
                if all(wave_area_and_base[2,:] > 0):
                    self['U-Wave'] = {
                        'Kind':wave_flags[2] & 0x3,
                        'Artefact': bool(wave_flags[2] & 0x4),
                        'Split-T': bool(wave_flags[2] & 8),
                        'Area': wave_area_and_base[2,0],
                        'Baseline': wave_area_and_base[2,1]
                    }
                else:
                    self['U-Wave'] = None
            else:
                self.Atrial_flutter = None
                self['U-Wave'] = None
            if all(wave_area_and_base[0,:] > 0):
                self.P_Wave_parameters = dict(
                    Kind = wave_flags[0] & 0x3,
                    Artefact = bool(wave_flags[0] & 0x4),
                    Area = wave_area_and_base[0,0],
                    Baseline = wave_area_and_base[0,1]
                )
            else:
                self.P_Wave_parameters = None
            self['T-Kind'] = wave_flags[1] & 0x3
            self['T-Area'] = wave_area_and_base[1,0]
            self['T-Baseline'] = wave_area_and_base[1,1]
        st_segments = compact.get('ST_Segments',None)
        if st_segments is not None:
            self['ST-Segments'] = st_segments
        self.QRS_pt_peak = compact['QRS_pt_peak']
        self.traces = compact['Traces'].tolist()
    
