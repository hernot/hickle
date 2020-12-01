
def create_function_dataset(py_obj,h_group,name,**kwargs):
    if isinstance(py_obj,types.MethodType) and not isinstance(py_obj.__self__,type):
        raise TypeError("bound instance methods can not be hickled!")
    return h_group.create_dataset(
        name,
        shape = None,
        dtype = int,
        **no_compression(kwargs)
    ),()
    
def create_class_dataset(py_obj,h_group,name,**kwargs):
    return h_group.create_dataset(
        name,
        shape = None,
        dtype = int,
        **no_compression(kwargs)
    ),()

def create_object_dataset(py_obj,h_group,name,**kwargs):
    reducer = pickle.Pickler.dispatch.get(py_obj.__class__,None)
    if reducer is None:
        reducer = getattr(py_obj,'__reduce_ex__',None)
        if reducer is None:
            reducer = getattr(py_obj,'__reduce__',None)
            if reducer is None:
                raise NotHicklable("can pickle handle?")
            reduced_obj = reducer()
        else:
            reduced_obj = reducer(pickle.DEFAULT_PROTOCOL)
    else:
        reduced_obj = reducer(py_obj)
    object_group = h_group.create_group(name)

    def present_object():
        yield "0",reduced_obj[0],{},kwargs
        yield "1",reduced_obj[1],{},kwargs
        if len(reduced_obj) < 3:
            return
        if reduced_obj[2] is not None:
            if len(reduced_obj) > 5 and reduced_obj[5] is not None:
                yield "5",reduced_obj[5],{},kwargs
            elif getattr(py_obj,'__setstate__',None) is None and ( not isinstance(reduced_obj[2],dict) or getattr(py_obj,'__dict__',None) is None ):
                raise NotHicklable("can pickle handle?")
            yield "2",reduced_obj[2],{},kwargs
        # TODO what if reduced_obj has length of 6 and reduced_obj[2] is None ?
        #      extend as soon as related usecase occurs
        if len(reduced_obj) < 4:
            return
        if reduced_obj[3] is not None:
            yield "3",IteratorProxy(reduced_obj[3],length_hint=operator.length_hint(py_obj)),{},kwargs
        if len(reduced_obj) > 4 and reduced_obj[4] is not None:
            yield "4",IteratorProxy(reduced_obj[4],length_hint=operator.length_hint(py_obj)),{},kwargs
            
    return object_group,present_object()
    
class ObjectContainer(PyContainer):
    __slots__ = ()

    _notset = b''
    
    def __init__(self,h5_attrs, base_type, object_type):
        super(ObjectContainer,self).__init__(h5_attrs,base_type,object_type,_content = [self._notset] * 6)
    
    def append(self,name,py_obj,h5_attrs):
        item_index = int(name)
        if item_index is None or self._content[item_index] is not self._notset:
            raise ValueError("double binding of '{}' object item".format(name))
        self._content[item_index] = py_obj

    def convert(self):
        py_obj = self._content[0](*self._content[1])
        if self._content[5] is not self._notset:
            if self._content[2] is self._notset:
                raise ValueError("lost 'state' object item")
            self._content[5](py_obj,self._content[2])
        elif self._content[2] is not self._notset:
            set_state = getattr(py_obj,'__setstate__',None)
            if set_state is None:
                py_obj.__dict__.update(self._content[2])
            elif not isinstance(self._content[2],bool) or self._content[2]:
                set_state(self._content[2])
        if self._content[3] is not self._notset:
            extend = getattr(py_obj,'extend',None)
            if extend is None or len(self._content[3]) < 5:
                iterator = iter(self._content[3])
                item = next(iterator,self._notset)
                while item is not self._notset:
                    py_obj.append(item)
                    item = next(iterator,self._notset)
            else:
                extend(self._content[3])
        if self._content[4] is not self._notset:
            iterator = iter(self._content[4])
            key,value = next(iterator,(self._notset,None))
            while key is not self._notset:
                py_obj[key] = value
                key,value = next(iterator,(self._notset,None))
        return py_obj
