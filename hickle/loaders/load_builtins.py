# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of
hickle.

"""


# %% IMPORTS
# Built-in imports
import warnings
import re
import operator
import collections

# Package imports
import numpy as np
import dill as pickle
import h5py
import warnings

# hickle imports
from hickle.helpers import PyContainer,NotHicklable,no_compression,nobody_is_my_name


# Define conversion dict of all dict key types which shall be mapped
# to their string representation. Any other key types are directly handled
# by create_dictlike_dataset function
dict_key_types_dict = {
    b'float': float,
    b'bool': bool,
    b'int': int,
    b'complex': complex,
    b'NoneType': eval,
    b'f':float,
    b'b':bool,
    b'i':int,
    b'c':complex,
    'N':eval,
    'f':float,
    'b':bool,
    'i':int,
    'c':complex,
    'N':eval
}

# %% FUNCTION DEFINITIONS

def create_scalar_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a scalar (int, float,
            bool, str, etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        correspoinding h5py.Dataset and empty subitems list
    """

    # If py_obj is an integer and cannot be stored in 64-bits, convert to str
    if isinstance(py_obj, int) and (py_obj.bit_length() > 64):
        byte_array = str(py_obj).encode('ascii')
        return h_group.create_dataset(name,data = byte_array,**kwargs),()

    return h_group.create_dataset(name, data=py_obj, **no_compression(kwargs)),()


def create_none_dataset(py_obj, h_group, name, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset
    Returns:
        correspoinding h5py.Dataset and empty subitems list
    """
    return h_group.create_dataset(name, shape=None,dtype = 'V1',**no_compression(kwargs)),()


def check_iterable_item_type(first_item,iter_obj):
    """
    checks if for all items of an iterable sequence (list, tuple, etc.) a least common
    dtype exists to which all items can be safely be casted.

    Args:
        first_item: the first item of the iterable sequence used to initialize the dtype
        iter_obj: the remaing items of the iterable sequence

    Returns:
        the least common dtype or none if not all items can be casted
    """

    if (
        operator.length_hint(first_item) > 1 or
        ( operator.length_hint(first_item) == 1 and not isinstance(first_item,(str,bytes)) ) or
        np.ndim(first_item) != 0
    ):
        return None
    dtype = np.dtype(first_item.__class__)
    if dtype.name == 'object' or 'str' in dtype.name or ( 'bytes' in dtype.name and len(first_item) > 1):
        return None
    for item in iter_obj:
        if np.ndim(item) != 0:
            return None
        common_dtype = np.result_type(np.dtype(item.__class__),dtype)
        if common_dtype.name == 'object' or 'str' in common_dtype.name or ( 'bytes' in common_dtype.name and len(item) > 1 ):
            return None
        if dtype != common_dtype:
            dtype = common_dtype
    return dtype

def create_listlike_dataset(py_obj, h_group, name,list_len = -1,item_dtype = None, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        Group or Dataset representing listlike object and a list of subitems toi
        be stored within this group. In case of Dataset this list is allways empty 
    """

    if isinstance(py_obj,(str,bytes)):
        # strings and bytes are stored as array of bytes with strings encoded using utf8 encoding
        if isinstance(py_obj,str):
            py_obj = py_obj.encode("utf8")
        dataset = h_group.create_dataset( name, data = np.array(py_obj,copy=False), **no_compression(kwargs) )
        #dataset.attrs["bytes"] = py_obj.__class__.__name__.encode("ascii")
        return dataset,()
        
    if len(py_obj) < 1:
        # listlike object is empty just store empty dataset
        return h_group.create_dataset(name,shape=None,dtype=int,**no_compression(kwargs)),()

    if list_len < 0:
        # neither length nor dtype of items is know compute them now
        item_dtype = check_iterable_item_type(py_obj[0],py_obj[1:])
        list_len = len(py_obj)

    if item_dtype or list_len < 1:
        # create a dataset and mapp all items to least common dtype
        shape = (list_len,) if list_len > 0 else None
        dataset = h_group.create_dataset(name,shape = shape,dtype = item_dtype,**kwargs)
        for index,item in enumerate(py_obj,0):
            dataset[index] = item_dtype.type(item)
        return dataset,()

    # crate group and provide generator yielding all subitems to be stored within
    item_name = "{:d}"
    def provide_listlike_items():
        if list_len < 100000000:
            for index,item in enumerate(py_obj,0):
                yield item_name.format(index),item,{},kwargs
            return
        for index,item in enumerate(py_obj,0):
            yield item_name.format(index),item,{"idx":index},kwargs
    h_subgroup = h_group.create_group(name)
    h_subgroup.attrs["cnt"] = list_len
    return h_subgroup,provide_listlike_items()
                

def create_setlike_dataset(py_obj,h_group,name,**kwargs):
    """
    Creates a dataset or group for setlike objects. 

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        Group or Dataset representing listlike object and a list of subitems toi
        be stored within this group. In case of Dataset this list is allways empty 
    """

    # set objects do not support indexing thus determination of item dtype has to
    # be handled specially. Call create_listlike_dataset for proper creation
    # of corresponding dataset
    if not py_obj:
        # dump empty set
        return h_group.create_dataset(name,data = list(py_obj),shape = None,dtype = int,**no_compression(kwargs)),()
    set_iter = iter(py_obj)
    first_item = next(set_iter)
    item_dtype = check_iterable_item_type(first_item,set_iter)
    return create_listlike_dataset(py_obj,h_group,name,list_len = len(py_obj),item_dtype = item_dtype,**kwargs)
    

_byte_slashes = re.compile(b'[\\/]')
_str_slashes = re.compile(r'[\\/]')

def create_dictlike_dataset(py_obj, h_group, name, **kwargs):

    """ Creates a data group for each key in dictionary

    Notes:
        This is a very important function which uses the recursive _dump
        method to build up hierarchical data models stored in the HDF5 file.
        As this is critical to functioning, it is kept in the main hickle.py
        file instead of in the loaders/ directory.

    Args:
        py_obj: python object to dump; should be dictionary
        h_group (h5.File.group): group to dump data into.
        name (str): h5 node name 
            iterable.
    """

    h_dictgroup = h_group.create_group(name)
    key_value_pair_name = "k{:d}"
    def makeindexandkey(index,key):
        return {'idx':index,'key':key}
    def makeindex(index):
        return {'idx':index}
    def makekeyonly(index,key):
        return {'key':key}
    def makeemtpy(index):
        return {}
    if isinstance(py_obj,collections.OrderedDict):
        create_attrs = makeindexandkey 
        create_index = makeindex
        h_dictgroup.attrs['order'] = True
    else:
        create_attrs = makekeyonly
        create_index = makeemtpy
        h_dictgroup.attrs['order'] = False

    def package_dict_items():
        """
        generator yielding appropriate parameters for dumping each
        dict key value pair
        """
        for idx, (key, py_subobj) in enumerate(py_obj.items()):
            # Obtain the raw string representation of this key
            key_base_type = key.__class__.__name__.encode("utf8")
            if isinstance(key,str):
                if not _str_slashes.search(key):
                    yield r's"{}"'.format(key),py_subobj,create_index(idx),kwargs
                    continue
            elif isinstance(key,bytes):
                if not _byte_slashes.search(key):
                    try:
                        h_key = key.decode("utf8")
                    except UnicodeError: # pragma nocover
                        pass
                    else:
                        yield r'B"{}"'.format(h_key),py_subobj,create_index(idx),kwargs
                        continue
            elif key_base_type in dict_key_types_dict:
                h_key = "{}{!r}".format(key_base_type[0],key)
                if not _str_slashes.search(h_key):
                    yield h_key,py_subobj,create_index(idx),kwargs
                    continue
            sub_node_name = key_value_pair_name.format(idx)
            yield sub_node_name,(key,py_subobj),create_index(idx),kwargs
    return h_dictgroup,package_dict_items()



def load_scalar_dataset(h_node,base_type,py_obj_type):
    """
    loads scalar dataset

    Args:
        h_node (h5py.Dataset): the hdf5 node to load data from
        base_type (bytes): bytes string denoting base_type 
        py_obj_type: final type of restored scalar

    Returns:
        resulting python object of type py_obj_type
    """
    data = h_node[()]
    # if h_node.dtype < 2 else bytearray(h_node[()])


    return py_obj_type(data) if data.__class__ is not py_obj_type else data

def load_none_dataset(h_node,base_type,py_obj_type):
    """
    returns None value as represented by underlying dataset
    """
    return None
    
def load_list_dataset(h_node,base_type,py_obj_type):
    """
    loads any kind of list like dataset

    Args:
        h_node (h5py.Dataset): the hdf5 node to load data from
        base_type (bytes): bytes string denoting base_type 
        py_obj_type: final type of restored scalar

    Returns:
        resulting python object of type py_obj_type
    """

    if h_node.shape is None:
        # empty list tuple or set just return new instance of py_obj_type
        return py_obj_type() if isinstance(py_obj_type,tuple) else py_obj_type(())
    str_type = h_node.attrs.get('str_type', base_type)
    content = h_node[()]
    if str_type == b'str':
        #if "bytes" in h_node.dtype.name and ( h_node.shape == () or h_node.dtype.itemsize > 1 ):
        #        # string dataset 4.0.x style convert it back to python string
        #        content = np.array(content, copy=False, dtype=str).tolist()
        #else:
        #    # decode bytes representing python string before final conversion
        content = bytes(content).decode("utf8")
    return py_obj_type(content) if content.__class__ is not py_obj_type else content

class ListLikeContainer(PyContainer):
    """
    PyContainer for all list like objects excempt set
    """

    __slots__ = ()

    # regular expression used to extract index value from name of group or dataset
    # representing subitem appended to the final list
    extract_index = re.compile(r'\d+$')

    # as None can and may be a valid list entry define an alternative marker for
    # missing items and indices
    def __init__(self,h5_attrs,base_type,object_type):
        # if number of items is defind upon group resize content to
        # at least match this ammount of subitems
        num_items = h5_attrs.get('count',None)
        if num_items is None:
            num_items = h5_attrs.get('num_items',0)
        super(ListLikeContainer,self).__init__(h5_attrs,base_type,object_type,_content = [nobody_is_my_name] * num_items)

    def append(self,name,item,h5_attrs):
        # load item index from attributes if known else extract it from name 
        index = h5_attrs.get("idx",None)
        if index is None:
            index = h5_attrs.get("item_index",None)
            if index is None:
                index_match = self.extract_index.search(name)
                if index_match is None:
                    if item is nobody_is_my_name:
                        # dummy data injected likely by load_nothing, ignore it
                        return
                    raise KeyError("List like item name '{}' not understood".format(name))
                index = int(index_match.group(0))
        # if index exceeds capacity of extend list apropriately
        if len(self._content) <= index:
            self._content.extend([nobody_is_my_name] * ( index - len(self._content) + 1 ))
        if self._content[index] is not nobody_is_my_name:
            raise IndexError("Index {} already set".format(index))
        self._content[index] = item
        
    def convert(self):
        return self._content if self.object_type is self._content.__class__ else self.object_type(self._content)

class SetLikeContainer(PyContainer):
    """
    PyContainer for all set like objects.
    """
    __slots__ = ()

    def __init__(self,h5_attrs, base_type, object_type):
        super(SetLikeContainer,self).__init__(h5_attrs,base_type,object_type,_content=set())

            
    def append(self,name,item,h5_attrs):
        self._content.add(item)

    def convert(self):
        return self._content if self._content.__class__ is self.object_type else self.object_type(self._content)

class DictLikeContainer(PyContainer):
    """
    PyContainer for all dict like objects
    """

    __slots__ = ('_random_order')

    
    _swap_key_slashes = re.compile(r"\\")

    def __init__(self,h5_attrs, base_type, object_type):
        super().__init__(h5_attrs,base_type,object_type)
        self._random_order = None if h5_attrs.get('order',True) else 0 

    def append(self,name,item,h5_attrs):
        key_base_type = h5_attrs.get('key',None)
        start_name = 0
        if key_base_type is None:
            key_base_type = h5_attrs.get('key_base_type',None)
            if key_base_type is None:
                key_base_type = name[0]
                start_name=1
        if key_base_type in(b'str','s'):
            item = (
                name[start_name+1:-1] if name[start_name] == '"' else self._swap_key_slashes.sub(r'/',name)[1:-1],
                item
            )
        elif key_base_type in (b'bytes','B','b'):
            item = (
                name[2:-1].encode("utf8") if name[:2] in ('B"','b"') else self._swap_key_slashes.sub(r'/',name)[1:-1],
                item
            )
        elif key_base_type not in (b'key_value','k'):
            load_key  = dict_key_types_dict.get(key_base_type,None)
            if load_key is None:
                if key_base_type not in {b'tuple'}:
                    raise ValueError("key type '{}' not understood".format(key_base_type.decode("utf8")))
                load_key = eval
            item = (
                load_key(self._swap_key_slashes.sub(r'/',name[start_name:])),
                item
            )
        key_index = h5_attrs.get('key_idx',None)
        if key_index is None:
            key_index = h5_attrs.get('idx',None)
            if key_index is None:
                if item[1] is nobody_is_my_name:
                    # dummy data injected most likely by load_nothing ignore it
                    return
                key_index = self._random_order
                try:
                    self._random_order += 1
                except TypeError:
                    raise KeyError("invalid dict item key_index missing")
        if len(self._content) <= key_index:
            self._content.extend([nobody_is_my_name] * ( key_index - len(self._content) + 1))
        if self._content[key_index] is not nobody_is_my_name:
            raise IndexError("Key index {} already set".format(key_index))
        self._content[key_index] = item

    def convert(self):
        return self.object_type(self._content)

    

# %% REGISTERS
class_register = [
    [list, b"list", create_listlike_dataset, load_list_dataset,ListLikeContainer],
    [tuple, b"tuple", create_listlike_dataset, load_list_dataset,ListLikeContainer],
    [dict, b"dict",create_dictlike_dataset,None,DictLikeContainer],
    [set, b"set", create_setlike_dataset, load_list_dataset,SetLikeContainer],
    [bytes, b"bytes", create_listlike_dataset, load_list_dataset],
    [str, b"str", create_listlike_dataset, load_list_dataset],
    [int, b"int", create_scalar_dataset, load_scalar_dataset,None,False],
    [float, b"float", create_scalar_dataset, load_scalar_dataset,None,False],
    [complex, b"complex", create_scalar_dataset, load_scalar_dataset,None,False],
    [bool, b"bool", create_scalar_dataset, load_scalar_dataset,None,False],
    [None.__class__, b"None", create_none_dataset, load_none_dataset,None,False]
]

exclude_register = []
