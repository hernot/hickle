"""
#lookup.py

This file manages all the mappings between hickle/HDF5 metadata and python
types.
There are three dictionaries that are populated here:

1) types_dict
Mapping between python types and dataset and group creation functions, e.g.
    types_dict = {
        list: (create_listlike_dataset, 'list'),
        int: (create_python_dtype_dataset, 'int'),
        np.ndarray: (create_np_array_dataset, 'ndarray'),
        }

2) hkl_types_dict
Mapping between hickle metadata and dataset loading functions, e.g.
    hkl_types_dict = {
        'list': load_list_dataset,
        'tuple': load_tuple_dataset
        }

3) hkl_container_dict
Mapping between hickle metadata and group container classes, e.g.
    hkl_contianer_dict = {
        'list': ListLikeContainer,
        'tuple': TupleLikeContainer,
        'dict': DictLikeContainer
    }

## Extending hickle to add support for other classes and types

The process to add new load/dump capabilities is as follows:

1) Create a file called load_[newstuff].py in loaders/
2) In the load_[newstuff].py file, define your create_dataset and load_dataset
   functions, along with the 'class_register' and 'exclude_register' lists.

"""


# %% IMPORTS
# Built-in imports
import sys
import warnings
import types
import operator
import functools as ft
import itertools
from importlib.util import find_spec, module_from_spec

# Package imports
import dill as pickle
import copyreg
import numpy as np
import h5py as h5

# hickle imports
from .helpers import PyContainer,not_dumpable,nobody_is_my_name,NotHicklable,no_compression


# %% GLOBALS
# Define dict of all acceptable types
types_dict = {}

# Define dict of all acceptable hickle types
hkl_types_dict = {}

# Define dict of all acceptable hickle container types
hkl_container_dict = {}

# Empty list (hashable) of loaded loader names
loaded_loaders = set()


# %% FUNCTION DEFINITIONS
def load_nothing(h_node,base_type,py_obj_type): # pragma: nocover
    """
    loads nothing
    """
    return nobody_is_my_name

def dump_nothing(py_obj, h_group, name, **kwargs): # pragma: nocover
    """
    dumps nothing
    """
    return nobody_is_my_name

def _force_dump(memo,key,value): # pragma: nocover
    pass

# h5py < 2.10 ref_dtype has to be explicitly created by
# call to special_dtype(ref=h5py.Reference) 
# h5py >= 2.10 provides already a ref_dtype where metadata is already properly
# set. Both ways to define dtype are equivalent
h5_major,h5_minor,_ = [ int(v) for v in h5.__version__.split('.')]
if h5_major > 2 or ( h5_major == 2 and h5_minor >= 10 ):
    memo_link_dtype = h5.ref_dtype
else:
    memo_link_dtype = np.dtype('O',metadata={'ref':h5.Reference})

def is_memo_link(dt):
    """
    checks whether dtype represence ref_dtype or not
    is modeled along h5py.check_ref_dtype (h5py >= 2.10).
    In constrast to the latter it just returns True if data type indicates
    h5py.Reference object and false otherwise.
    """
    try:
        return dt.metadata.get('ref',None) is h5.Reference
    except AttributeError: # pragma: nocover
        return False 

# %% CLASS DEFINITIONS

class _DictItem(): # pragma: nocover
    """
    dummy py_obj for dict_item loader
    """

class SerializedWarning(UserWarning): # pragma: nocover
    """ An object type was not understood

    The data will be serialized using pickle.
    """

class TypeMemoTables():
    """
    creates and maintains global type memoizatoin table listing 
    pickle strings for all types required to properly restore the content of
    the hickle file. The 'type' attribute of each data set and group contains
    a h5py.Reference to the appropriate entry.
    """
    __slots__ = ( '_py_type_table','_obj_type_link','_base_type_link')

    # list of the tables of all currently open hickle files
    tables = dict()

    def __init__(self,h_root_group):
        """
        create subgroup of root_group representing table of py_ob_type entries
        if table already exists load existing entries

        Parameters:
        -----------
            h_root_group (h5py.Group): root group to wich data is dumped
        """
        self._obj_type_link = dict()
        self._base_type_link = dict()
        self._py_type_table = h_root_group.get("py_obj_types_table",None)
        if self._py_type_table is None:
            self._py_type_table = h_root_group.create_group("py_obj_types_table",track_order = True)
        else:
            for index,entry in self._py_type_table.items():
                content = entry[()]
                if index[0] == 'b':
                    content = bytes(content)
                    base_link = self._base_type_link.get(content,None)
                    if base_link is None:
                        self._base_type_link[content] = entry
                    continue
                py_obj_type = pickle.loads(content)
                self._obj_type_link[id(py_obj_type)] = entry
        
    def py_obj_type_ref(self,py_obj_type,base_type=None,**kwargs):
        type_id = id(py_obj_type)
        entry = self._obj_type_link.get(type_id,None)
        if entry is not None:
            return entry.ref
        type_entry = bytearray(pickle.dumps(py_obj_type))
        type_entry = np.array([type_entry],copy = False)
        type_entry.dtype = '|S1'
        entry_id = len(self._py_type_table)
        entry = self._py_type_table.create_dataset(str(entry_id),data=type_entry,**kwargs)
        if base_type is not None:
            h_base =  self._base_type_link.get(base_type,None)
            if h_base is None:
                self._base_type_link[base_type] = h_base = self._py_type_table.create_dataset('b{}'.format(len(self._base_type_link)),data=np.array(base_type),**no_compression(kwargs))
            entry.attrs['base'] = h_base.ref
        self._obj_type_link[type_id] = entry
        return entry.ref

    @staticmethod
    def create_table(h_root_group):
        table = TypeMemoTables.tables.get(h_root_group.file.id,None)
        if table is None:
            table = TypeMemoTables.tables[h_root_group.file.id] = TypeMemoTables(h_root_group)
        return table

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        if self._py_type_table is None:
            return
        TypeMemoTables.tables.pop(self._py_type_table.file.id,None)
        self._py_type_table = None
        self._obj_type_link = None

class TypeLookupTables():
    # TODO add base type link table
    __slots__ = ('_py_obj_type_table','_py_obj_type_link','_pickle_loads')

    tables = dict()
    def __init__(self,obj_type_table,pickle_loads):
        self._py_obj_type_table = obj_type_table
        self._py_obj_type_link = dict()
        self._pickle_loads = pickle_loads

    def py_obj_type_from_ref(self,type_ref,h_attrs):
        #type_ref = h_node.attrs.get('type',None)
        if type_ref is None:
            return object,b'pickle'
        if not isinstance(type_ref,h5.Reference):
            return self._pickle_loads(type_ref),h_attrs.get('base_type',b'pickle')
        entry = self._py_obj_type_table.get(type_ref,None)
        if entry is None:
            raise ValueError("type_ref invalid reference")
        type_info = self._py_obj_type_link.get(entry.id,None)
        if type_info is None:
            base_type = entry.attrs.get('base',None)
            if base_type is None:
                base_type = b'pickle'
            else:
                base_type = bytes(self._py_obj_type_table[base_type][()])
            type_info = self._py_obj_type_link[entry.id] = (
                self._pickle_loads(entry[()]),
                base_type
            )
        return type_info
                
    @staticmethod
    def load_table(h_root_group,pickle_loads):
        table = TypeLookupTables.tables.get(h_root_group.file.id,None)
        if table is not None:
            return table
        py_obj_type_table = h_root_group.get('py_obj_types_table',None)
        if isinstance(py_obj_type_table,h5.Group):
            table = TypeLookupTables.tables[h_root_group.file.id] = TypeLookupTables(py_obj_type_table,pickle_loads)
        else:
            table = TypeLookupTables.tables[h_root_group.file.id] = NoLookupTables(h_root_group,pickle_loads)
        return table

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        if self._py_obj_type_table is None:
            return
        TypeLookupTables.tables.pop(self._py_obj_type_table.file.id,None)
        self._py_obj_type_table = None
        self._py_obj_type_link = None
        
class NoLookupTables(TypeLookupTables):
    def __init__(self,h_root_group,pickle_loads):
        super().__init__(h_root_group,pickle_loads)

    def py_obj_type_from_ref(self,type_id,h_attrs):
        try:
            return self._pickle_loads(type_id),h_attrs.get('base_type',b'pickle')
        except Exception as invalid:
            raise ValueError("no type memo tables") from invalid


#####################
# loading optional  #
#####################



# This function registers a class to be used by hickle
def register_class(myclass_type, hkl_str, dump_function=None, load_function=None, container_class=None,memoize = True):
    """ Register a new hickle class.

    Parameters:
    -----------
        myclass_type type(class): type of class
        hkl_str (str): String to write to HDF5 file to describe class
        dump_function (function def): function to write data to HDF5
        load_function (function def): function to load data from HDF5
        container_class (class def): proxy class to load data from HDF5

    Raises:
    -------
        TypeError:
            myclass_type represents a py_object the loader for which is to
            be provided by hickle.lookup and hickle.hickle module only
            
    """

    if (
        myclass_type is object or
        isinstance(
            myclass_type,
            (types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)
        ) or
        issubclass(myclass_type,(type,_DictItem))
    ):
        # object as well als all kinds of functions and methods as well as all class objects and
        # the special _DictItem class are to be handled by hickle core only. 
        dump_module = getattr(dump_function,'__module__','').split('.')
        load_module = getattr(load_function,'__module__','').split('.')
        container_module = getattr(container_class,'__module__','').split('.')
        ishickle = {'hickle',''}
        if ( 
            dump_module[0] not in ishickle or
            load_module[0] not in ishickle or
            container_module[0] not in ishickle
        ):
            raise TypeError(
                "loader for '{}' type managed by hickle only".format(
                    myclass_type.__name__
                )
            )
        if (
            dump_module[1:2] == ["loaders"] or
            load_module[1:2] == ["loaders"] or
            container_module[1:2] == ["loaders"]
        ):
            raise TypeError(
                "loader for '{}' type managed by hickle core only".format(
                    myclass_type.__name__
                )
            )
    # add loader
    if dump_function is not None:
        types_dict[myclass_type] = (dump_function, hkl_str, ( dict.__setitem__ if memoize else _force_dump ) )
    if load_function is not None:
        hkl_types_dict[hkl_str] = load_function
    if container_class is not None:
        hkl_container_dict[hkl_str] = container_class


def register_class_exclude(hkl_str_to_ignore):
    """ Tell loading funciton to ignore any HDF5 dataset with attribute
    'type=XYZ'

    Args:
        hkl_str_to_ignore (str): attribute type=string to ignore and exclude
            from loading.
    """

    if hkl_str_to_ignore in {b'dict_item',b'pickle'}:
        raise ValueError(
            "excluding '{}' base_type managed by hickle core not possible".format(
                hkl_str_to_ignore
            )
        )
    hkl_types_dict[hkl_str_to_ignore] = load_nothing
    hkl_container_dict[hkl_str_to_ignore] = NoContainer


def load_loader(py_obj_type, type_mro = type.mro):
    """
    Checks if given `py_obj` requires an additional loader to be handled
    properly and loads it if so. 

    Parameters:
    -----------
        py_obj:
            the Python object to find an appropriate loader for

    Returns:
    --------
        py_obj:
            the Python object the loader was requested for

        (create_dataset,base_type):
            tuple providing create_dataset function and name of base_type
            used to represent py_obj, if create_data reads None instead than
            create_pickled_dataset has to be called independent of base_type.

    Raises:
    -------
        RuntimeError:
            in case py object is defined by hickle core machinery.

    """

    # any function or method object, any class object will be passed to pickle
    # ensure that in any case create_pickled_dataset is called.

    # get the class type of py_obj and loop over the entire mro_list
    for mro_item in type_mro(py_obj_type):
        # Check if mro_item can be found in types_dict and return if so
        loader_item = types_dict.get(mro_item,None)
        if loader_item is not None:
            return py_obj_type,loader_item

        # Obtain the package name of mro_item
        package_list = mro_item.__module__.split('.',3)

        if package_list[0] == 'hickle':
            if package_list[1] != 'loaders':
                print(mro_item,package_list)
                raise RuntimeError(
                    "objects defined by hickle core must be registerd"
                    " before first dump or load"
                )
            if (
                len(package_list) < 3 or
                not package_list[2].startswith("load_") or
                '.' in package_list[2][5:]
            ):
                warnings.warn(
                    "ignoring '{!r}' dummy type not defined by loader module".format(py_obj_type),
                    RuntimeWarning
                )
                continue
            # dummy objects are not dumpable ensure that future lookups return that result
            loader_item = types_dict.get(mro_item,None)
            if loader_item is None:
                loader_item = types_dict[mro_item] = ( not_dumpable, b'NotHicklable',_force_dump )
            # ensure module of mro_item is loaded as loader as it will contain
            # loader which knows how to handle group or dataset with dummy as 
            # py_obj_type 
            loader_name = mro_item.__module__
            if loader_name in loaded_loaders:
                # loader already loaded as triggered by dummy abort search and return
                # what found so far as fallback to further bases does not make sense
                return py_obj_type,loader_item
        else:
            # Obtain the name of the associated loader
            loader_name = 'hickle.loaders.load_%s' % (package_list[0])

        # Check if this module is already loaded, and return if so
        if loader_name in loaded_loaders:
            # loader is loaded but does not define loader for mro_item
            # check next base class
            continue

        # check if loader module has already been loaded. If use that instead
        # of importing it anew
        loader = sys.modules.get(loader_name,None)
        if loader is None:
            # Try to load a loader with this name
            loader_spec = find_spec(loader_name)
            if loader_spec is None:
    
                # no module sepecification found for module
                # check next base class
                continue
            # import the the loader module described by module_spec
            # any import errors and exceptions result at this stage from
            # errors inside module and not cause loader module does not
            # exists
            loader = module_from_spec(loader_spec)
            loader_spec.loader.exec_module(loader)
            sys.modules[loader_name] = loader

        # load all loaders defined by loader module
        # no performance benefit of starmap or map if required to build
        # list or tuple of None's returned
        for next_loader in loader.class_register:
            register_class(*next_loader)
        for drop_loader in loader.exclude_register:
            register_class_exclude(drop_loader)
        loaded_loaders.add(loader_name)

        # check if loader module defines a loader for base_class mro_item
        loader_item = types_dict.get(mro_item,None)
        if loader_item is not None:
            # return loader for base_class mro_item
            return py_obj_type,loader_item
        # the new loader does not define loader for mro_item
        # check next base class

    # no appropriate loader found return fallback to pickle
    return py_obj_type,(create_pickled_dataset,b'pickle',dict.__setitem__)

def type_legacy_mro(cls):
    """
    drop in replacement of type.mro for loading legacy hickle 4.0.x files which were
    created without generalized PyContainer objects in mind. consequently some
    h5py.Datasets and h5py.Group objects expose function objets as their py_obj_type
    type.mro expects classes only.

    Parameters:
    -----------
        cls (type):
            the py_obj_type/class of the object to load or dump

    Returns:
    --------
        mro list for cls as returned by type.mro  or in case cls is a function or method
        a single element tuple is returned
    """
    if isinstance(
        cls,
        (types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)
    ):
        return (cls,)
    return type.mro(cls) 

load_legacy_loader = ft.partial(load_loader,type_mro = type_legacy_mro)

# %% BUILTIN LOADERS (not maskable)

class NoContainer(PyContainer): # pragma: nocover
    """
    load nothing container
    """

    def convert(self):
        pass


class _DictItemContainer(PyContainer):
    """
    PyContainer reducing hickle version 4.0.0 dict_item type h5py.Group to 
    its content for inclusion within dict h5py.Group
    """

    def convert(self):
        return self._content[0]

register_class(_DictItem, b'dict_item',dump_nothing,load_nothing,_DictItemContainer)

def create_pickled_dataset(py_obj, h_group, name, reason = None, **kwargs):
    """
    try to call __reduce_ex__ or __reduce__ on object if defined
    and dump as object group. Alternatively create pickle string in
    case py_obj is class, function, method or in case object can not
    be reduced. In this latter case raise a warning that not undersood
    object could not be 
    If no match is made, raise a warning and convert to pickle string

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    reason_str = " (Reason: %s)" % (reason) if reason is not None else ""
    warnings.warn(
        "{!r} type not understood, data is serialized:{:s}".format(
            py_obj.__class__.__name__, reason_str
        ),
        SerializedWarning
    )
        

    # store object as pickle string
    pickled_obj = bytearray(pickle.dumps(py_obj))
    pickled_obj = np.array([pickled_obj],copy=False)
    pickled_obj.dtype = 'S1'
    d = h_group.create_dataset(name, data=pickled_obj, **kwargs)
    return d,() 

def load_pickled_data(h_node, base_type, py_obj_type):
    """
    loade pickle string and return resulting py_obj
    """

    return pickle.loads(h_node[()])
        
# no dump method is registered for object as this is the default for
# any unknown object and for classes, functions and methods
register_class(object,b'pickle',None,load_pickled_data)

def create_compact_dataset(py_obj, h_group, name, **kwargs):
    compact = getattr(py_obj,'__compact__',None)
    table = TypeMemoTables.tables.get(h_group.file.id,None)
    if compact is not None and table is not None:
        compact_py_obj = compact()
        if compact_py_obj is not None:
            py_obj_type, (create_dataset, base_type, memoize) = load_loader(compact_py_obj.__class__)
            try:
                h_node,h_subitems = create_dataset(compact_py_obj, h_group, name, **kwargs)
            except NotHicklable:
                pass
            else:
                if create_dataset not in (create_pickled_dataset,):
                    h_node.attrs['compact_type'] = table.py_obj_type_ref(py_obj_type,base_type,**kwargs)
                return h_node,h_subitems
    return create_pickled_dataset(py_obj)

def load_compact_dataset(h_node,base_type,py_obj_type):
    compact_type = h_node.attrs.get('compact_type',None)
    if not isinstance(compact_type,h5.Reference):
        return pickle.loads(h_node[()])
    table = TypeLookupTables.tables.get(h_node.file.id,None)
    if table is None:
        return None
    expanded_data = py_obj_type.__new__(py_obj_type)
    py_obj_type,base_type = table.py_obj_type_from_ref(compact_type,h_node.attrs)
    py_obj_type,(_,_,memoize) = load_loader(py_obj_type)
    load_fn = hkl_types_dict.get(base_type,)
    data = load_fn(h_node,base_type,py_obj_type)
    expanded_data.__expand__(data)
    return expanded_data


class CompactContainer(PyContainer):
    """
    PyContainer handling restore of object from object group
    """

    __slots__ = ('_compact_container','append')

    def __init__(self,h5_attrs, base_type, object_type):
        super(CompactContainer,self).__init__(h5_attrs,base_type,object_type,_content = dict())
        self._compact_container = None
        self.append = self._append

    def filter(self,items):
        gettype,iterateall = itertools.tee(items,2)
        itemiter = iter(gettype)
        nextitem = next(itemiter,self)
        if nextitem is self:
            return
        name,h_node = nextitem
        compact_type = self._h5_attrs.get('compact_type',None)
        table = TypeLookupTables.tables.get(h_node.file.id,None)
        if table is None or not isinstance(compact_type,h5.Reference):
            return
        py_obj_type,base_type = table.py_obj_type_from_ref(compact_type,h_node.attrs)
        py_obj_type,(_,_,memoize) = load_loader(py_obj_type)
        py_container_class = hkl_container_dict.get(base_type,None)
        if py_container_class is None:
            raise RuntimeError("Cannot load container proxy for %s data type " % base_type)
        self._compact_container = py_container_class(self._h5_attrs,base_type,py_obj_type)
        self.append = self._compact_container.append
        yield from self._compact_container.filter(iterateall)

    def _append(self,name,item,h5_attrs):
        raise Exception("not replaced by _compact_container.append")

    def convert(self):
        expanded_object = self.object_type.__new__(self.object_type)
        if not self._compact_container:
            return expanded_object
        compact_data = self._compact_container.convert()
        expanded_object.__expand__(compact_data)
        return expanded_object

# ensure files containing compacted dataset can be loaded even if 
# compacting was not activated for the affected objects
register_class(object,b'compact',None,load_compact_dataset,CompactContainer)

def enable_compact_expand(*args):
    for py_obj_type in args:
        if not isinstance(py_obj_type,type):
            raise TypeError("'py_obj_type' must be class")
        if not callable(getattr(py_obj_type,'__compact__')) or not callable(getattr(py_obj_type,'__expand__')):
            raise TypeError("'{}' type object does not support hickl compact expand protocol".format(py_obj_type.__name__))
        # register create_compact_dataset as dump funktion for py_obj_type
        # registering load_compact_dataset and CompactContainer is not necessary as both 
        # are provided by hickle in any case
        register_class(py_obj_type,b'compact',create_compact_dataset,None,None)

def disable_compact_expand(*args):
    for py_obj_type in args:
        if not isinstance(py_obj_type,type):
            raise TypeError("'py_obj_type' must be class")
        if not callable(getattr(py_obj_type,'__compact__')) or not callable(getattr(py_obj_type,'__expand__')):
            raise TypeError("'{}' type object does not support hickl compact expand protocol".format(py_obj_type.__name__))
        itementry = types_dict.get(py_obj_type,None)
        if itementry is None or itementry[:2] != (create_compact_dataset,b'compact'):
            # compacting for object not enabled
            continue
        types_dict.pop(py_obj_type)
        # check whether a dedicated loader would exist for py_obj_type
        # if reload it
        load_loader(py_obj_type)
    

def _moc_numpy_array_object_lambda(x):
    """
    drop in replacement for lambda object types which seem not
    any more be accepted by pickle for Python 3.8 and onward.
    see fix_lambda_obj_type function below

    Parameters:
    -----------
        x (list): itemlist from which to return first element

    Returns:
        first element of provided list
    """
    return x[0]

register_class(_moc_numpy_array_object_lambda,b'moc_lambda',dump_nothing,load_nothing)

def fix_lambda_obj_type(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict"):
    """
    drop in replacement for pickle.loads method when loading files created by hickle 4.0.x 
    It captures any TypeError thrown by pickle.loads when encountering a picle string 
    representing a lambda function used as py_obj_type for a h5py.Dataset or h5py.Group
    While in Python <3.8 pickle loads creates the lambda Python >= 3.8 throws an 
    error when encountering such a pickle string. This is captured and _moc_numpy_array_object_lambda
    returned instead. futher some h5py.Group and h5py.Datasets do not provide any 
    py_obj_type for them object is returned assuming that proper loader has been identified
    by other objects already
    """
    if bytes_object is None:
        return object
    try:
        return pickle.loads(bytes_object,fix_imports=fix_imports,encoding=encoding,errors=errors)
    except TypeError:
        print("reporting ",_moc_numpy_array_object_lambda)
        return _moc_numpy_array_object_lambda
