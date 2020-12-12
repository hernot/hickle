#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle.py

Unit tests for hickle module.

"""


# %% IMPORTS
# Built-in imports
from collections import OrderedDict as odict
import os
import re
from pprint import pprint


# Package imports
import pytest
import dill as pickle
import h5py
import numpy as np
from py.path import local

# hickle imports
from hickle import dump, helpers, hickle, load, lookup

# Set current working directory to the temporary directory
local.get_temproot().chdir()


# %% GLOBALS

# %% HELPER DEFINITIONS

# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    """
    import h5py as h5
    dummy_file = h5.File('hickle_lookup_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()
    
@pytest.fixture
def test_file_name(request):
    yield "{}.hkl".format(request.function.__name__)

# %% FUNCTION DEFINITIONS

def test_file_opener(h5_data,test_file_name):
    """
    test file opener function
    """

    # check that file like object is properly initialized for writing
    filename = test_file_name.replace(".hkl","_{}.{}")
    with open(filename.format("w",".hdf5"),"w") as f:
        h5_file,path,close_flag = hickle.file_opener(f,"root","w")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r+'
        h5_file.close()

    # check that file like object is properly initialized for reading
    with open(filename.format("w",".hdf5"),"r") as f:
        h5_file,path,close_flag = hickle.file_opener(f,"root","r")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r'
        h5_file.close()

    # check that h5py.File object is properly intialized for writing
    h5_file,path,close_flag = hickle.file_opener(h5_data,"","w")
    assert isinstance(h5_file,h5py.File) and path == "/root_group"
    assert h5_file.mode == 'r+' and not close_flag

    # check that a new file is created for provided filename and properly intialized
    h5_file,path,close_flag = hickle.file_opener(filename.format("w",".hkl"),"root_group","w")
    assert isinstance(h5_file,h5py.File) and path == "/root_group"
    assert h5_file.mode == 'r+' and close_flag
    h5_file.close()
    
    # check that any other object not beein a file like object, a h5py.File object or
    # a filename string triggers an  FileError exception
    with pytest.raises(
        hickle.FileError,
        match = r"Cannot\s+open\s+file.\s+Please\s+pass\s+either\s+a\s+"
                r"filename\s+string,\s+a\s+file\s+object,\s+or\s+a\s+h5py.File"
    ):
        h5_file,path,close_flag = hickle.file_opener(dict(),"root_group","w")
        
def test_recursive_dump(h5_data):
    """
    test _dump function and that it properly calls itself recursively 
    """

    # check that dump function properly creates a list dataset and
    # sets appropriate values for 'type' and 'base_type' attributes
    data = simple_list = [1,2,3,4]
    memo = {}
    type_memo = lookup.TypeMemoTables.create_table(h5_data.parent)
    check_memo = lookup.TypeLookupTables.load_table(h5_data.parent,pickle.loads)
    hickle._dump(data, h5_data, "simple_list",memo=memo,type_memo = type_memo)
    dumped_data = h5_data["simple_list"]
    assert check_memo.py_obj_type_from_ref(dumped_data.attrs['type']) == data.__class__
    assert dumped_data.attrs['base_type'] == b'list'
    assert np.all(dumped_data[()] == simple_list)

    # check that dump function properly creats a group representing
    # a dictionary and its keys and values and sets appropriate values
    # for 'type', 'base_type' and 'key_base_type' attributes
    data = {
        '12':12,
        (1,2,3):'hallo'
    }
    hickle._dump(data, h5_data, "some_dict",memo = memo,type_memo = type_memo)
    dumped_data = h5_data["some_dict"]
    assert check_memo.py_obj_type_from_ref(dumped_data.attrs['type']) == data.__class__

    # check that the name of the resulting dataset for the first dict item
    # resembles double quouted string key and 'type', 'base_type 'key_base_type'
    # attributes the resulting dataset are set accordingly
    assert dumped_data.attrs['base_type'] == b'dict'
    first_item = dumped_data['"12"']
    assert first_item[()] == 12 and first_item.attrs['key_base_type'] == b'str'
    assert first_item.attrs['base_type'] == b'int'
    assert check_memo.py_obj_type_from_ref(first_item.attrs['type']) == data['12'].__class__ 
    
    # check that second item is converted into key value pair group, that
    # the name of that group reads 'data0' and that 'type', 'base_type' and
    # 'key_base_type' attributes are set accordingly
    second_item = dumped_data.get("data0",None)
    if second_item is None:
        second_item = dumped_data["data1"]
    assert second_item.attrs['key_base_type'] == b'key_value'
    assert second_item.attrs['base_type'] == b'tuple'
    assert check_memo.py_obj_type_from_ref(second_item.attrs['type']) == tuple

    # check that content of key value pair group resembles key and value of
    # second dict item
    key = second_item['data0']
    value = second_item['data1']
    assert np.all(key[()] == (1,2,3)) and key.attrs['base_type'] == b'tuple'
    assert check_memo.py_obj_type_from_ref(key.attrs['type']) == tuple
    assert bytes(value[()]) == 'hallo'.encode('utf8') and value.attrs['base_type'] == b'str'
    assert check_memo.py_obj_type_from_ref(value.attrs['type']) == str

    # check that objects for which no loader has been registred or for which
    # available loader raises NotHicklable exception are handled by 
    # create_pickled_dataset function 
    backup_dict_loader,back_up_memo_entry = lookup.types_dict[dict],memo.pop(id(data))
    def fail_create_dict(py_obj,h_group,name,**kwargs):
        raise helpers.NotHicklable("test loader shrugg")
    lookup.types_dict[dict] = fail_create_dict,*backup_dict_loader[1:]
    with pytest.warns(lookup.SerializedWarning):
        hickle._dump(data, h5_data, "pickled_dict",memo = memo,type_memo = type_memo)
    dumped_data = h5_data["pickled_dict"]
    lookup.types_dict[dict] = backup_dict_loader
    memo[id(data)] = back_up_memo_entry
    assert set(key for key in dumped_data.keys()) == {'create','args','keys','values'}

def test_recursive_load(h5_data):
    """
    test _load function and that it properly calls itself recursively 
    """

    # check that simple scalar value is properly restored on load from
    # corresponding dataset
    data = 42
    data_name = "the_answer"
    dump_memo = {}
    memo = {}
    create_memo = lookup.TypeMemoTables.create_table(h5_data.parent)
    type_memo = lookup.TypeLookupTables.load_table(h5_data.parent,pickle.loads)
    hickle._dump(data, h5_data, data_name,memo = dump_memo,type_memo = create_memo)
    py_container = hickle.RootContainer(h5_data.attrs,b'hickle_root',hickle.RootContainer)
    hickle._load(py_container, data_name, h5_data[data_name],memo = memo,type_memo = type_memo)
    assert py_container.convert() == data

    # check that dict object is properly restored on load from corresponding group
    data = {'question':None,'answer':42}
    data_name = "not_formulated"
    hickle._dump(data, h5_data, data_name,memo = dump_memo,type_memo = create_memo)
    py_container = hickle.RootContainer(h5_data.attrs,b'hickle_root',hickle.RootContainer)
    hickle._load(py_container, data_name, h5_data[data_name],memo = memo,type_memo = type_memo)
    assert py_container.convert() == data

    
    # check that objects for which no loader has been registred or for which
    # available loader raises NotHicklable exception are properly restored on load
    # from corresponding copy protocol group or pickled data string 
    backup_dict_loader = lookup.types_dict[dict]
    def fail_create_dict(py_obj,h_group,name,**kwargs):
        raise helpers.NotHicklable("test loader shrugg")
    lookup.types_dict[dict] = fail_create_dict,*backup_dict_loader[1:]
    data_name = "pickled_dict"
    hickle._dump(data, h5_data, data_name,memo = dump_memo,type_memo = create_memo)
    hickle._load(py_container, data_name, h5_data[data_name],memo = memo,type_memo = type_memo)
    assert py_container.convert() == data
    lookup.types_dict[dict] = backup_dict_loader
# %% ISSUE RELATED TESTS

def test_invalid_file():
    """ Test if trying to use a non-file object fails. """

    with pytest.raises(hickle.FileError):
        dump('test', ())


def test_binary_file(test_file_name):
    """ Test if using a binary file works

    https://github.com/telegraphic/hickle/issues/123"""

    filename = test_file_name.replace(".hkl",".hdf5")
    with open(filename, "w") as f:
        hickle.dump(None, f)

    with open(filename, "wb") as f:
        hickle.dump(None, f)


def test_file_open_close(test_file_name,h5_data):
    """ https://github.com/telegraphic/hickle/issues/20 """
    import h5py
    f = h5py.File(test_file_name.replace(".hkl",".hdf"), 'w')
    a = np.arange(5)

    dump(a, test_file_name)
    dump(a, test_file_name)

    dump(a, f, mode='w')
    f.close()
    with pytest.raises(hickle.ClosedFileError):
        dump(a, f, mode='w')
    h5_data.create_dataset('nothing',data=[])
    with pytest.raises(ValueError,match = r"Unable\s+to\s+create\s+group\s+\(name\s+already\s+exists\)"):
        dump(a,h5_data.file,path="/root_group")


def test_hdf5_group(test_file_name):
    import h5py
    hdf5_filename = test_file_name.replace(".hkl",".hdf5")
    file = h5py.File(hdf5_filename, 'w')
    group = file.create_group('test_group')
    a = np.arange(5)
    dump(a, group)
    file.close()

    a_hkl = load(hdf5_filename, path='/test_group')
    assert np.allclose(a_hkl, a)

    file = h5py.File(hdf5_filename, 'r+')
    group = file.create_group('test_group2')
    b = np.arange(8)

    dump(b, group, path='deeper/and_deeper')
    file.close()

    b_hkl = load(hdf5_filename, path='/test_group2/deeper/and_deeper')
    assert np.allclose(b_hkl, b)

    file = h5py.File(hdf5_filename, 'r')
    b_hkl2 = load(file['test_group2'], path='deeper/and_deeper')
    assert np.allclose(b_hkl2, b)
    file.close()



def test_with_open_file(test_file_name):
    """
    Testing dumping and loading to an open file

    https://github.com/telegraphic/hickle/issues/92"""

    lst = [1]
    tpl = (1,)
    dct = {1: 1}
    arr = np.array([1])

    with h5py.File(test_file_name, 'w') as file:
        dump(lst, file, path='/lst')
        dump(tpl, file, path='/tpl')
        dump(dct, file, path='/dct')
        dump(arr, file, path='/arr')

    with h5py.File(test_file_name, 'r') as file:
        assert load(file, '/lst') == lst
        assert load(file, '/tpl') == tpl
        assert load(file, '/dct') == dct
        assert load(file, '/arr') == arr


def test_load(test_file_name):
    a = set([1, 2, 3, 4])
    b = set([5, 6, 7, 8])
    c = set([9, 10, 11, 12])
    z = (a, b, c)
    z = [z, z]
    z = (z, z, z, z, z)

    print("Original:")
    pprint(z)
    dump(z, test_file_name, mode='w')

    print("\nReconstructed:")
    z = load(test_file_name)
    pprint(z)




def test_multi_hickle(test_file_name):
    """ Dumping to and loading from the same file several times

    https://github.com/telegraphic/hickle/issues/20"""

    a = {'a': 123, 'b': [1, 2, 4]}

    if os.path.exists(test_file_name):
        os.remove(test_file_name)
    dump(a, test_file_name, path="/test", mode="w")
    dump(a, test_file_name, path="/test2", mode="r+")
    dump(a, test_file_name, path="/test3", mode="r+")
    dump(a, test_file_name, path="/test4", mode="r+")

    load(test_file_name, path="/test")
    load(test_file_name, path="/test2")
    load(test_file_name, path="/test3")
    load(test_file_name, path="/test4")


def test_improper_attrs(test_file_name):
    """
    test for proper reporting missing mandatory attributes for the various
    supported file versions
    """

    # check that missing attributes which disallow to identify
    # hickle version are reported
    data = "my name? Ha I'm Nobody"
    dump(data,test_file_name)
    manipulated = h5py.File(test_file_name,"r+")
    root_group = manipulated.get('/')
    root_group.attrs["VERSION"] = root_group.attrs["HICKLE_VERSION"]
    del root_group.attrs["HICKLE_VERSION"]
    manipulated.flush()
    with pytest.raises(
        ValueError,
        match= r"Provided\s+argument\s+'file_obj'\s+does\s+not\s+appear"
               r"\s+to\s+be\s+a\s+valid\s+hickle\s+file!.*"
    ):
        load(manipulated)


# %% MAIN SCRIPT
if __name__ == '__main__':
    """ Some tests and examples """
    from _pytest.fixtures import FixtureRequest

    for h5_root,filename in (
        ( h5_data(request),test_file_name(request) )
        for request in (FixtureRequest(test_file_opener),)
    ):
        test_file_opener(h5_root,filename)
    for h5_root in h5_data(FixtureRequest(test_recursive_dump)):
        test_recursive_dump(h5_root)
    for h5_root in h5_data(FixtureRequest(test_recursive_load)):
        test_recursive_load(h5_root)
    test_invalid_file()
    for filename in test_file_name(FixtureRequest(test_binary_file)):
        test_binary_file(filename)
    for h5_root,filename in (
        ( h5_data(request),test_file_name(request) )
        for request in (FixtureRequest(test_file_open_close),)
    ):
        test_file_open_close(h5_root,filename)
    for filename in test_file_name(FixtureRequest(test_hdf5_group)):
        test_hdf5_group(filename)
    for filename in test_file_name(FixtureRequest(test_with_open_file)):
        test_with_open_file(filename)

    for filename in test_file_name(FixtureRequest(test_load)):
        test_load(filename)
    for filename in test_file_name(FixtureRequest(test_multi_hickle)):
        test_multi_hickle(filename)
    for filename in test_file_name(FixtureRequest(test_improper_attrs)):
        test_improper_attrs(filename)

