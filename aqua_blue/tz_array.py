import numpy as np
import datetime
from typing import List
from zoneinfo import ZoneInfo
from numpy.typing import NDArray
from typing import IO, Union
from pathlib import Path
from dataclasses import dataclass, field

class TZArray(np.ndarray):
    """
    Timezone-Aware Wrapper for NumPy Arrays. 
    Timezone awareness is a deprecated NumPy feature due to the deprecation of pytz. 
    This subclass provides a workaround for this issue by storing the timezone information in the array.
    The datetime objects are stored in UTC time and converted to the specified timezone when accessed.
    This is a very simple implementation that works for 1 dimensional arrays. It is meant to satisfy our datetime processing 
    requirements, not for timezone NumPy integration in general. 

    """
    tz: datetime.tzinfo = ZoneInfo('UTC')
    """ 
    Store the timezone information for the array. Default is None.
    """
    
    def __new__(cls, input_array: List[datetime.datetime], dtype='datetime64[s]', buffer=None, offset=0, strides=None, order=None):
        # Store the timezone information of the first element - this means that all elements must belong to the same timezone.

        try:
            tz = ZoneInfo(input_array[0].tzinfo.zone)
        except AttributeError:
            tz = input_array[0].tzinfo
        
        for dt in input_array:
            try:
                current_tz = ZoneInfo(dt.tzinfo.zone)
            except AttributeError:
                current_tz = dt.tzinfo
            if current_tz != tz:
                raise ValueError("All elements must belong to the same timezone.")
        
        # Purge the timezone information from the datetime objects
        naive_array = [dt.replace(tzinfo=None) for dt in input_array]
        datetime64_array = np.array([np.datetime64(dt.isoformat()) for dt in naive_array], dtype=dtype)
        
        if tz is not None:
            tc_offset = tz.utcoffset(input_array[0])
            np_offset = np.timedelta64(int(np.abs(tc_offset.total_seconds())), 's')
            
            if(tc_offset.total_seconds() < 0):
                datetime64_array += np_offset
            else:
                datetime64_array -= np_offset
        
        obj = super().__new__(cls, datetime64_array.shape, dtype, buffer, offset, strides, order)
        obj[:] = datetime64_array

        if tz is not None:
            obj.tz = tz
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.tz = getattr(obj, 'tz', ZoneInfo('UTC'))
    
        
    def __repr__(self):
        """ 
        Update the representation of the array to include the timezone information
        """
        return f"TZArray({super().__repr__()}, tz={self.tz})"
    
    def __eq__(self, other):
        return (super().__eq__(other)).all() and self.tz == other.tz
    
    def copy(self, order='C'):
        return self.view(type(self)).__array_finalize__(self)
    
    def tolist(self):
        """ 
        Convert the array back to a list of datetime objects with timezone information
        """
        utc_list = super().tolist()
        utc_list = [date.replace(tzinfo=ZoneInfo('UTC')) for date in utc_list]
        out =  [dt.astimezone(self.tz) for dt in utc_list]
        return out 
    
    def toFile(self, filename: Union[IO, str, Path], tz: datetime.tzinfo=ZoneInfo("UTC")):
        """ 
        Save a TZArray instance to a text file
        
        Args:
            self: TZArray instance to be saved
            filename: The file-like object, path name, or Path in which to save
            tz: Timezone information to write the data in
        """
        arr = self.tolist() 
        arr = [dt.astimezone(tz) for dt in arr]
        arr = [dt.replace(tzinfo=None) for dt in arr]
        np.savetxt(filename, np.array(arr), fmt='%s')

def fromNDArray(arr: NDArray, tz: datetime.tzinfo=ZoneInfo("UTC")) -> TZArray: 
    """ 
    Convert a numpy array to a TZArray instance
    
    Args:
        arr: numpy array to be converted
        tz: timezone information that the original array is in
    """
    
    datetime_array = arr.tolist() 
    datetime_array = [dt.replace(tzinfo=tz) for dt in datetime_array]
    
    return TZArray(datetime_array)

def fromFile(filename: Union[IO, str, Path], tz: datetime.tzinfo=ZoneInfo('UTC')) -> TZArray:
    """ 
    Load a text file and convert it to a TZArray instance
    
    Args:
        filename: The file-like object, path name, or Path in which to read
        tz: Timezone information that the original array is in
    """
    data = np.loadtxt(filename, dtype='datetime64[s]')
    return fromNDArray(data, tz)