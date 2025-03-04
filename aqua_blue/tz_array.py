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
    
    """
    tz: datetime.tzinfo
    """ 
    Store the timezone information for the array. Default is None.
    """
    np_offset: np.timedelta64
    """
    Store the timezone offset for the array. Default is None.
    """
    direction: int
    """
    Store the direction of the timezone offset for the array. Default is None.
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
        
        tc_offset = tz.utcoffset(input_array[0])
        np_offset = np.timedelta64(int(np.abs(tc_offset.total_seconds())), 's')

        if(tc_offset.total_seconds() < 0):
            datetime64_array += np_offset
        else:
            datetime64_array -= np_offset
        
        obj = super().__new__(cls, datetime64_array.shape, dtype, buffer, offset, strides, order)
        obj[:] = datetime64_array
        obj.tz = tz
        obj.np_offset = np_offset
        if(tc_offset.total_seconds() < 0):
            obj.direction = -1
        else:
            obj.direction = 1
        
        return obj
    
    def __setitem__(self, index, value): 
        if( not isinstance(value, datetime.datetime)):
            raise ValueError("All elements must be datetime objects.")
        
        super().__set__item(index, value.astimezone(datetime.timezone.utc)) 
        
    def __getitem__(self, index)->datetime.datetime: 
        out = super().__getitem(index)
        return out.astimezone(self.tz)
    
    def tolist(self):
        """ 
        Convert the array back to a list of datetime objects with timezone information
        """
        local = self.copy() 
        self += self.np_offset * self.direction
        utc_list = super().tolist()
        out =  [dt.astimezone(self.tz) for dt in utc_list]
        self = local 
        return out 
    
def fromNDArray(self, arr: NDArray, tz: datetime.tzinfo) -> TZArray: 
    """ 
    Convert a numpy array to a TZArray instance
    
    Args:
        arr: numpy array to be converted
        tz: timezone information for the array  
    """
    
    datetime_array = arr.tolist() 
    datetime_array = [dt.astimezone(tz) for dt in datetime_array]
    return TZArray(datetime_array)

def fromFile(filename: Union[IO, str, Path], tz: datetime.tzinfo) -> TZArray:
    """ 
    Load a text file and convert it to a TZArray instance
    
    Args:
        filename: The file-like object, path name, or Path in which to read
    """
    data = np.loadtxt(filename, dtype='datetime64[s]')
    return fromNDArray(data, tz)


# Create a list of datetime objects with timezone information using zoneinfo
input_array = [
    datetime.datetime(2025, 3, 3, 12, 0, 0, tzinfo=ZoneInfo('America/Los_Angeles')),
    datetime.datetime(2025, 3, 4, 12, 0, 0, tzinfo=ZoneInfo('America/Los_Angeles')),
    datetime.datetime(2025, 3, 5, 12, 0, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
]

# Create a TZArray instance
tz_array = TZArray(input_array)

print(tz_array)
print(tz_array.tz)
print(tz_array.tolist())
test = np.loadtxt('output.txt', dtype='datetime64[s]')
print(test)