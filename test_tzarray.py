from aqua_blue.tz_array import * 
import numpy as np 
from typing import List
from zoneinfo import ZoneInfo

#####################################################################
#                          TZARRAY CREATION                         # 
#####################################################################

# From List[datetime.datetime] 
datetime_list = [datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo = ZoneInfo("America/New_York")),
                 datetime.datetime(2021, 1, 2, 1, 0, 0, tzinfo = ZoneInfo("America/New_York")),
                 datetime.datetime(2021, 1, 3, 2, 0, 0, tzinfo = ZoneInfo("America/New_York"))]  

tz_array_from_list = TZArray(datetime_list)
print(f'TZArray Initialized from List[datetime.datetime]: {tz_array_from_list}\n')

# From NDArray[datetime64] 
datetime_ndarray = np.arange(np.datetime64('2021-01-01T00:00:00'), np.datetime64('2021-01-04T00:00:00'), np.timedelta64(1, 'D'), dtype = 'datetime64[s]')

tz_array_from_ndarray = fromNDArray(datetime_ndarray, ZoneInfo("America/New_York"))
print(f'TZArray Initialized from NDArray[datetime64]: {tz_array_from_ndarray}\n')

# From File
tz_array_from_file = fromFile('data.txt', ZoneInfo("America/New_York"))
print(f'TZArray Initialized from File: {tz_array_from_file}\n')

#####################################################################
#                          TZARRAY OPERATIONS                       # 
#####################################################################

# Write to File 
tz_array_from_list.toFile('data.txt', ZoneInfo("America/New_York"))
print(f'TZArray Written to File: {tz_array_from_list}\n')
