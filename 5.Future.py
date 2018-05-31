from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# With __future__ module's inclusion,
# you can slowly be accustomed to features/keywords fucionality from future python versions

import sys, os, time
import string #absolute_import

for x in range(0,10):
    print("Hello", "World",x , sep='----', end='\n\n')  # this type of print supported python 3, 
    #but with imported future print_function this statement is functional in older python version too
    time.sleep(1)

print(8/7)  # prints 1.1428571428571428
print(8//7) # prints 1



print(string.ascii_uppercase) #string
