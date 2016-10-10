#!/usr/bin/env python
import sys, json
##from base64 import b64decode
##
# Load the data that PHP sent us
try:
    data = json.loads(sys.argv[1])
except:
    print "ERROR"
    sys.exit(1)

# Generate some data to send to PHP
result = data+10

# Send it to stdout (to PHP)
print json.dumps(result)

##import sys
##print sys.argv[1]
##print sys.argv[2]
