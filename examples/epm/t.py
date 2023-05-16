import shlex
import json
with open('./start.sh') as f:
    s = f.read()
    print(json.dumps(shlex.split(s)[2:]))

    