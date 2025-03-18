import json

from chimera.utils import parse_times_file

with open("time.json") as fp:
    json.dump(parse_times_file(), fp)
