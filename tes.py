import sys
import json

i = 3

s = "c\\n:usernait"
print(s)
# s = json.dumps(s)
# print(f'dumps {s}')
with open("output.json", "w") as f:
    s = json.loads(s)
    s = json.dump(s, f)
print(s)

