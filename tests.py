import json

s = {"source_string": "Hello 34 I'm 233 years old","regex": "(\d+)","replacement": "NUMBERS"}

print(json.loads(s))
