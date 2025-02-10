import json

with open("C:/Users/reddyk6780/Desktop/CQLSTM/config/CQLSTM.jsonnet", "r") as f:
    data = f.read()
    json.loads(data)  # This will throw an error if the JSON is invalid
