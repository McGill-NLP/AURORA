import json

with open('3d_data/GSO_dict_filtered.json', 'r') as file:
    data = json.load(file)

# output_json = f"3d_data/GSO_dict.json"
# with open(output_json, "w") as response_file:
#     keys = list(data["assets"].keys())
#     key_to_key_dict = {key: key for key in keys}

#     json.dump(key_to_key_dict, response_file, indent=4)


exchanged_dict = {}
for key, values in data.items():
    for value in values:
        exchanged_dict[value] = key


output_json = f"3d_data/GSO_dict.json"
with open(output_json, "w") as response_file:
    json.dump(exchanged_dict, response_file, indent=4)