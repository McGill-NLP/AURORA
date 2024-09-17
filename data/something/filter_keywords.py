import json

train = json.load(open('train.json'))
old_len = len(train)
new_train = []

filter_words = ['pretend', 'holding', 'fail', 'roll', 'then letting it', 'until it falls down', 'spin', 'nothing happens', 'show', 'falling like', 'squeezing', 'throwing', 'slightly']
# these words should not be in the description

for item in train:
    if not any(word in item['instruction'] for word in filter_words):
        if "falls off" in item['instruction']:
            if "but" in item['instruction']:
                new_train.append(item)  # Includes "falls off" but also has "but"
        else:
            new_train.append(item)  # Does not contain "falls off"
            
json.dump(new_train, open('train_.json', 'w'), indent=4)
new_len = len(new_train)

print(f'Old length: {old_len}, New length: {new_len}')
