import ujson as json
samples = 5000
with open('dataset/IR_data_news_12k.json', 'r') as f:
	data = json.load(f)
	short_data = {}
	count = 0
	for key, value in data.items():
		short_data[key] = value
		count += 1
		if count > samples:
			break
	with open("dataset/data_" + str(samples) + ".json", "w") as nf:
		json.dump(short_data, nf)
