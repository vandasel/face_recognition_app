from gradio_client import Client, handle_file
import os
import numpy as np
import time as t
import json
paths = []


np.random.seed(10)

for dirpath, dirnames, filenames in os.walk("/workspaces/face_recognition_app/dataset"):
    for filename in filenames:
        paths.append(os.path.join(dirpath, filename))
        np.random.shuffle(paths)

test_paths = paths[0:100]

client = Client("http://127.0.0.1:7860/")

sizes = []
times = []
modeltimes = []
dbtimes = []
i = 0
for path in test_paths:
    sizeof = os.path.getsize(path)
    start = t.time()
    result,modelt,dbt = client.predict(
            input_img=handle_file(path),
            api_name="/predict"
    )
    stop = t.time() 
    modeltimes.append(modelt)
    dbtimes.append(dbt)
    times.append(stop-start)
    sizes.append(sizeof)
    i+=1
    print(i)

result = {
    "sizes" : sizes,
    "times" : times,
    "timemodel" : modeltimes,
    "timedb" : dbtimes,
    "average_time" : sum(times)/len(times),
    "average_size" : sum(sizes)/len(sizes)
}

json_object = json.dumps(result, indent = 4)
with open("efficiency.json", "w") as outfile:
    outfile.write(json_object)
    outfile.close() 

print()