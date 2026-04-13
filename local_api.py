import requests

url = "http://127.0.0.1:8000"

r_get = requests.get(url)

print(f"Status Code: {r_get.status_code}")

print(f"Result: {r_get.json()['message']}")


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r_post = requests.post(f"{url}/data/", json=data)

print(f"Status Code: {r_post.status_code}")

print(f"Result: {r_post.json()['result']}")
