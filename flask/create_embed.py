import requests

# File and endpoint setup
url = "http://127.0.0.1:7888/upload"
file_path = "Online Quran Academy.pdf"

# Send POST request with the file
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files)

# Print the server response
print(response.status_code)
print(response.json())
