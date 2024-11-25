import requests
from bs4 import BeautifulSoup
import json

url = "https://katex.org/docs/supported.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table")
# print(tables)

for idx, table in enumerate(tables):
    code_elements = table.find_all("code")
    code_strings = [element.string for element in code_elements]
    with open(f"katex_tokens/{str(idx)}.txt", "w", encoding="utf-8") as f:
        for code in code_strings:
            if not code.isascii():
                print(code)
            f.write(code + "\n")
