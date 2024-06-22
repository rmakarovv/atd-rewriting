import requests
from readability import Document
from bs4 import BeautifulSoup

url = "https://media.innopolis.university/news/Free-course-for-engineering-students-IU-2024/"

response = requests.get(url)
doc = Document(response.content)
summary = doc.summary()

soup = BeautifulSoup(summary, 'html.parser')
main_text = soup.get_text(separator=' ', strip=True)

print(main_text)
