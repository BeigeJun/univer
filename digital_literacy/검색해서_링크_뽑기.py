# 네이버 검색 API 예제 - 블로그 검색
import os
import sys
import urllib.request
import json
client_id = "N3dZeWiGwlDzZLwbNcTk"
client_secret = "N3dZeWiGwlDzZLwbNcTk"
encText = urllib.parse.quote("검색할 단어")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    body = json.load(response_body.decode('utf-8'))
    total = body['total']
    print(total)
    for item in body['items]
else:
    print("Error Code:" + rescode)
