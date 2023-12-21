import pandas as pd
hollys = pd.read_csv("c:/Temp/할리스.csv", encoding='CP949',
                            index_col=0, header=0, engine='python')
addr = []
for address in hollys.address:
    addr.append(address.split())
for i in range(len(addr)):
    if addr[i][0]=='서울' : addr[i][0]='서울특별시'
    elif addr[i][0]=='서울시' : addr[i][0]='서울특별시'
    elif addr[i][0]=='제주' : addr[i][0]='제주특별자치도'  # 그 외는 생략 예:전남->전라남도, 경남->경상남도
addr2 = []
for i in range(len(addr)):
    if addr[i][0]=='서울' : addr[i][0]='서울특별시'
    elif addr[i][0]=='서울시' : addr[i][0]='서울특별시'
    elif addr[i][0]=='제주' : addr[i][0]='제주특별자치도'
    addr2.append(' '.join(addr[i]))

addr2 = pd.DataFrame(addr2, columns=['address2'])
hollys = pd.concat([hollys, addr2], axis=1)
hollys.to_csv('c:/Temp/할리스_주소.csv')

