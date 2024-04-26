dict ={}

for j in range(2):
    if j == 0:
        fruit=['사과','배','파인애플','포도']
        price=[5000,7000,4500,6000]
    else:
        fruit.append("바나나")
        price.append(3500)

    for i in range(len(fruit)):
        dict[fruit[i]] = price[i]

    print("과일 리스트 :",fruit)
    print("가격 리스트 :",price)
    print("과일, 가격 딕셔너리 :",dict)
