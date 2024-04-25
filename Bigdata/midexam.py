print("%100d" % 100)
# %7.1 -> 이건 7자리를 출력하는데 .도 한자리를 차지한다 소숫점 첫번째 자리까지 출력
# 소수는 무조건 6자리까지 출력이 디폴트값
print("{2:d} {1:d} {0:d}".format(100,200,300))
#이건 {}안에 첫자리가 포멧 번째 수고 : 옆이 형식
# /' -> '입력
# /" -> "입력
# \t -> 탭
a = 10
print(type(a))
A = [10,20,30]
if a in A:
    print("ㅇㅇ")

import random
number = []
number.append(random.randrange(0,10))
#0에서 10까지 랜덤한 숫자 생성

# a = int(input("넣어라"))
# b = int(input("넣어라"))
# c = int(input("넣어라"))
# sum = 0
# for i in range(a,b+c,c):
#     sum += i
#     print(i)
# print(sum)

print("%04d" % 876)
print("%5s" % "cookbook")
print("%.1f" % 123.45)
# 3.14e5 = 3.14*10^5
num = 100
num +=1
num -=1
num *=1
num /=1
num = int(num)
print(num)

num = 5
res = "짝수" if num % 2 == 0 else "홀수"
print(res)
sum =0
for i in range(0,101,1):
    sum+=1
print(sum)

for i in range(5,-1,-1):
    print(i)

while(1>5):
    print("a")
sun = 0
for i in range(3333,9999,1):
    if i % 1234 != 0:
        if sum < 100000:
            sum += i
            print(i)
            continue
    else:
        break
print(sum)

for i in range(3,101,1):
    correct = 0
    for j in range(i,1,-1):
        if i % j == 0:
            correct += 1
            # print(i,j)
    if correct == 1:
        print(i, end=' ')

# 2 4 3 6
m = [[n*m for n in range(1,3)] for m in range(2,4)]
print(m)

# ss = 'it_cookbook_python'
# print(ss[0])
# print(ss[0:-1])
# for i in range(0,len(ss)):
#     if i % 2 == 0:
#         print(ss[-i-1],end=' ')
#     else:
#         print('#',end=' ')

a = "나의 이름은 서준 이라고 한다 서준"
print(a.find("서준"))
