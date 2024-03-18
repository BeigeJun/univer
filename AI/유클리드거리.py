import numpy as np

def make_arr():
    H = int(input("배열 높이 : "))
    W = int(input("배열 너비 : "))
    A = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            print("배열의","[",i,"]","[",j,"]", "요소 설정 : ",end= '')
            num = int(input(""))
            A[i][j] = num
    print("생성한 배열")
    print(A)
    return A, H, W
def SQ_Dist(A,B,row,col):
  sum = 0
  for i in range(row):
    for j in range(col):
      sum += (A[i][j]-B[i][j])**2
  return sum
a = [[1,2,3],
     [4,5,6],
     [7,8,9]]

b = [[10,11,12],
     [13,14,15],
     [16,17,18]]

row = 3
col = 3
print(SQ_Dist(a, b, row, col))
