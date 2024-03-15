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
def mtx_mul(arr1, arr1_H, arr1_W, arr2, arr2_H, arr2_W):
    if arr1_H == arr2_W:
        reselt = [[0] * arr2_W for _ in range(arr1_H)]
        for i in range(len(reselt)):
            for j in range(len(reselt[i])):
                for z in range(arr1_W):
                    reselt[i][j] += arr1[i][z] * arr2[z][j]
        return reselt
    else:
        return "곱할 수 없는 행렬입니다."

A, A_H, A_W = make_arr()
B, B_H, B_W = make_arr()
print("행렬곱 결과 : ")
C = mtx_mul(A,A_H,A_W,B,B_H,B_W)
for i in range(len(C)):
  print(C[i])
