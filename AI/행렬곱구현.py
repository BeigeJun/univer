def make_arr():
    H = int(input("배열 높이 : "))
    W = int(input("배열 너비 : "))
    A = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            print("배열의","[",i,"]","[",j,"]", "요소 설정 : ",end= '')
            num = int(input(""))
            A[i][j] = num
    print(A)
    return A
def dot(arr1, arr2):
    if len(A[0]) == len(B):
        reselt = [[0] * len(arr2[0]) for _ in range(len(arr1))]
        for i in range(len(reselt)):
            for j in range(len(reselt[i])):
                for z in range(len(arr1[i])):
                    reselt[i][j] += arr1[i][z] * arr2[z][j]
        return reselt
    else:
        return "곱할 수 없는 행렬입니다."

A = make_arr()
B = make_arr()
print("행렬곱 결과 : ",dot(A,B))
