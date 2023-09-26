class visnere:
    def __init__(self,pwd,key):
        self.password = pwd
        self.key = key
        self.PW = pwd
        self.password = self.password.upper()
        self.key = self.key.upper()
        self.password = self.password.replace(' ','')
        self.key = self.key.replace(' ','')

    def makeFullkey(self):
        self.Fullkey = ""
        a = 0
        for i in range(0,len(self.password)):
            if a == len(self.key):
                a = 0
            self.Fullkey+=self.key[a]
            a = a+1


    def makeTable(self,One_pwd):
        asci_pwd = ord(One_pwd)
        table = ""
        count = 1
        count1 = 0
        x=0
        y=0
        for x in range(0,26):
            if asci_pwd+count == 91:
                continue
            table += chr(asci_pwd+count)
            count = count +1
        for y in range(0,27-count):
            table += chr(65+count1)
            count1 = count1 + 1
        self.table = table

    def Make_real_password(self):
        base_key = "BCDEFGHIJKLMNOPQRSTUVWXYZA"
        self.real_password = ""
        for z in range(0,len(self.password)):
            one_pwd = self.password[z]
            one_key = self.Fullkey[z]
            self.makeTable(one_pwd)
            for r in range(0,26):
                if one_key == base_key[r]:
                    key_num = r
            self.real_password +=self.table[key_num]
        print(self.real_password)

    def back(self):
        base_key1 = "BCDEFGHIJKLMNOPQRSTUVWXYZA"
        self.back_password = ""
        for c in range(0,len(self.PW)):
            back_pwd = self.PW[c]
            back_key = self.Fullkey[c]
            self.makeTable(back_key)
            back_key_num = 0
            for s in range(0,26):
                if self.table[s] == back_pwd:
                    back_key_num = s
            self.back_password += base_key1[back_key_num]
        print(self.back_password)

while 1:
    what = int(input("1. 암호화 2. 복호화 3. 나가기 :"))
    if(what == 1):
        P = input("암호화 할 문장 : ")
        K = input("키 : ")
        a = visnere(P,K)
        a.makeFullkey()
        a.Make_real_password()
    elif(what == 2):
        B = input("복호화 할 문장 : ")
        k = input("키 값 : ")
        b = visnere(B,k)
        b.makeFullkey()
        b.back()
    elif(what == 3):
        print("종료합니다.")
        break
    else:
        print("잘못된 입력입니다. 다시입력하십시오.")
