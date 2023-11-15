
f = open("test.txt","r") #텍스트 불러오기
t = f.read() #t에 읽은 내용 저장
f.close() #파일 닫기
txt =t.lower() #읽어온 텍스트 전부 소문자로 변환

word_count = dict() #값을 저장할 딕션어리 생성
txt_word = list(txt) #읽어온 내용 한글자씩 잘라서 리스트 만들기

all_count = 0
for word in txt_word:
    if ord(word) >= 97 and ord(word) <=122: #글자만 읽어드리기
        if word in word_count:  #딕셔너리 키에 문자가 존재하면 +1 아니면 새로 생성하기
            word_count[word] = word_count[word] + 1
        else :
            word_count[word] = 1
            all_count = all_count + 1

sorted_word = sorted(word_count.items(),key=lambda x:x[1])

save = ""
for w in range(all_count):
    save = save + sorted_word[w][0] + ':' + str(sorted_word[w][1]) + '\n'

f = open("test_save.txt","w")
f.write(save)
f.close()
#------------------------------키값만들기----------------------------------------------------------
all_len = len(txt_word)
save2 = ""

for x in range(all_len):
    if ord(txt_word[x]) >= 97 and ord(txt_word[x]) <=122:
        save2 = save2 + sorted_word[ord(txt_word[x][0])-97][0]
    else:
        save2 = save2 + txt_word[x]
f = open("test_remake.txt","w")
f.write(save2)
f.close()
#--------------------------암호화하기--------------------------------------------------------------
f = open("key.txt","r") #텍스트 불러오기
t = f.read() #t에 읽은 내용 저장
f.close() #파일 닫기
key_s = []
for i in range(len(t)):
    if ord(t[i]) >= 97 and ord(t[i]) <= 122:
        key_s.append(t[i])
print(key_s)
#-------------------------키값만 빼서 리스트 만들기------------------------------------------------------
save3 =""

f = open("test_remake.txt","r")
t = f.read()
f.close()
all_len3 = len(t)
txt_word = list(t)

for x in range(all_len3):
    if ord(txt_word[x]) >= 97 and ord(txt_word[x]) <=122:
        i = 0
        for i in range(26):
            if key_s[i] == txt_word[x]:
                break
        save3 = save3 + chr(97+i)
    else:
        save3 = save3 + txt_word[x]
#-------------------------------복호화하기---------------------------------------------------
f = open("test_final.txt","w")
f.write(save3)
f.close()
#--------------------------------복호화파일저장---------------------------------------------------
