
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

sorted_word = sorted(word_count.items(),key=lambda x:x[1],reverse=True)


save = ""
for w in range(all_count):
    save = save + sorted_word[w][0] + ':' + str(sorted_word[w][1]) + '\n'

f = open("test_save.txt","w")
f.write(save)
f.close()

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

