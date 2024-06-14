from tkinter import *
from tkinter.filedialog import *
import csv

def makeEmptySheet(row, col):
    retList = []

    for i in range(0, row):
        tmpList = []

        for k in range(0, col):
            ent = Entry(window, text='', width=10, font=("맑은고딕", 20))
            ent.grid(row=i, column=k)
            tmpList.append(ent)

        retList.append(tmpList)

    return retList
def func_open():
    saveFp = asksaveasfilename(parent=window, defaultextension=".csv",
                                  filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    with open(saveFp, "r") as inFp:
        csvReader = csv.reader(inFp)
        header_list = next(csvReader)

        idx = header_list.index("인원")

        csvList.append(header_list)

        for row_list in csvReader:
            csvList.append(row_list)

    rowNum = len(csvList)
    colNum = len(csvList[0])
    workSheet = makeEmptySheet(rowNum, colNum)

    for i in range(0, rowNum):
        for k in range(0, colNum):
            if (csvList[i][idx].isnumeric()):
                if (int(csvList[i][idx]) >= 7):
                    ent = workSheet[i][idx]
                    ent.configure(bg='magenta')

            workSheet[i][k].insert(0, csvList[i][k])


def func_exit():
    window.quit()
    window.destroy()

window = Tk()
csvList = []
rowNum, colNum = 0, 0
workSheet = []
idx = 0
mainmenu = Menu(window)
window.config(menu = mainmenu)

fileMenu = Menu(mainmenu)
mainmenu.add_cascade(label= "파일", menu = fileMenu)
fileMenu.add_command(label="열기", command=func_open)
fileMenu.add_separator()

fileMenu.add_command(label="종료", command=func_exit)

window.mainloop()
