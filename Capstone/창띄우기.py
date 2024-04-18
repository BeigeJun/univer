import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont

colors = [[255, 0, 0], [0, 0, 255]]
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        self.setWindowTitle('POS')
        self.move(1500, 300)
        self.resize(230, 300)
        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_rectangles(qp)
        self.draw_text(qp)
        qp.end()

    def draw_rectangles(self, qp):
        col = QColor(0, 0, 0)
        qp.setPen(col)

        x = 20
        y = 20
        width = 50
        height = 50
        distance = 20
        color = 'Red'
        for _ in range(4):
            for _ in range(3):
                qp.drawRect(x, y, width, height)
                qp.fillRect(x, y, width, height, QColor(colors[0][0], colors[0][1], colors[0][2]))
                x += width + distance
            x = 20
            y += height + distance

    def draw_text(self, qp):
        qp.setPen(QColor(0, 0, 0))
        qp.setFont(QFont('Decorative', 10))
        text = '목 상태'
        qp.drawText(90, 15, text)
        text = '어깨 위치 상태'
        qp.drawText(75, 87, text)
        text = '어깨 기울기 상태'
        qp.drawText(70, 157, text)
        text = '상체 위치 상태'
        qp.drawText(70, 227, text)

        text = '앞            정상            뒤'
        qp.drawText(40, 51, text)
        text = '좌            정상            우'
        qp.drawText(40, 121, text)
        text = '좌            정상            우'
        qp.drawText(40, 191, text)
        text = '좌            정상            우'
        qp.drawText(40, 261, text)


app = QApplication(sys.argv)
ex = MyApp()
sys.exit(app.exec_())
