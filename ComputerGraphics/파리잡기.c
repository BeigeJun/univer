#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <gl/glut.h>

GLfloat X = 0.0;
GLfloat Y = 0.0;
GLfloat tmp_x = 0.0;
GLfloat tmp_y = 0.0;
bool time_start = false;
GLint time_ = 100;
int R = 0.0;
int B = 0.0;
int G = 1.0;
void MyDisplay() {
    glClear(GL_COLOR_BUFFER_BIT);

    /////////////////////////////////
    ///////   그림 영역   ///////////
    /////////////////////////////////
    glBegin(GL_POLYGON);                  //다각형

    glColor3f(R, G, B);
    glVertex3f((X - 0.1), (Y + 0.1), 0.0);
    glVertex3f((X + 0.1), (Y + 0.1), 0.0);
    glVertex3f((X + 0.1), (Y - 0.1), 0.0);
    glVertex3f((X - 0.1), (Y - 0.1), 0.0);

    glEnd();
    glutSwapBuffers();
}


void MyTimer(int Value) {

    X = (float)rand() / RAND_MAX - 0.5;
    Y = (float)rand() / RAND_MAX - 0.5;
    printf("X : %.2f, Y : %.2f\n", X, Y);
    R = 0.0;
    G = 1.0;
    B = 0.0;
    glutPostRedisplay();
    glutTimerFunc(time_, MyTimer, 3);   //1/1000초

}

void MyMainMenu(int entryID) {
    if (entryID == 1)
    {
        glutTimerFunc(time_, MyTimer, 3);   //1/1000초기준, 함수, 타이머I
    }
    else if (entryID == 2)
    {
        exit(0);
    }
    else if (entryID == 3)
    {
        time_ = 100;
    }
    else if (entryID == 4)
    {
        time_ = 500;
    }
    else if (entryID == 5)
    {
        time_ = 1000;
    }
    glutPostRedisplay(); //인바이어렉 같은느낌
}



void MyMouseClick(GLint Button, GLint State, GLint Mouse_X, GLint Mouse_Y) {
    if (Button == GLUT_LEFT_BUTTON && State == GLUT_DOWN) {
        printf("(%d,%d)\n", Mouse_X, Mouse_Y);
        tmp_x = ((GLfloat)Mouse_X / 300.0 - 0.5);
        tmp_y = ((GLfloat)(300 - Mouse_Y) / 300.0 - 0.5);                     //일반좌표계와 gl좌표계의 Y좌표가 반대이기때문에 꼭 !!!!!
        printf("(%f,%f)\n", tmp_x, tmp_y);
        if ((tmp_x >= X - 0.25 && tmp_x <= X + 0.25) && (tmp_y >= Y - 0.25 && tmp_y <= Y + 0.25)) {
            R = 1.0;
            G = 0.0;
            B = 0.0;
            printf("잡았다.\n");
        }
    }

}
int main(int argc, char** argv) {
    /////////////////////////////////
    /////// 윈도우 초기화 ///////////
    /////////////////////////////////
    glutInit(&argc, argv);                     //glut초기화
    glutInitWindowPosition(0, 0);               //윈도우 위치 초기화
    glutInitWindowSize(300, 300);               //윈도우 크기 지정         
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);   //윈도우 버퍼 수, 종류 및 색상 모드 지정
    glutCreateWindow("OpenGL Display CallBack");   //윈도우 생성

    /////////////////////////////////
    /////// OpenGL 초기화 ///////////
    /////////////////////////////////

    glClearColor(1.0, 1.0, 1.0, 1.0);      //윈도우 배경색 지정 RGBA
    glMatrixMode(GL_PROJECTION);         //파이프라인에 쓰이는 Matrix설정
    glLoadIdentity();               //Matrix 초기화(단위행렬 지정)
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);   //무대 설정(그림 영역 설정 glvertex3f에서 1 ~ -1로 지정한 이유)

    /////////////////////////////////
    ///////  CallBack등록 ///////////
    /////////////////////////////////

    glutDisplayFunc(MyDisplay);   //CallBack 함수 등록(Display)
    GLint MyMainMenuID = glutCreateMenu(MyMainMenu);
    glutAddMenuEntry("start", 1); //메뉴추가
    glutAddMenuEntry("end", 2);
    glutAddMenuEntry("0.1", 3);
    glutAddMenuEntry("0.5", 4);
    glutAddMenuEntry("1.0", 5);
    glutMouseFunc(MyMouseClick);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutMainLoop();                  //Window 메시지 루프
    return 0;
}
