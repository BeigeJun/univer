#include <gl/glut.h>
#include <gl/gl.h>                     
#include <gl/glu.h>
#include <stdio.h>
#include <math.h>

GLfloat X = 0.0;
GLfloat Y = 0.0;
GLfloat DirX = 0.01;
GLfloat DirY = -0.01;
GLfloat tmp_x = 0.0;
GLfloat tmp_y = 0.0;
GLfloat blue = 1.0;
GLfloat red = 0.0;

int flag = 1; // 초기 값은 1

void MyDisplay() {
   glClear(GL_COLOR_BUFFER_BIT);

   /////////////////////////////////
   ///////   그림 영역   ///////////
   /////////////////////////////////
   glBegin(GL_POLYGON);                  //다각형

   glColor3f(red, 0.0, blue);
   glVertex3f((X - 0.035), (Y + 0.035), 0.0);
   glVertex3f((X + 0.035), (Y + 0.035), 0.0);
   glVertex3f((X + 0.035), (Y - 0.035), 0.0);
   glVertex3f((X - 0.035), (Y - 0.035), 0.0);

   glEnd();
   glutSwapBuffers();
}

void MyTimer(int Value) {

   X = X + DirX;
   Y = Y + DirY;


   if (X > 0.95)
      DirX = -0.03;                     // + -
   else if (X < -0.95)
      DirX = 0.03;                     // - -
   else if (Y < -0.93)
      DirY = 0.02;                     // - +
   else if (Y > 0.93)      
      DirY = -0.02;                     // + -


   red = 0.0;
   blue = 1.0;
   

   glutPostRedisplay();
   glutTimerFunc(1000, MyTimer, 1);   //1/1000초, 타이머 콜백은 한번만 실행되므로 매번 타이머 함수에서 실행 해줘야함!

}

void MyMouseClick(GLint Button, GLint State, GLint Mouse_X, GLint Mouse_Y) {
   if (Button == GLUT_LEFT_BUTTON && State == GLUT_DOWN) {

      tmp_x = 2.0 * ((GLfloat)Mouse_X / 300.0 - 0.5);
      tmp_y = 2.0 * ((GLfloat)(300 - Mouse_Y) / 300.0 - 0.5);                     //일반좌표계와 gl좌표계의 Y좌표가 반대이기때문에 꼭 !!!!!

      if ((tmp_x >= X - 0.035 && tmp_x <= X + 0.035) && (tmp_y >= Y - 0.035 && tmp_y <= Y + 0.035)) {
         printf("잡았다.\n");
         red = 1.0;
         blue = 0.0;

      }
      else {
         red = 0.0;
         blue = 1.0;
         printf("못잡았다\n");
         printf("tmp_x : %f, tmp_y : %f, X : %f, Y : %f\n", tmp_x, tmp_y, X, Y);
      }
   }

}

void MyKeyboard(unsigned char KeyPressed, int XX, int XY) {
   switch (KeyPressed) {
   case 'r':
      X = (float)rand() / RAND_MAX - 0.5;
      Y = (float)rand() / RAND_MAX - 0.5;
      break;
   }
   glutPostRedisplay();
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
   glMatrixMode(GL_PROJECTION);      //파이프라인에 쓰이는 Matrix설정
   glLoadIdentity();            //Matrix 초기화(단위행렬 지정)
   glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);   //무대 설정(그림 영역 설정 glvertex3f에서 1 ~ -1로 지정한 이유)



   /////////////////////////////////
   ///////  CallBack등록 ///////////
   /////////////////////////////////

   glutDisplayFunc(MyDisplay);   //CallBack 함수 등록(Display)
   glutTimerFunc(40, MyTimer, 3);   //1/1000초기준, 함수, 타이머ID
   glutMouseFunc(MyMouseClick);
   glutKeyboardFunc(MyKeyboard);

   /////////////////////////////////
   ///////  MainLoop     ///////////
   /////////////////////////////////

   glutMainLoop();                  //Window 메시지 루프
   return 0;


}
