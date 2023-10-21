//#include <gl\GL.h>
//#include <gl\GLU.h>

#include <gl/glut.h>
#include <gl/gl.h>                     
#include <gl/glu.h>
#include <stdlib.h>
#include <stdio.h>
GLint IsStart = 0;
GLfloat where[6] = { 0,0,0,0,0,0 };
GLint stop[6] = { 0,0,0,0,0,0 };
GLfloat tmp_x;
GLfloat tmp_y;
GLfloat move = 0.1;
int random = 100;
void MyDisplay()               //디스플레이 콜백함수   
{
   glClear(GL_COLOR_BUFFER_BIT);   //GL 상태변수 설정
   glViewport(0, 0, 640, 480);

   glColor3f(0.0, 0.0, 1.0);
   for (int i = 0; i < 6; i++)
   {
      if (stop[i] == 1)
      {
         glColor3f(1.0, 0.0, 0.0);
      }
      else
      {
         glColor3f(0.0, 0.0, 1.0);
      }
      glBegin(GL_POLYGON);              //입력 기본요소 정의   
      glVertex3f(0.0 + where[i], 0.9 - (0.18 * i), 0.0);   // 좌측 하단
      glVertex3f(0.1 + where[i], 0.9 - (0.18 * i), 0.0);   // 우측 하단
      glVertex3f(0.1 + where[i], 1.0 - (0.18 * i), 0.0); // 우측 상단
      glVertex3f(0.0 + where[i], 1.0 - (0.18 * i), 0.0); // 좌측 상단 
      glEnd(); // 다각형을 그릴때 반 시계 방향으로 그릴 것
   }
   glFlush();          // 그림을 모니터로 쏴줌   
}

void MyTimer(int Value) // Value : 알람 번호    // WIN32 API 와는 다르게 타이머가 한번 울리고 끝나므로 계속 선언을 해주어야함
{
   random = rand() % 6;
   if (stop[random] == 0)
   {
      where[random] = where[random] + move;
   }
   for (int i = 0; i < 6; i++)
   {
      if (0.0 + where[i] >= 0.9) {
         where[i] = 0;
         stop[i] = 1;
         break;
      }
   }
   int count = 0;
   for (int i = 0; i < 6; i++)
   {
      if (stop[i] == 1) {
         count++;
      }
   }
   if (count == 6)
   {
      exit(0);
   }
   
   glutPostRedisplay();
   if (IsStart == 1)
      glutTimerFunc(1000, MyTimer, 3);
}

void MyMainMenu(int entryID) // entryID = AddMenuEntry 시 적은 번호
{
   if (entryID == 1)
   {
      IsStart = 1;
      glutTimerFunc(1000, MyTimer, 3);
   }
   else if (entryID == 2)
      IsStart = 0;
   glutPostRedisplay();
}
void MyMouseClick(GLint Button, GLint State, GLint Mouse_X, GLint Mouse_Y) {
   if (Button == GLUT_LEFT_BUTTON && State == GLUT_DOWN) {

      tmp_x = ((GLfloat)Mouse_X / 640.0);
      tmp_y = ((GLfloat)(480 - Mouse_Y) / 480.0);                     //일반좌표계와 gl좌표계의 Y좌표가 반대이기때문에 꼭 !!!!!
      printf("(%f,%f)\n", tmp_x, tmp_y);
      int num;
      for (int i = 0; i < 6; i++)
      {
         if ((tmp_y >= 0.9 - (0.18 * i) && tmp_y <= 1.0 - (0.18 * i))) {
            num = i;
            where[num] = tmp_x;
            break;
         }
      }
      
   }

}
void MyKeyboard(unsigned char KeyPressed, int X, int Y) {
   switch (KeyPressed) {
   case 'i':
      move = move + 0.1; break;
   case 'm':
      move = move - 0.1; break;
   }
   glutPostRedisplay();
}
int main(int argc, char** argv)
{
   //GLUT 윈도우 함수, 윈도우 초기화
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB);
   glutInitWindowSize(640, 480);       // 윈도우 크기          
   glutInitWindowPosition(200, 200);     // 윈도우 생성 위치
   glutCreateWindow("OpenGL Sample Drawing");

   // openGL
   glClearColor(0.0, 0.0, 0.0, 0.0);   //GL 상태변수 설정 , 기본 r,g,b 0으로 설정 , 특수용 -> 0

   // 다음 세줄은 그냥 외울것
   glMatrixMode(GL_PROJECTION);  // 원근법 없이
   glLoadIdentity();
   glOrtho(0.0, 1.0, .0, 1.0, -1.0, 1.0);   //무대설정  x좌표 -1 ~ 1,  y좌표 -1 ~1 , z좌표 -1 ~ 1

   GLint MyMainMenuID = glutCreateMenu(MyMainMenu);      // 함수 등록 
   glutAddMenuEntry("START", 1);            // 인자 : 1) 문자열 , 2) 번호(int)
   glutAddMenuEntry("END", 2);
   glutAttachMenu(GLUT_RIGHT_BUTTON);

   glutDisplayFunc(MyDisplay);          //GLUT 콜백함수 등록
   glutMouseFunc(MyMouseClick);
   glutKeyboardFunc(MyKeyboard);

   glutMainLoop();               //이벤트 루프 진입
   return 0;
}
