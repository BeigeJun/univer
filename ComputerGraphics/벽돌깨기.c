#include <gl/glut.h>               
#include <gl/gl.h>                  
#include <gl/glu.h>   
#include <stdio.h>
GLfloat Delta_X = 0.0;
GLfloat Delta_Y = 0.0;
GLint flag_x = 1; // flag_xy는 공이 움직는 방향 즉 왼쪽 오른쪽 아래 위
GLint flag_y = 1;
GLfloat bal_x = 0.0; // 바 움직이기
GLint block_life[5][5] = { {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1} };

void MyDisplay() {
   glClear(GL_COLOR_BUFFER_BIT);

   //공
   glBegin(GL_POLYGON);
   glColor3f(0.0, 0.5, 0.8);
   glVertex3f(-0.05 + Delta_X, -0.05 + Delta_Y, 0.0);
   glVertex3f(0.05 + Delta_X, -0.05 + Delta_Y, 0.0);
   glVertex3f(0.05 + Delta_X, 0.05 + Delta_Y, 0.0);
   glVertex3f(-0.05 + Delta_X, 0.05 + Delta_Y, 0.0);
   glEnd();

   //밑에 바
   glBegin(GL_POLYGON);
   glColor3f(0.0, 0.5, 0.8);
   glVertex3f(-0.3 + bal_x, -0.8, 0.0);
   glVertex3f(0.3 + bal_x, -0.8, 0.0);
   glVertex3f(0.3 + bal_x, -0.7, 0.0);
   glVertex3f(-0.3 + bal_x, -0.7, 0.0);
   glEnd();

   //블록
   for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
         if (block_life[i][j] == 1)
         {
            glBegin(GL_POLYGON);
            glColor3f(0.0, 0.5, 0.8);
         }
         else
         {
            glBegin(GL_POLYGON);
            glColor3f(1.0, 0.0, 0.0);
         }
         glVertex3f(-1.0 + (j * 0.4) + 0.01, 1.0 - (i * 0.1) - 0.01, 0.0);
         glVertex3f(-0.6 + (j * 0.4) - 0.01, 1.0 - (i * 0.1) - 0.01, 0.0);
         glVertex3f(-0.6 + (j * 0.4) - 0.01, 0.90 - (i * 0.1) + 0.01, 0.0);
         glVertex3f(-1.0 + (j * 0.4) + 0.01, 0.90 - (i * 0.1) + 0.01, 0.0);
         glEnd();
      }
   }
   glutSwapBuffers();
}

void MyIdle() {
   Delta_X = Delta_X + flag_x * (0.005);
   Delta_Y = Delta_Y + flag_y * (0.005);
   if (Delta_X > 1)
      flag_x = -1;
   else if (Delta_X < -1)
      flag_x = 1;

   if (Delta_Y > 1)
      flag_y = -1;
   else if (Delta_Y < -1)
      flag_y = 1;
   //-0.05 + Delta_X, -0.05+Delta_Y//공 왼쪽아래
   //-0.3+bal_x , -0.7//바 왼쪽위 
   if (((-0.05 + Delta_X) >= (-0.3 + bal_x)) && ((-0.05 + Delta_X) <= (0.3 + bal_x)) &&
      (-0.05 + Delta_Y) <= -0.75 && (-0.05 + Delta_Y) >= -0.85) // 막대기 좌표
   {
      flag_y = 1; //막대바랑 공이 만나면 가는 방향 바꾸기
   }
   if ((-0.05 + Delta_Y) <= -0.9) // 끝좌표
   {
      printf("끝");
      exit(0);
   }
   for (int i = 0; i < 5; i++)
   {
      for (int j = 0; j < 5; j++)
      {
         //-1.0 + (j * 0.4) + 0.01,       0.90 - (i * 0.1) + 0.01 블록의 왼쪽아래
         //-0.6 + (j * 0.4) - 0.01,       0.90 - (i * 0.1) + 0.01 블록의 오른쪽 아래
         //-0.05 + Delta_X, 0.05+Delta_Y 공의 왼쪽 위
         if (block_life[i][j] == 1)
         {
            if ((-0.05 + Delta_X) >= (-1.0 + (j * 0.4) + 0.01) && (-0.05 + Delta_X) <= (-0.6 + (j * 0.4) - 0.01) &&
               (0.05 + Delta_Y) <= (0.90 - (i * 0.1) + 0.01 + 0.05) && (0.05 + Delta_Y) >= (0.90 - (i * 0.1) + 0.01 - 0.05))
            {
               flag_y = -1;
               block_life[i][j] = 0;
            }
         }
         else if (block_life[i][j] == 0)
         {
            if ((-0.05 + Delta_X) >= (-1.0 + (j * 0.4) + 0.01) && (-0.05 + Delta_X) <= (-0.6 + (j * 0.4) - 0.01) &&
               (0.05 + Delta_Y) <= (0.90 - (i * 0.1) + 0.01 + 0.05) && (0.05 + Delta_Y) >= (0.90 - (i * 0.1) + 0.01 - 0.05))
            {
               flag_y = -1;
               block_life[i][j] = 1;
            }
         }
      }
   }

   glutPostRedisplay();
}


void MyKeyboard(unsigned char KeyPressed, int X, int Y) {
   switch (KeyPressed) {
   case 'j':
      bal_x -= 0.1;
      break;
   case 'k':
      bal_x += 0.1;
      break;
   }
   glutPostRedisplay();
}

int main(int argc, char** argv) {
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   glutInitWindowSize(300, 300);
   glutInitWindowPosition(0, 0);
   glutCreateWindow("OpenGL Drawing Example");

   glClearColor(1.0, 1.0, 1.0, 1.0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

   glutDisplayFunc(MyDisplay);
   glutIdleFunc(MyIdle);
   glutKeyboardFunc(MyKeyboard);

   glutMainLoop();
   return 0;
}
