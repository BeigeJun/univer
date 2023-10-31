#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<Windows.h>

#include <gl/glut.h>                  
#include <gl/gl.h>                     
#include <gl/glu.h>   



GLint TopLeftX, TopLeftY, BottomRightX, BottomRightY;

int flag = 0, color = 0, random = 0, start_s = 0, a_count = 0, red = 0, green = 1, blue = 0;

void MyDisplay() {
    glViewport(0, 0, 400, 400);
    glClear(GL_COLOR_BUFFER_BIT);
    int cnt = 0;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (cnt == random)
                glColor3f(red, green, blue);
            else
                glColor3f(1.0, 1.0, 1.0);

            glBegin(GL_POLYGON);

            glVertex3f(0 + (40 * j), 400 - (40 * i), 0.0);
            glVertex3f(0 + (40 * j), 200 - (40 * i), 0.0);
            glVertex3f(100 + (40 * j), 200 - (40 * i), 0.0);
            glVertex3f(100 + (40 * j), 400 - (40 * i), 0.0);

            glEnd();
            cnt+=1;
        }
    }

    glFlush();
}


void MyTimer(int Value) {

        random = rand()%100;
        glutPostRedisplay();
        system("cls");
        printf("%d", a_count);
        glutTimerFunc(1000, MyTimer, 1);
        green = 1.0;
        red = 0.0;

}
void MyMouseClick(GLint Button, GLint State, GLint X, GLint Y) {
    if (Button == GLUT_LEFT_BUTTON && State == GLUT_DOWN) {
        int cnt = 0;
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                if (cnt == random)
                {
                    if (X >= 0 + (40 * j) && X <= 40 + (40 * j) && Y <= 40 + (40 * i) && Y >= 0 + (40 * i))
                    {
                        red = 1.0;
                        green = 0.0;
                    }
                    else {
                        green = 0.0;
                        red = 0.0;
                    }


                }
                cnt++;
            }

        }
    }
    glutPostRedisplay();
}
/*
void MyMainMenu(int entryID) {
    if (entryID == 0)
    {
        start_s = 1;
        glutTimerFunc(500, MyTimer, 1);
    }
    else if (entryID == 1)
        start_s = 0;
    else if (entryID == 2)
        exit(1);
    glutPostRedisplay();
}

void MyKeyboard(unsigned char KeyPressed, int X, int Y) {
    switch (KeyPressed) {
    case 'r':
        red = 1;
        green = 0;
        blue = 0;
        break;
    case 'g':
        red = 0;
        green = 1;
        blue = 0;
        break;
    case 'b':
        red = 0;
        green = 0;
        blue = 1;
        break;
    }
    glutPostRedisplay();
}*/
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(400, 400);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("OpenGL Drawing Example");

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    //glOrtho(0.0, 300.0, 0.0, 300.0, -1.0, 1.0);
    //glOrtho(-1.0, 1.0,   -1.0, 1.0,   -1.0, 1.0); // Ori
    glOrtho(0, 400, 0, 400, -1.0, 1.0);

    glutDisplayFunc(MyDisplay);
    glutMouseFunc(MyMouseClick);
    /*GLint MyMainMenuID = glutCreateMenu(MyMainMenu);
    glutAddMenuEntry("start", 0);
    glutAddMenuEntry("stop", 1);
    glutAddMenuEntry("Exit", 2);*/
    glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutTimerFunc(1000, MyTimer, 1);
    //glutKeyboardFunc(MyKeyboard);

    glutMainLoop();
    return 0;
}
