#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<Windows.h>

#include <gl/glut.h>                  
#include <gl/gl.h>                     
#include <gl/glu.h>   



GLint TopLeftX, TopLeftY, BottomRightX, BottomRightY;

int flag = 0, color = 0, random = 0, start_s = 0, a_count = 0, red = 0, green = 0, blue = 1;

void MyDisplay() {
    glViewport(0, 0, 400, 400);
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POLYGON);
    glColor3f(red, 0.0, blue);
    glVertex3f(50, 50, 0.0);
    glVertex3f(80, 50, 0.0);
    glVertex3f(80, 80, 0.0);
    glVertex3f(50, 80, 0.0);

    glEnd();
  

    glFlush();
}





int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(300, 300);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("OpenGL Drawing Example");

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();


    glOrtho(0, 300, 0, 300, -1.0, 1.0);

    glutDisplayFunc(MyDisplay);



    glutMainLoop();
    return 0;
}
