#include <gl/glut.h>					
#include <gl/gl.h>						
#include <gl/glu.h>	

void MyDisplay() {
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, 300, 300); // 해도 괜찮고 안해도 괜찮음
    glColor3f(1.0, 0.0, 0.0);

    glMatrixMode(GL_MODELVIEW); //
    glLoadIdentity();

    glRotatef(50.0, 1.0, 0.0, 1.0); //회전 강도 / x/ y/z
    glTranslatef(0.0, 0.0, 0.0); // 이동이동
    //glRotatef(45.0, 0.0, 0.0, 1.0);

    glutSolidCube(0.1);
    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("OpenGL Sample Drawing");
    glClearColor(1.0, 1.0, 1.0, 1.0);

    glMatrixMode(GL_PROJECTION); //원근법 사용을 안한다.
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glutDisplayFunc(MyDisplay);

    glutMainLoop();
    return 0;
}

