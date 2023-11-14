#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdio.h>

GLfloat eX = 1.0;
GLfloat eY = 0.0;
GLfloat eZ = 0.0;
GLfloat aX = 0.0;
GLfloat aY = 0.0;
GLfloat aZ = 0.0;
GLfloat uX = 0.0;
GLfloat uY = 1.0;
GLfloat uZ = 0.0;
GLfloat movex = 0.1;
/*GLfloat RRX = 1.0;
GLfloat RRZ = 0.0;
GLfloat RLX = 1.0;
GLfloat RLZ = 0.0;*/
/*eX, eY, eZ: 눈(eye)의 좌표를 나타냅니다. 이는 카메라의 위치를 결정합니다.

aX, aY, aZ: 시야의 중점(at) 좌표를 나타냅니다. 이는 카메라가 바라보는 지점을 결정합니다.

uX, uY, uZ: 카메라의 상단(up) 방향을 나타냅니다. 이는 카메라의 방향에 대한 기준이 되는 벡터입니다.

RRX, RRZ: 오른쪽(right) 벡터를 나타냅니다. 이 벡터는 카메라가 바라보는 방향과 상단 방향에 수직이며, 오른쪽으로 향하는 벡터입니다.

RLX, RLZ: 왼쪽(left) 벡터를 나타냅니다. 이 벡터는 오른쪽 벡터와 수직이며, 왼쪽으로 향하는 벡터입니다.*/
void InitLight() {

    GLfloat mat_diffuse[] = { 0.5, 0.4, 0.3, 1.0 };
    GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat mat_ambient[] = { 0.5, 0.4, 0.3, 1.0 };
    GLfloat mat_shininess[] = { 15.0 };
    GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
    GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
    GLfloat light_position[] = { -3, 6, 3.0, 0.0 };

    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
}

void MyDisplay() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eX, 0.5, eZ, 0.0, 0.0, 0.0, uX, uY, uZ);

    glutSolidTeapot(0.1);
    //    glTranslatef(0.2, 0.0, 0.0); //좌표입력값 만큼 움직이기
    glFlush();
}

void MyReshape(int w, int h) {

    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
}

void KB(unsigned char KeyPressed, int X, int Y) {

    switch (KeyPressed) {
    case '1':
        eX = 1.0;
        eY = 0.0;
        eZ = 0.0;
        break;

    case 'k':

        if (eX > 0.0 && eZ >= 0.0) {
            eX -= 0.1;
            eZ = 1 - eX;
        }
        else if (eX <= 0.0 && eZ > 0.0) {
            eX -= 0.1;
            eZ = 1 + eX;
        }
        else if (eX < 0.0 && eZ <= 0.0) {
            eX += 0.1;
            eZ = -(1 + eX);
        }
        else if (eX >= 0.0 && eZ < 0.0) {
            eX += 0.1;
            eZ = -(1 - eX);
        }
        break;

    case 'j':

        if (eX >= 0.0 && eZ <= 0.0) {
            eX -= 0.1;
            eZ = -(1 - eX);
        }//eZ = 1 - eX;

        else if (eX <= 0.0 && eZ < 0.0) {
            eX -= 0.1;
            eZ = -(1 + eX); //eZ = -(1 + eX);
        }
        else if (eX < 0.0 && eZ >= 0.0) {
            eX += 0.1;
            eZ = 1 + eX;
        }
        else if (eX >= 0.0 && eZ > 0.0) {
            eX += 0.1;
            eZ = 1 - eX;
        }
        break;


    }
    printf("eye(%.2f, %.2f, %.2f) at(%.2f, %.2f, %.2f) up(%.2f, %.2f, %.2f)\n", eX, eY, eZ, aX, aY, aZ, uX, uY, uZ);

    glutPostRedisplay();
}

int main(int argc, char** argv) {

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(400, 400);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("OpenGL Sample Drawing");

    glClearColor(0.4, 0.4, 0.4, 0.0);
    InitLight(); // 조명함수

    glutDisplayFunc(MyDisplay);
    glutReshapeFunc(MyReshape);
    glutKeyboardFunc(KB);
    glutMainLoop();

    return 0;
}
