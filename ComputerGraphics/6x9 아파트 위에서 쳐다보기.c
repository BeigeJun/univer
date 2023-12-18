#include <cmath>
#include <cstdio>
#include <gl/glut.h>					
#include <gl/gl.h>						
#include <gl/glu.h>

GLint x = 3; //행
GLint y = 4; //열

GLfloat ax = 0.5;
GLfloat az = 0.5;

GLint angle = 0;
GLfloat radian = 0;

void MyLightInit()
{
	GLfloat global_ambient[] = { 1, 0.2, 0.3, 1.0 };	//전역 주변반사 // 무조건 넣기

	GLfloat light1_ambient[] = { 0.0, 0.0, 0.0, 1.0 };	//1번 광원 특성	// 주변광
	GLfloat light1_diffuse[] = { 1, 0.1, 0.1, 1.0 };					// 확선광 - 본 빛
	GLfloat light1_specular[] = { 0.0, 0.0, 0.0, 1.0 };					// 경명광

	GLfloat material_ambient[] = { 0.1, 0.1, 0.1, 1.0 };	//물체 특성  // 물체 자체의 색
	GLfloat material_diffuse[] = { 0.0, 0.8, 0.0, 1.0 };
	GLfloat material_specular[] = { 0.0, 0.0, 1.0, 1.0 };
	GLfloat material_shininess[] = { 25.0 };								// 광택개수

	glShadeModel(GL_SMOOTH);	//구로 셰이딩
	glEnable(GL_DEPTH_TEST);	//깊이 버퍼 활성화

	glEnable(GL_LIGHTING);		//조명 활성화    // 조명을 설치했다

	glEnable(GL_LIGHT1);		//1번 광원 활성화
	glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient);	//1번 광원 특성할당		fv : float bectorr
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light1_specular);

	glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse);//물체 특성할당
	glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
	glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient);
	glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);//전역주변반사					그대로 가져다 쓸 것
	//glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE); //근접시점
}
void MyTimer(int Value) // Value : 알람 번호    // WIN32 API 와는 다르게 타이머가 한번 울리고 끝나므로 계속 선언을 해주어야함
{
	angle += 10;
	radian = angle * (3.141592 / 180);
	ax = cos(radian);
	az = sin(radian);

	printf("%d\n", angle);
	glutPostRedisplay();
	glutTimerFunc(100, MyTimer, 3);
}


void apt_1_complex()
{
	glColor3f(0.0, 0.8, 0.0);
	glTranslatef(-0.15, 0.0, 0.1);

	for (int m = 0; m < 3; m++)
	{
		glPushMatrix();
		for (int n = 0; n < 2; n++)
		{
			glTranslatef(0.1, 0.0, 0.0);
			glPushMatrix();
			glScalef(3.0, 7.0, 4.0);
			glutSolidCube(0.015);
			glPopMatrix();
		}
		glPopMatrix();
		glTranslatef(0.0, 0.0, -0.1);
	}
}

void MyDisplay()
{
	GLfloat LightPosition1[] = { 0.1, 0.5, 0.1, 1.0 };	//1번 광원위치
	GLfloat LightDirection1[] = { 0, 0, -1, 1.0 };	//1번 광원 방향 방향이기때문에 어딜 향해 비춘다가 아님 어느 방향으로 비춘다임
	GLfloat SpotAngle1[] = { 45.0 };

	glClear(GL_COLOR_BUFFER_BIT);

	glFrontFace(GL_CCW);
	glEnable(GL_CULL_FACE);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluPerspective(60, 1, 0.01, 50.0);
	gluLookAt(0.0, 1, 0.001, 0, 0., 0, 0, 1, 0);
	//glRotatef(30.0, 1.0, .0, .0);

	glLightfv(GL_LIGHT1, GL_POSITION, LightPosition1);	//스포트라이트

	glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, LightDirection1);	//방향
	glLightfv(GL_LIGHT1, GL_SPOT_CUTOFF, SpotAngle1);	//각도
	glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 1.0);			//코사인 승수


	//glColor3f(0.0, 1.0, 0.0);
	//glutSolidCube(0.1);

	glPushMatrix();
	glTranslatef(-0.2, 0.0, 0.5);
	for (int i = 0; i < 3; i++)
	{
		glPushMatrix();
		for (int j = 0; j < 4; j++)
		{
			glTranslatef(0.0, 0.0, -0.2);
			glPushMatrix();
			apt_1_complex();
			glPopMatrix();
		}
		glPopMatrix();
		glTranslatef(0.2, 0.0, 0.0);
	}


	glPopMatrix();

	glFlush();
	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("OpenGL Drawing Example");
	glClearColor(1.0, 1.0, 1.0, 1.0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	//MyLightInit( );
	glutDisplayFunc(MyDisplay);
	glutTimerFunc(100, MyTimer, 3);
	glutMainLoop();
	return 0;
}
