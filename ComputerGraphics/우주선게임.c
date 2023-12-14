#include <GL/glut.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
GLfloat sunRadius = 0.6f; // 태양의 반지름
GLfloat ex = 10.0, ey = 0.0, ez = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
GLfloat missileSpeed = 0.5; // 미사일의 속도
int score = 0;
// 미사일의 정보 구조체
struct Missile {
    GLfloat position[3]; // 현재 위치
    GLfloat direction[3]; // 이동 방향
    bool active;          // 미사일 활성 여부
};

// 미사일 객체
Missile missile = { {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, false };

// 각 행성의 정보 구조체
struct Planet {
    GLfloat radius;   // 반지름
    GLfloat distance; // 궤도 반지름
    GLfloat speed;    // 공전 속도
    GLfloat angle;    // 초기 각도
    GLfloat color[3]; // 색상
};

// 행성 정보 배열
Planet planets[] = {
    {0.3, 1.5, 2.0 * std::acos(-1.0) / 150.0, 0.0, {1.0, 1.0, 1.0}},
    {0.4, 3.5, 2.0 * std::acos(-1.0) / 350.0, 0.0, {0.0, 1.0, 0.0}},
    {0.5, 4.5, 2.0 * std::acos(-1.0) / 450.0, 0.0, {1.0, 0.0, 0.0}},
    {0.7, 5.5, 2.0 * std::acos(-1.0) / 550.0, 0.0, {1.0, 1.0, 0.0}},
    {0.8, 7.5, 2.0 * std::acos(-1.0) / 750.0, 0.0, {0.0, 1.0, 1.0}}
};

// 소행성 정보 배열
const int numAsteroids = 12;
Planet asteroids[numAsteroids];

void initAsteroids() {
    for (int i = 0; i < numAsteroids; ++i) {
        asteroids[i].radius = 0.2;
        asteroids[i].distance = 3.0 + i * 1.0;  // 수정된 거리
        asteroids[i].speed = 2.0 * std::acos(-1.0) / (100.0 * asteroids[i].distance);
        asteroids[i].angle = static_cast<GLfloat>(rand()) / RAND_MAX * 2.0 * std::acos(-1.0);
    }
}

void init() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);

    GLfloat lightPosition[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat lightAmbient[] = { 0.2, 0.2, 0.2, 1.0 };
    GLfloat lightDiffuse[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat lightSpecular[] = { 1.0, 1.0, 1.0, 1.0 };

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    glClearColor(0.0, 0.0, 0.0, 0.0); // 배경색을 흰색으로 설정
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, 1200, 0, 900); // 윈도우 좌표 설정

    glEnable(GL_COLOR_MATERIAL);
}

void drawSun() {
    glColor3f(1.0, 1.0, 0.0);
    glutSolidSphere(sunRadius, 50, 50);
}

void drawPlanet(Planet planet) {
    // 태양에서 행성으로 그림자를 만들기 위한 조명 설정
    GLfloat lightPosition[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat planetPosition[] = { planet.distance * std::cos(planet.angle), 0.0, planet.distance * std::sin(planet.angle), 1.0 };

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, planet.color);

    // 행성의 좌표를 이동하여 태양과 행성 간의 상대 위치를 고려
    glPushMatrix();
    glTranslatef(planetPosition[0], planetPosition[1], planetPosition[2]);

    // 그림자 생성
    glEnable(GL_LIGHT1);
    GLfloat lightDirection[] = { -planetPosition[0], -planetPosition[1], -planetPosition[2] };
    glLightfv(GL_LIGHT1, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, lightDirection);
    glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 90.0);

    // 행성을 사각형으로 그리기 (사이즈 조절 가능)
    GLfloat planetSize = planet.radius * 2.0;
    glScalef(planetSize, planetSize, planetSize);
    glutSolidCube(1.0);  // 또는 glutSolidDodecahedron() 등을 사용할 수 있습니다.

    glPopMatrix();

    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHT1);
    glDisable(GL_LIGHTING);
}

void drawMissile() {
    glColor3f(1.0, 0.0, 0.0);
    glPushMatrix();
    glTranslatef(missile.position[0], missile.position[1], missile.position[2]);
    glutSolidSphere(0.1, 10, 10);
    glPopMatrix();
}

void drawAsteroids() {
    glColor3f(0.5, 0.5, 0.5);
    for (const auto& asteroid : asteroids) {
        glPushMatrix();
        glTranslatef(asteroid.distance * std::cos(asteroid.angle),
            0.0,
            asteroid.distance * std::sin(asteroid.angle));
        glutSolidSphere(asteroid.radius, 20, 20);
        glPopMatrix();
    }
}
// 우주쓰레기의 정보 구조체
struct SpaceJunk {
    GLfloat position[3]; // 현재 위치
    GLfloat direction[3]; // 이동 방향
    GLfloat rotationSpeed; // 회전 속도
    bool active;          // 우주쓰레기 활성 여부
};

// 우주쓰레기 객체 배열
const int numSpaceJunk = 3;
SpaceJunk spaceJunk[numSpaceJunk];

void initSpaceJunk() {
    for (int i = 0; i < numSpaceJunk; ++i) {
        spaceJunk[i].active = false;

        // 우주쓰레기를 우주선과 태양의 반대편에서 생성
        spaceJunk[i].position[0] = -ex;
        spaceJunk[i].position[1] = -ey;
        spaceJunk[i].position[2] = -ez;

        // 방향은 플레이어(나)를 향하도록 설정
        GLfloat playerDirection[] = { ex - spaceJunk[i].position[0], ey - spaceJunk[i].position[1], ez - spaceJunk[i].position[2] };
        GLfloat length = sqrt(playerDirection[0] * playerDirection[0] + playerDirection[1] * playerDirection[1] + playerDirection[2] * playerDirection[2]);
        spaceJunk[i].direction[0] = playerDirection[0] / length;
        spaceJunk[i].direction[1] = playerDirection[1] / length;
        spaceJunk[i].direction[2] = playerDirection[2] / length;

        spaceJunk[i].active = true;
    }
}

void drawSpaceJunk() {
    glColor3f(0.5, 0.5, 0.5);
    for (const auto& junk : spaceJunk) {
        if (junk.active) {
            glPushMatrix();
            glTranslatef(junk.position[0], junk.position[1], junk.position[2]);
            glRotatef(junk.rotationSpeed, 1.0, 1.0, 1.0);  // 회전 속도에 따라 회전

            // 크기 조절
            GLfloat scaleFactor = 0.2;  // 원하는 크기로 조절
            glScalef(scaleFactor, scaleFactor, scaleFactor);

            glutSolidDodecahedron(); // 정 12면체로 변경
            glPopMatrix();
        }
    }
}

void updateSpaceJunk() {
    for (auto& junk : spaceJunk) {
        if (!junk.active) {
            // 새로운 우주쓰레기 생성
            junk.position[0] = static_cast<GLfloat>(rand()) / RAND_MAX * 20.0 - 10.0; // x 좌표 랜덤 생성
            junk.position[1] = static_cast<GLfloat>(rand()) / RAND_MAX * 20.0 - 10.0; // y 좌표 랜덤 생성
            junk.position[2] = static_cast<GLfloat>(rand()) / RAND_MAX * 20.0 - 10.0; // z 좌표 랜덤 생성

            // 방향은 플레이어(나)를 향하도록 설정
            GLfloat playerDirection[] = { ex - junk.position[0], ey - junk.position[1], ez - junk.position[2] };
            GLfloat length = sqrt(playerDirection[0] * playerDirection[0] + playerDirection[1] * playerDirection[1] + playerDirection[2] * playerDirection[2]);
            junk.direction[0] = playerDirection[0] / length;
            junk.direction[1] = playerDirection[1] / length;
            junk.direction[2] = playerDirection[2] / length;

            junk.active = true;
        }

        // 우주쓰레기 위치 업데이트
        junk.position[0] += junk.direction[0] * 0.1; // 우주쓰레기의 이동 속도를 조절할 수 있습니다.
        junk.position[1] += junk.direction[1] * 0.1;
        junk.position[2] += junk.direction[2] * 0.1;

        // 우주쓰레기가 화면을 벗어나면 비활성화
        if (junk.position[0] > 20.0 || junk.position[0] < -20.0 || junk.position[1] > 20.0 || junk.position[1] < -20.0 || junk.position[2] > 20.0 || junk.position[2] < -20.0) {
            junk.active = false;
        }

        // 미사일과 우주쓰레기 간의 충돌 여부 확인
        if (missile.active) {
            GLfloat distanceToJunk = sqrt(pow(missile.position[0] - junk.position[0], 2) +
                pow(missile.position[1] - junk.position[1], 2) +
                pow(missile.position[2] - junk.position[2], 2));

            if (distanceToJunk < (0.1 + 0.1)) { // 미사일과 우주쓰레기의 충돌 반지름을 조절할 수 있습니다.
                junk.active = false;
                missile.active = false;
                score += 1; // 스코어 증가
                printf("우주쓰레기 제거! 현재 점수: %d\n", score);
            }
        }

    }
}
void updatePlanets() {
    for (auto& planet : planets) {
        planet.angle += planet.speed; // 각 행성의 회전 속도에 따라 각도 갱신
    }
}

void updateAsteroids() {
    for (auto& asteroid : asteroids) {
        asteroid.angle += asteroid.speed; // 각 소행성의 회전 속도에 따라 각도 갱신
    }
}

void updateMissile() {
    if (missile.active) {
        // 미사일과 태양 간의 충돌 여부 확인
        GLfloat distanceToSun = sqrt(pow(missile.position[0], 2) +
            pow(missile.position[1], 2) +
            pow(missile.position[2], 2));

        if (distanceToSun < (sunRadius + 0.1)) {
            sunRadius = 0.0;
            missile.active = false;

            printf("태양과 충돌!\n");
            score = score - 10;
            printf("점수 : %d\n", score);
        }

        // 각 행성에 대한 충돌 여부 확인
        for (auto& planet : planets) {
            GLfloat distanceToPlanet = sqrt(pow(missile.position[0] - planet.distance * cos(planet.angle), 2) +
                pow(missile.position[1], 2) +
                pow(missile.position[2] - planet.distance * sin(planet.angle), 2));

            if (distanceToPlanet < (planet.radius + 0.1)) {
                planet.radius = 0.0;
                missile.active = false;

                printf("행성과 충돌!\n");
                score = score - 5;
                printf("점수 : %d\n", score);
            }
        }

        // 각 소행성에 대한 충돌 여부 확인
        for (auto& asteroid : asteroids) {
            GLfloat distanceToAsteroid = sqrt(pow(missile.position[0] - asteroid.distance * cos(asteroid.angle), 2) +
                pow(missile.position[1], 2) +
                pow(missile.position[2] - asteroid.distance * sin(asteroid.angle), 2));

            if (distanceToAsteroid < (asteroid.radius + 0.1)) {
                asteroid.radius = 0.0;
                missile.active = false;

                printf("소행성과 충돌!\n");
                score = score + 5;
                printf("점수 : %d\n", score);
            }
        }

        // 미사일 위치 업데이트
        missile.position[0] += missile.direction[0] * missileSpeed;
        missile.position[1] += missile.direction[1] * missileSpeed;
        missile.position[2] += missile.direction[2] * missileSpeed;

        // 미사일이 화면을 벗어나면 비활성화
        if (missile.position[0] < -20.0) {
            missile.active = false;
        }
    }
}

void drawMiniMap() {
    // 더 작은 미니맵 크기로 조절
    int miniMapWidth = glutGet(GLUT_WINDOW_WIDTH) / 6;
    int miniMapHeight = glutGet(GLUT_WINDOW_HEIGHT) / 4;
    int miniMapX = (glutGet(GLUT_WINDOW_WIDTH) - miniMapWidth) / 2;
    int miniMapY = (glutGet(GLUT_WINDOW_HEIGHT) - miniMapHeight) / 4;

    glViewport(miniMapX, miniMapY, miniMapWidth, miniMapHeight);
    glScissor(miniMapX, miniMapY, miniMapWidth, miniMapHeight);
    glEnable(GL_SCISSOR_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (GLfloat)miniMapWidth / (GLfloat)miniMapHeight, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 35.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);

    drawSun();
    for (const auto& planet : planets) {
        drawPlanet(planet);
    }
    drawAsteroids();
    drawMissile();
    drawSpaceJunk();
    glColor3f(1.0, 0.0, 0.0);
    glPushMatrix();
    glTranslatef(ex, ey, ez);
    glutSolidTeapot(0.3);
    glPopMatrix();

    glDisable(GL_SCISSOR_TEST);
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 왼쪽 뷰
    glViewport(0, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glScissor(0, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glEnable(GL_SCISSOR_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, (GLfloat)glutGet(GLUT_WINDOW_WIDTH) / (GLfloat)glutGet(GLUT_WINDOW_HEIGHT), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex, ey, ez, ex - 5.0, ey, ez + 10.0, 0.0, 1.0, 0.0);
    drawSun();
    for (const auto& planet : planets) {
        drawPlanet(planet);
    }
    drawSpaceJunk();
    drawAsteroids();
    drawMissile();

    // 중앙 뷰
    glViewport(glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glScissor(glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, (GLfloat)glutGet(GLUT_WINDOW_WIDTH) / (GLfloat)glutGet(GLUT_WINDOW_HEIGHT), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex, ey, ez, mx, my, mz, 0.0, 1.0, 0.0);
    drawSun();
    for (const auto& planet : planets) {
        drawPlanet(planet);
    }
    drawSpaceJunk();
    drawAsteroids();
    drawMissile();

    // 오른쪽 뷰
    glViewport(2 * glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glScissor(2 * glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2, glutGet(GLUT_WINDOW_WIDTH) / 3, glutGet(GLUT_WINDOW_HEIGHT) / 2);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, (GLfloat)glutGet(GLUT_WINDOW_WIDTH) / (GLfloat)glutGet(GLUT_WINDOW_HEIGHT), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex, ey, ez, ex - 5.0, ey, ez - 10.0, 0.0, 1.0, 0.0);

    drawAsteroids();
    drawMissile();
    drawSpaceJunk();
    drawMiniMap();  // 미니 맵 추가

    glDisable(GL_SCISSOR_TEST);

    glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
}

void checkGameEnd() {
    bool allAsteroidsDestroyed = true;
    for (const auto& asteroid : asteroids) {
        if (asteroid.radius > 0.0) {
            allAsteroidsDestroyed = false;
            break;
        }
    }

    if (allAsteroidsDestroyed) {
        printf("게임 종료! 최종 점수: %d\n", score);
        exit(0);
    }
}

void checkCollisionWithSun() {
    // 태양과의 충돌 여부 확인
    GLfloat distanceToSun = sqrt(pow(ex, 2) + pow(ey, 2) + pow(ez, 2));
    if (distanceToSun < (sunRadius + 0.3)) {
        printf("태양과 충돌! 게임 종료! 최종 점수: %d\n", score);
        exit(0);
    }
}

void checkCollisionWithPlanets() {
    // 행성과의 충돌 여부 확인
    for (const auto& planet : planets) {
        GLfloat distanceToPlanet = sqrt(pow(ex - planet.distance * cos(planet.angle), 2) +
            pow(ey, 2) +
            pow(ez - planet.distance * sin(planet.angle), 2));

        if (distanceToPlanet < (planet.radius + 0.3)) {
            printf("행성과 충돌! 게임 종료! 최종 점수: %d\n", score);
            exit(0);
        }
    }
}

bool checkCollisionWithObject(GLfloat x1, GLfloat y1, GLfloat z1, GLfloat x2, GLfloat y2, GLfloat z2, GLfloat radius) {
    GLfloat distance = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
    return distance < radius;
}
void checkCollisionWithSpaceJunk() {
    for (const auto& junk : spaceJunk) {
        if (junk.active && checkCollisionWithObject(ex, ey, ez, junk.position[0], junk.position[1], junk.position[2], 0.3)) {
            printf("우주쓰레기와 충돌! 게임 종료! 최종 점수: %d\n", score);
            exit(0);
        }
    }
}

void checkCollisionWithspace() {
    for (const auto& planet : planets) {
        GLfloat distanceToPlanet = sqrt(pow(ex - planet.distance * cos(planet.angle), 2) +
            pow(ey, 2) +
            pow(ez - planet.distance * sin(planet.angle), 2));

        if (distanceToPlanet < (planet.radius + 0.3)) {
            printf("행성과 충돌! 게임 종료! 최종 점수: %d\n", score);
            exit(0);
        }
    }
}

void update(int value) {
    updatePlanets();
    updateAsteroids();
    updateMissile();
    checkGameEnd();
    updateSpaceJunk();
    checkCollisionWithPlanets();
    checkCollisionWithSpaceJunk();
    checkCollisionWithSun();
    checkCollisionWithPlanets();
    glutPostRedisplay();
    glutTimerFunc(10, update, 0);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'a':
        ez += 0.1;
        mz += 0.1;
        break;
    case 'd':
        ez -= 0.1;
        mz -= 0.1;
        break;
    case 'w':
        ex -= 0.1;
        mx -= 0.1;
        break;
    case 's':
        ex += 0.1;
        mx += 0.1;
        break;
    case 'l':
        if (!missile.active) {
            missile.position[0] = ex;
            missile.position[1] = ey;
            missile.position[2] = ez;
            missile.active = true;
        }
        break;
    }
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(1200, 900);

    glutCreateWindow("Solar System");

    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(10, update, 0);
    glutKeyboardFunc(keyboard);
    srand(time(nullptr));
    initSpaceJunk();
    initAsteroids();
    glEnable(GL_SCISSOR_TEST);

    // glutReshapeFunc에 reshape 함수 등록
    glutReshapeFunc(reshape);

    glutMainLoop();
    return 0;
}
