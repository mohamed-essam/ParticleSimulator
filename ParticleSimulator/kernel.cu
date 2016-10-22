#include <stdio.h>
#include <glew/glew.h>
#include <GL/freeglut.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <GL/glut.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;
const int COUNT = 30000000;

//float4* positions;
float2* velocity;
float2* xy;

__global__ void simulateFrame(float* positions, float2* velocity, float2* xy, float delta, float particleRadius)
{
	delta /= 1000.0;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= COUNT) return;

	xy[idx].x += velocity[idx].x * delta;
	xy[idx].y += velocity[idx].y * delta;
	if (xy[idx].x < 0 + particleRadius) {
		velocity[idx].x *= -1;
		xy[idx].x *= -1;
	}
	if (xy[idx].x + particleRadius >= WIDTH) {
		velocity[idx].x *= -1;
		xy[idx].x -= 2 * ((xy[idx].x + particleRadius) - WIDTH);
	}
	if (xy[idx].y < 0 + particleRadius) {
		velocity[idx].y *= -1;
		xy[idx].y *= -1;
	}
	if (xy[idx].y + particleRadius >= HEIGHT) {
		velocity[idx].y *= -1;
		xy[idx].y -= 2 * ((xy[idx].y + particleRadius) - HEIGHT);
	}
	positions[idx * 3] = xy[idx].x;
	positions[idx * 3 + 1] = xy[idx].y;

}

__global__ void setVal(float2* arr, float2 val) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= COUNT) return;
	arr[idx] = val;
}

void initFrame();

void displayFrame();

GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

int main(int argv, char ** argc)
{
	float2* h_velocity = new float2[COUNT];

	for (int idx = 0; idx < COUNT; idx++)
	{
		h_velocity[idx].x = rand() % 200 + 10;
		if (rand() % 2)
			h_velocity[idx].x *= -1;
		h_velocity[idx].y = rand() % 200 + 10;
		if (rand() % 2)
			h_velocity[idx].y *= -1;
	}

	cudaMalloc(&velocity, COUNT * sizeof(float2));
	cudaMemcpy(velocity, h_velocity, COUNT * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc(&xy, COUNT * sizeof(float2));
	setVal<<<(COUNT+1023)/1024, 1024>>>(xy, make_float2(100, 100));
	//cudaGLSetGLDevice(0);

	glutInit(&argv, argc);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);

	glutInitWindowPosition(0, 0);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Particles");

	glewInit();

	initFrame();
	glutDisplayFunc(displayFrame);



	glutMainLoop();
	return 0;
}

GLfloat* positions;

int lastTime = 0;
void initFrame(){
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glPointSize(4.0);
	glLineWidth(2.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, WIDTH, 0.0, HEIGHT);
	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	unsigned int size = WIDTH * HEIGHT * 4 * sizeof(GLfloat);
	glBufferData(GL_ARRAY_BUFFER, size, positions, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	lastTime = glutGet(GLUT_ELAPSED_TIME);
}

//void displayFrame() {
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glColor3f(1.0, 0.0, 1.0);
//
//	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
//	size_t num_bytes;
//	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);
//	// Execute kernel
//	simulateFrame <<<(COUNT + 1023) / 1024, 1024 >>>(positions, velocity, xy, glutGet(GLUT_ELAPSED_TIME), 1.0);
//	// Unmap buffer object
//	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
//	// Render from buffer object
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
//	glVertexPointer(4, GL_FLOAT, 0, 0);
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glDrawArrays(GL_POINTS, 0, WIDTH * HEIGHT);
//	glDisableClientState(GL_VERTEX_ARRAY);
//	// Swap buffers
//	glutSwapBuffers();
//	glutPostRedisplay();
//	/*
//	glBegin(GL_POINTS);
//	glVertex2f(10.0, 10.0);
//	glVertex2f(10.0, 30.0);
//	glEnd();
//	glFlush();
//	*/
//}
void displayFrame() {
	int ms = glutGet(GLUT_ELAPSED_TIME) - lastTime;
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3f(1.0, 0.0, 1.0);

	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);
	// Execute kernel
	lastTime = glutGet(GLUT_ELAPSED_TIME);
	simulateFrame <<<(COUNT + 1023) / 1024, 1024 >>>(positions, velocity, xy, ms, 4.0);
	// Unmap buffer object
	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
	// Render from buffer object
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 1.0, 1.0);
	glDrawArrays(GL_POINTS, 0, COUNT);
	glDisableClientState(GL_VERTEX_ARRAY);
	// Swap buffers
	glutSwapBuffers();
	glutPostRedisplay();
	/*
	glBegin(GL_POINTS);
	glVertex2f(10.0, 10.0);
	glVertex2f(10.0, 30.0);
	glEnd();
	glFlush();
	*/
}