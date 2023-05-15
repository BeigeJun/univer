#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>
#define BUFSIZE 256
#include "gpiolib.c"

#define BUZZER_PIN 02 /* GPIO 21 */

#define NOTE_C4  100 /* frequency of C4 note in Hz */
#define NOTE_D4  200 
#define NOTE_E4  300
#define NOTE_F4  400
#define NOTE_G4  500
#define NOTE_A4  600
#define NOTE_B4  700
#define NOTE_C5  800
/*
#define NOTE_C4  262 // frequency of C4 note in Hz
#define NOTE_D4  294 
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
*/

/* Function to play a note on the buzzer */
// note : freq(=W), T = 1/W
// duration : time in msec
// period = 10**6 / note = T (in usec)
int swc(int gpio){
		int fd;
		char buf[BUFSIZE];
		char inCh;
		
		fd = open("/sys/class/gpio/export",O_WRONLY);
		sprintf(buf,"%d",gpio);
		write(fd,buf,strlen(buf));
		close(fd);
		
		sprintf(buf,"/sys/class/gpio/gpio%d/direction",gpio);
		fd = open(buf,O_WRONLY);
		write(fd,"in",3);
		close(fd);
		
		sprintf(buf,"/sys/class/gpio/gpio%d/value",gpio);
		fd = open(buf,O_RDONLY); //WRONLY = write
		
		read(fd,&inCh,1);
		
		
		//for(int i = 0 ; i < mode ; i ++)
		//{
		//	write(fd,"1",2);
		//	sleep(1);
		//	write(fd,"0",2);
		//	sleep(1);
		//}
		close(fd);

			
		fd = open("/sys/class/gpio/unexport",O_WRONLY);
		sprintf(buf,"%d",gpio);
		write(fd,buf,strlen(buf));
		close(fd);
		
		return (inCh-'0');
}
void playNote(int note, int duration) {
  int period = 1000000 / note;
  int halfPeriod = period / 2;
  int cycles = duration * note / 1000;
  for (int i = 0; i < cycles; i++) { // == cycles * T == duration * 1000 (in usec)
	gpioWrite(BUZZER_PIN, 1);
	usleep(halfPeriod); // T/2
	gpioWrite(BUZZER_PIN, 0);
	usleep(halfPeriod); // T/2
  }
}

/* Function to play a melody */
void playMelody(int notes, int durations, int numNotes) {
  for (int i = 0; i < numNotes; i++) {
    playNote(notes, durations);
    usleep(500000); /* pause between notes to avoid overlapping */
  }
}


int main(int argc, char **argv){
	char a;
	int gpio = BUZZER_PIN;
	gpioExport(gpio);
	gpioDirection(gpio, 1) ; // "out"
	
	while(1){
	a =getchar();
	if(a == 'q'){
		gpioUnexport(gpio);
		break;
		}
	
	if( a == 'a')
	{
		playMelody(NOTE_C4, 100, 1);
	}
	else if( a == 's')
	{
		playMelody(NOTE_D4, 100, 1);
	}
	else if( a == 'd')
	{
		playMelody(NOTE_E4, 100, 1);
	}
	else if( a == 'f')
	{
		playMelody(NOTE_F4, 100, 1);
	}
	else if( a == 'g')
	{
		playMelody(NOTE_G4, 100, 1);
	}

	else if( a == 'h')
	{
		playMelody(NOTE_A4, 100, 1);
	}
	else if( a == 'j')
	{
		playMelody(NOTE_B4, 100, 1);
	}
	else if( a == 'k')
	{
		playMelody(NOTE_C5, 100, 1);
	}
}
return 0;
}
