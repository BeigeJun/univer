#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "gpiolib.c"
#define BH1750_ADDR 0x23 // Replace with the actual I2C address of the sensor
#define BUFSIZE 256
int ledControl(int gpio, int mode){
		int fd;
		char buf[BUFSIZE];
		
		fd = open("/sys/class/gpio/export",O_WRONLY);
		sprintf(buf,"%d",gpio);
		write(fd,buf,strlen(buf));
		close(fd);
		
		sprintf(buf,"/sys/class/gpio/gpio%d/direction",gpio);
		fd = open(buf,O_WRONLY);
		write(fd,"out",4);
		close(fd);
		
		sprintf(buf,"/sys/class/gpio/gpio%d/value",gpio);
		fd = open(buf,O_WRONLY); //WRONLY = write
		
		write(fd,"in",3);
		
		if(mode == 0)
			write(fd,"0",2);
		else
			write(fd,"1",2);
		close(fd);

			
		fd = open("/sys/class/gpio/unexport",O_WRONLY);
		sprintf(buf,"%d",gpio);
		write(fd,buf,strlen(buf));
		close(fd);
		
		return 0;
}

int main()
{
    int fd;
        int     luxRaw;
        float   lux;
    unsigned char buf[3] = {0};
    
    
    while(1){
    if ((fd = open("/dev/i2c-1", O_RDWR)) < 0) { // Open the I2C device file
        perror("open");
        exit(1);
    }

    if (ioctl(fd, I2C_SLAVE, BH1750_ADDR) < 0) { // Set the I2C slave address
        perror("ioctl");
        exit(1);
    }


    // Send measurement request to the sensor
    //buf[0] = 0x01; // Power On
    //buf[0] = 0x07; // Reset
    buf[0] = 0x23; // One time L-Resolution measurement
    if (write(fd, buf, 1) != 1) {
        perror("write");
        exit(1);
    }
    // Wait for measurement to be ready (typically takes 120ms)
    usleep(120000);

    // Read the measurement value from the sensor
    if (read(fd, buf, 2) != 2) {
        perror("read");
        exit(1);
    }
    // Convert the measurement value to lux
    luxRaw = (buf[0] << 8) | buf[1];
    lux = (float)luxRaw / 1.2; // Divide by 1.2 to get the actual lux value
    printf("Lux: %d(%7.3f) :: %x, %x\n", luxRaw, lux, buf[0], buf[1]);
    close(fd);
    
    if( luxRaw > 50)
    {
		ledControl(21,1);
	}
	else
	{
		ledControl(21,0);
	}
}
    return 0;

}
